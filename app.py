import os
import shutil
import torch
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
import gradio as gr
import logging
from pathlib import Path
import json
from typing import Tuple # generate_response の型ヒントのために追加

# ragsys03.py から RAGSystem クラスと QueryResult をインポート
from ragsys03 import RAGSystem, QueryResult 

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# グローバル変数
rag_system: RAGSystem = None
llm_pipeline = None

# RAGデータ保存用のベースディレクトリ
RAG_BASE_DIR = Path('rag_data')

def initialize_systems():
    """
    RAGシステムとLLMパイプラインを初期化します。
    """
    global rag_system, llm_pipeline

    if rag_system is None:
        logger.info("RAGSystemを初期化中...")
        try:
            # 最新のインデックスをロードするか、新しいディレクトリを作成する
            rag_system = RAGSystem(
                model_name='all-MiniLM-L6-v2', # SentenceTransformer モデル
                index_type='ivf', # または 'flat'
                n_clusters=100, # IVFの場合のクラスタ数
                index_base_dir=RAG_BASE_DIR,
                load_latest=True # 最新のインデックスを自動的にロード
            )
            logger.info("RAGSystemの初期化が完了しました。")
            
            # 初期化時にインデックスが存在しない場合、ここで構築を促す
            if rag_system.index is None or rag_system.index.ntotal == 0:
                logger.warning("RAGシステムにインデックスがありません。文書をアップロードして「インデックスを構築」ボタンを押してください。")

        except Exception as e:
            logger.critical(f"RAGSystemの初期化中にエラーが発生しました: {e}")
            rag_system = None # 初期化失敗時はNoneに設定
            # Gradio UIでエラーメッセージを表示するための処理を考慮するか、
            # 各アクションでrag_systemがNoneの場合の処理を堅牢にする
            # ここでは単にログに出力し、後続の関数でNoneチェックを行う

    if llm_pipeline is None:
        logger.info("LLMパイプラインを初期化中: rinna/japanese-gpt-neox-3.6b-instruction-sft")
        try:
            # 量子化を無効にした設定
            llm_pipeline = pipeline(
                "text-generation",
                model="rinna/japanese-gpt-neox-3.6b-instruction-sft",
                tokenizer=AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft", use_fast=False),
                torch_dtype=torch.bfloat16, # CPUでも動くようにbfloat16のまま
                device_map="auto", # autoのままでOK (GPUがなければCPUに自動的に割り当たる)
                # model_kwargs={"quantization_config": quantization_config} を削除！
            )
            logger.info("LLMパイプラインの初期化が完了しました。")
        except Exception as e:
            logger.critical(f"LLMパイプラインの初期化中にエラーが発生しました: {e}")
            llm_pipeline = None # 初期化失敗時はNoneに設定
            # 同様に、エラー発生時はllm_pipelineをNoneのままにする

def generate_response(query: str, top_k: int, similarity_threshold: float) -> Tuple[str, str]:
    """
    RAGシステムとLLMを使用して応答を生成します。
    Args:
        query (str): ユーザーからのクエリ。
        top_k (int): 検索する文書の数。
        similarity_threshold (float): コサイン類似度の閾値。
    Returns:
        Tuple[str, str]: LLMの回答と、検索された文書のテキスト。
    """
    if rag_system is None or llm_pipeline is None:
        return "システムが初期化されていません。アプリケーションを再起動してください。", ""

    logger.info(f"クエリ処理開始: {query}")
    query_result: QueryResult = rag_system.query(query, top_k=top_k, similarity_threshold=similarity_threshold)

    llm_prompt = query_result['llm_prompt']
    retrieved_docs = query_result['retrieved_documents']
    search_time = query_result['search_time']

    if not retrieved_docs:
        llm_answer = "関連する情報が見つかりませんでした。別の表現で質問していただくか、より具体的な内容で質問してください。"
        return llm_answer, "関連文書なし"

    try:
        outputs = llm_pipeline(
            llm_prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            num_return_sequences=1,
            pad_token_id=llm_pipeline.tokenizer.eos_token_id # pad_token_idを指定
        )

        llm_answer = outputs[0]['generated_text']
        # プロンプト部分を除去して回答のみを抽出 (モデルの出力形式による)
        if llm_answer.startswith(llm_prompt):
            llm_answer = llm_answer[len(llm_prompt):].strip()
        
        logger.info(f"LLM応答生成完了 (検索時間: {search_time:.4f}秒)")
        
        retrieved_docs_text_formatted = "\n\n".join([f"**文書{i+1}:** {doc}" for i, doc in enumerate(retrieved_docs)])
        
        return llm_answer, retrieved_docs_text_formatted

    except Exception as e:
        logger.error(f"LLMによる応答生成中にエラーが発生しました: {e}")
        return f"応答生成中にエラーが発生しました: {str(e)}", ""

def upload_documents(file):
    """
    アップロードされたJSONファイルから文書をロードし、インデックスを再構築します。
    """
    global rag_system

    if rag_system is None:
        # RAGSystemが未初期化の場合、ここで初期化を試みる
        # initialize_systems() は、Gradioの起動時に一度しか呼ばれないため、
        # ここでrag_systemがNoneの場合、初期化に失敗している可能性がある。
        # 代わりに、エラーメッセージを返す
        return "エラー: RAGシステムが初期化されていません。アプリケーションを再起動してください。", ""

    if file is None:
        return "エラー: ファイルがアップロードされていません。", ""

    uploaded_file_path = Path(file.name)
    logger.info(f"アップロードされたファイル: {uploaded_file_path}")

    try:
        # ragsys03.py の load_documents メソッドを使用
        rag_system.load_documents(uploaded_file_path)
        
        # インデックス構築のメッセージを表示し、ユーザーにボタンを押してもらう
        return (
            f"'{uploaded_file_path.name}' から {len(rag_system.documents)} 件の文書を正常にロードしました。\n"
            "続けて「インデックスを構築」ボタンを押してください。",
            "文書がロードされました。インデックス構築が必要です。"
        )
    except Exception as e:
        logger.error(f"文書のロード中にエラーが発生しました: {e}")
        return f"文書のロード中にエラーが発生しました: {str(e)}", ""

def build_index_action():
    """
    RAGインデックスを構築します。
    """
    global rag_system
    if rag_system is None:
        return "エラー: RAGシステムが初期化されていません。", ""

    if not rag_system.documents:
        return "エラー: インデックス作成用の文書がロードされていません。", ""

    try:
        # force_rebuild=True で常に再構築
        rag_system.build_index(force_rebuild=True) 
        stats = rag_system.get_stats()
        return (
            f"インデックスが正常に構築されました。\n"
            f"総文書数: {stats.get('documents_count', 'N/A')}\n"
            f"インデックスタイプ: {stats.get('index_type', 'N/A')}\n"
            f"インデックスディレクトリ: {stats.get('current_index_dir', 'N/A')}\n"
            f"ファイルサイズ: {stats.get('index_file_size_mb', 'N/A')} MB"
        ), "インデックス構築完了"
    except Exception as e:
        logger.error(f"インデックスの構築中にエラーが発生しました: {e}")
        return f"エラー: インデックスの構築中に問題が発生しました - {str(e)}", ""

def get_rag_stats_action():
    """RAGシステムの統計情報を表示します。"""
    global rag_system
    if rag_system is None:
        return "RAGシステムはまだ初期化されていません。"
    try:
        stats = rag_system.get_stats()
        stats_str = "\n".join([f"{k}: {v}" for k, v in stats.items()])
        return f"RAGシステムの状態:\n{stats_str}"
    except Exception as e:
        logger.error(f"RAG統計情報の取得中にエラーが発生しました: {e}")
        return f"統計情報の取得中にエラーが発生しました: {str(e)}"

# Gradio UI の構築
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # RAG (Retrieval Augmented Generation) デモ
        rinna/japanese-gpt-neox-3.6b-instruction-sft と FAISS を利用したRAGシステムです。
        JSONファイルをアップロードして独自の知識ベースを構築し、質問をすることができます。

        ## 使用方法
        1. 「文書をアップロード」セクションで、`documents` キーに文字列のリストを持つJSONファイルをアップロードします。
        2. 「インデックスを構築」ボタンをクリックして、アップロードされた文書から検索インデックスを作成します。
        3. 「RAG質問」セクションで質問を入力し、「質問を送信」ボタンをクリックします。
        """
    )

    with gr.Tab("RAG質問"):
        gr.Markdown("### RAG質問")
        with gr.Row():
            query_input = gr.Textbox(label="質問を入力してください", placeholder="RAGシステムの主な利点は何ですか？", lines=2)
            submit_button = gr.Button("質問を送信")
        
        with gr.Row():
            top_k_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="取得文書数 (top_k)")
            similarity_threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="類似度閾値 (0.0-1.0)")

        llm_output = gr.Textbox(label="LLMの回答", lines=10, interactive=False)
        retrieved_docs_output = gr.Markdown("検索された関連文書", visible=True)

    with gr.Tab("文書管理"):
        gr.Markdown("### 文書とインデックス管理")
        with gr.Row():
            file_upload_button = gr.File(label="文書JSONファイルをアップロード", file_types=[".json"])
            upload_status = gr.Textbox(label="アップロードステータス", interactive=False)
        
        with gr.Row():
            build_index_button = gr.Button("インデックスを構築")
            build_index_status = gr.Textbox(label="インデックス構築ステータス", interactive=False)
        
        gr.Markdown("### RAGシステム情報")
        get_stats_button = gr.Button("RAGシステムの状態を表示")
        rag_stats_output = gr.Textbox(label="RAGシステム情報", interactive=False, lines=5)


    # イベントハンドラの登録
    submit_button.click(
        fn=generate_response,
        inputs=[query_input, top_k_slider, similarity_threshold_slider],
        outputs=[llm_output, retrieved_docs_output]
    )

    file_upload_button.upload(
        fn=upload_documents,
        inputs=file_upload_button,
        outputs=[upload_status, retrieved_docs_output] # アップロード結果と関連文書表示エリアを更新
    )

    build_index_button.click(
        fn=build_index_action,
        inputs=[],
        outputs=[build_index_status, retrieved_docs_output] # 構築結果と関連文書表示エリアを更新
    )

    get_stats_button.click(
        fn=get_rag_stats_action,
        inputs=[],
        outputs=rag_stats_output
    )

# システムの初期化をアプリケーション起動時に実行
# initialize_systems() の中でエラーが発生した場合、rag_system や llm_pipeline が None になる
# これにより、後続の関数呼び出しでこれらの変数がNoneであることのチェックが機能する
initialize_systems()

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
