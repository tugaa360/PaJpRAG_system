import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
import logging
import json
import pickle
from typing import List, Optional, Dict, Union, TypedDict, Tuple
from pathlib import Path
import hashlib
from datetime import datetime
import uuid
import os
import shutil

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryResult(TypedDict):
    llm_prompt: str # LLMに渡すための整形されたプロンプト
    retrieved_documents: List[str] # 検索された生文書
    search_time: float # 検索にかかった時間
    details: List[Dict[str, Union[str, float, int]]] # 各検索結果の詳細情報

class RAGSystem:
    """
    Retrieval Augmented Generation (RAG) システムを実装するクラス。
    SentenceTransformerを用いて文書のエンベディングを生成し、FAISSで効率的な類似性検索を行う。
    """

    DEFAULT_MODEL_NAME = 'all-MiniLM-L6-v2'
    DEFAULT_INDEX_TYPE = 'ivf'
    DEFAULT_N_CLUSTERS = 100 # IVFインデックスのデフォルトクラスタ数
    DEFAULT_NPROBE_RATIO = 0.1 # クラスタ数の10%をnprobeとしてデフォルト設定
    DEFAULT_INDEX_BASE_DIR = Path('rag_data') # インデックスを保存するベースディレクトリ
    DEFAULT_BATCH_SIZE = 64 # エンベディング生成時のデフォルトバッチサイズ
    MIN_DOCS_FOR_IVF = 100 # IVFインデックスを推奨する最小文書数

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, 
                 index_type: str = DEFAULT_INDEX_TYPE, 
                 n_clusters: int = DEFAULT_N_CLUSTERS, 
                 nprobe: Optional[int] = None, 
                 index_base_dir: Union[str, Path] = DEFAULT_INDEX_BASE_DIR,
                 load_latest: bool = True):
        """
        RAGシステムの初期化
        Args:
            model_name (str): SentenceTransformerのモデル名
            index_type (str): FAISSインデックスのタイプ（'flat'または'ivf'）
            n_clusters (int): IndexIVFFlat使用時のクラスタ数 (IVF選択時のみ有効)
            nprobe (int, optional): IVF検索時の探索クラスタ数。未指定時はクラスタ数の10%
            index_base_dir (Union[str, Path]): インデックスや文書データを保存するベースディレクトリ
            load_latest (bool): Trueの場合、指定されたベースディレクトリ内の最新のインデックスと文書を自動的にロードする。
                                Falseの場合、新しいインデックスディレクトリを作成する。
        """
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.index_type = index_type.lower()
        self.n_clusters = n_clusters
        self.nprobe = nprobe
        self.documents: List[str] = []
        self.dimension: Optional[int] = None
        self.documents_hash: Optional[str] = None
        
        if self.index_type not in ['flat', 'ivf']:
            raise ValueError(f"サポートされていないインデックスタイプです: {index_type}。'flat'または'ivf'である必要があります。")
        
        self.index_base_dir = Path(index_base_dir)
        self.current_index_dir: Optional[Path] = None
        self.index_path: Optional[Path] = None
        self.documents_path: Optional[Path] = None
        self.metadata_path: Optional[Path] = None

        # モデルの初期化
        self._initialize_model(model_name)

        if load_latest:
            self.load_latest_state()
        else:
            # 新しいインデックスディレクトリを作成
            self._create_new_index_dir()

    def _initialize_model(self, model_name: str) -> None:
        """SentenceTransformerモデルの初期化"""
        try:
            logger.info("埋め込みモデルをロード中: %s", model_name)
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info("モデルのロードに成功しました。埋め込み次元数: %d", self.dimension)
        except Exception as e:
            logger.exception("モデル '%s' のロードに失敗しました: %s", model_name, e)
            raise RuntimeError(f"モデルの初期化に失敗しました: {e}") from e

    def _create_new_index_dir(self) -> None:
        """新しいユニークなインデックスディレクトリを作成し、パスを設定する"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        self.current_index_dir = self.index_base_dir / f"{timestamp}_{unique_id}"
        self.current_index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.current_index_dir / 'faiss_index.bin'
        self.documents_path = self.current_index_dir / 'documents.json'
        self.metadata_path = self.current_index_dir / 'metadata.json'
        logger.info("新しいRAGデータディレクトリを作成しました: %s", self.current_index_dir)

    def _find_latest_index_dir(self) -> Optional[Path]:
        """ベースディレクトリ内で最新のインデックスディレクトリを見つける"""
        if not self.index_base_dir.exists():
            return None
        
        subdirs = []
        for d in self.index_base_dir.iterdir():
            if d.is_dir() and len(d.name) >= 15: # YYYYMMDD_HHMMSS_* 形式を期待
                try:
                    datetime.strptime(d.name[:15], '%Y%m%d_%H%M%S')
                    subdirs.append(d)
                except ValueError:
                    continue
        
        if not subdirs:
            return None
        
        # 最新のディレクトリ（作成日時でソート）
        subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return subdirs[0]

    def load_latest_state(self) -> bool:
        """最新のインデックスと文書データを自動的にロードする"""
        latest_dir = self._find_latest_index_dir()
        if latest_dir:
            self.current_index_dir = latest_dir
            self.index_path = self.current_index_dir / 'faiss_index.bin'
            self.documents_path = self.current_index_dir / 'documents.json'
            self.metadata_path = self.current_index_dir / 'metadata.json'
            logger.info("最新の状態をロードしようとしています: %s", self.current_index_dir)
            
            try:
                self.load_documents() # デフォルトパスから文書をロード
                self.load_index() # デフォルトパスからインデックスをロード
                return True
            except (FileNotFoundError, RuntimeError) as e:
                logger.warning("最新の状態を %s から完全にロードできませんでした: %s。新しい状態を作成します。", latest_dir, e)
                self._create_new_index_dir()
                return False
        else:
            logger.info("既存のRAGデータディレクトリが見つかりませんでした。新しい状態を作成します。")
            self._create_new_index_dir()
            return False

    def _calculate_documents_hash(self, documents: List[str]) -> str:
        """文書リストのハッシュ値を計算"""
        if not documents:
            return ""
        # 安定したハッシュのために、ソートしてからエンコード
        return hashlib.sha256(
            json.dumps(sorted(documents), sort_keys=True, ensure_ascii=False).encode('utf-8')
        ).hexdigest()

    def _save_metadata(self) -> None:
        """メタデータを保存"""
        if not self.current_index_dir or not self.metadata_path:
            logger.error("current_index_dirまたはmetadata_pathが設定されていません。メタデータを保存できません。")
            return

        metadata = {
            'index_type': self.index_type,
            'n_clusters': self.n_clusters,
            'nprobe': self.nprobe,
            'dimension': self.dimension,
            'documents_count': len(self.documents),
            'documents_hash': self.documents_hash,
            'created_at': datetime.now().isoformat(),
            'model_name': self.model.name_or_path if self.model else "unknown"
        }
        try:
            with self.metadata_path.open('w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.debug("メタデータを %s に保存しました", self.metadata_path)
        except Exception as e:
            logger.warning("メタデータを %s に保存できませんでした: %s", self.metadata_path, e)

    def _load_metadata(self) -> Dict:
        """メタデータを読み込み"""
        if not self.metadata_path or not self.metadata_path.exists():
            return {}
        try:
            with self.metadata_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning("メタデータを %s からロードできませんでした: %s", self.metadata_path, e)
        return {}

    def load_documents(self, file_path: Optional[Union[str, Path]] = None) -> None:
        """
        外部JSONファイルから文書データを読み込むか、現在のディレクトリの文書ファイルをロードする。
        Args:
            file_path (Union[str, Path], optional): 文書データを含むJSONファイルのパス。
                                                    指定がない場合は現在のRAGデータディレクトリのdocuments.jsonから読み込む。
        """
        target_path = Path(file_path) if file_path else self.documents_path
        
        if not target_path:
            logger.warning("ドキュメントファイルのパスが提供されておらず、current_index_dirも設定されていません。文書をロードできません。")
            return

        if not target_path.exists():
            if file_path: # 明示的にファイルパスが指定されたのに見つからない場合
                raise FileNotFoundError(f"ドキュメントファイルが見つかりません: {target_path}")
            else: # デフォルトパスが見つからない場合
                logger.info("%s に documents.json が見つかりませんでした。文書はロードされません。", target_path)
                self.documents = []
                self.documents_hash = ""
                return
            
        try:
            with target_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict) or 'documents' not in data:
                raise ValueError("JSONファイルは 'documents' キーを持ち、その値は文字列のリストである必要があります。")
            
            documents_from_file = data['documents']
            if not isinstance(documents_from_file, list):
                raise ValueError("'documents' はリストである必要があります。")
            
            valid_documents = []
            for i, doc in enumerate(documents_from_file):
                if isinstance(doc, str) and doc.strip():
                    valid_documents.append(doc.strip())
                elif isinstance(doc, str):
                    logger.warning("ファイル %s のインデックス %d にある空の文書をスキップします。", target_path, i)
                else:
                    logger.warning("ファイル %s のインデックス %d にある非文字列の文書をスキップします: %s", target_path, i, type(doc))
            
            if not valid_documents:
                raise ValueError(f"ファイル {target_path} に有効な文書が見つかりませんでした。")
            
            self.documents = valid_documents
            self.documents_hash = self._calculate_documents_hash(self.documents)
            
            # もし外部ファイルから読み込んだ場合、現在のRAGデータディレクトリに保存し直す
            if file_path and self.documents_path and Path(file_path).resolve() != self.documents_path.resolve():
                with self.documents_path.open('w', encoding='utf-8') as f:
                    json.dump({'documents': self.documents}, f, ensure_ascii=False, indent=2)
                logger.info("文書を '%s' から '%s' にコピーしました (%d件)", file_path, self.documents_path, len(self.documents))
            
            logger.info("'%s' から %d 件の有効な文書をロードしました。", target_path, len(self.documents))
            
        except json.JSONDecodeError as e:
            logger.exception("ファイル '%s' のJSON形式が不正です: %s", target_path, e)
            raise ValueError(f"不正なJSONファイルです: {target_path}") from e
        except Exception as e:
            logger.exception("ファイル '%s' から文書をロードできませんでした: %s", target_path, e)
            raise

    def build_index(self, batch_size: int = DEFAULT_BATCH_SIZE, force_rebuild: bool = False) -> None:
        """
        現在ロードされている文書のエンベディングを生成し、FAISSインデックスを構築する。
        Args:
            batch_size (int): エンベディング生成時のバッチサイズ。
            force_rebuild (bool): Trueの場合、既存のインデックスを強制的に再構築。
        """
        if not self.documents:
            raise ValueError("インデックス作成用の文書がありません。まず load_documents() を使用して文書をロードしてください。")
        
        if not self.current_index_dir or not self.index_path or not self.metadata_path:
            raise RuntimeError("RAGシステムのディレクトリが初期化されていません。load_latest_state()を呼び出すか、load_latest=Falseで初期化してください。")

        if self.model is None or self.dimension is None:
            raise RuntimeError("SentenceTransformerモデルが初期化されていません。")

        # バッチサイズの検証
        if batch_size <= 0:
            logger.warning("batch_size は正の値である必要があります。デフォルト値 %d を使用します。", self.DEFAULT_BATCH_SIZE)
            batch_size = self.DEFAULT_BATCH_SIZE
        batch_size = min(batch_size, len(self.documents))

        # 既存インデックスのチェック
        if self.index_path.exists() and not force_rebuild:
            metadata = self._load_metadata()
            # ハッシュだけでなく、次元数、インデックスタイプ、クラスタ数も一致するか確認
            if metadata.get('documents_hash') == self.documents_hash and \
               metadata.get('dimension') == self.dimension and \
               metadata.get('index_type') == self.index_type and \
               metadata.get('n_clusters') == self.n_clusters:
                try:
                    self.load_index()
                    logger.info("既存のインデックスが文書ハッシュとパラメータに一致しました。force_rebuild=True を使用して再構築できます。")
                    return
                except Exception as e:
                    logger.warning("既存のインデックスのロードに失敗しました: %s。再構築します。", e)
            else:
                logger.info("文書ハッシュの不一致、またはメタデータパラメータの変更がありました。インデックスを再構築します。")

        try:
            logger.info("%d 個の文書に対して埋め込みを作成中 (バッチサイズ: %d)", len(self.documents), batch_size)
            
            embeddings = self.model.encode(
                self.documents,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_tensor=False,
                normalize_embeddings=True # コサイン類似度ベースの検索には必須
            ).astype(np.float32)

            # インデックスの初期化
            if self.index_type == 'flat' or len(embeddings) < self.MIN_DOCS_FOR_IVF:
                if self.index_type == 'ivf' and len(embeddings) < self.MIN_DOCS_FOR_IVF:
                    logger.warning("IVFインデックスには文書数が少なすぎます (%d 件、最小: %d 件)。Flatインデックスにフォールバックします。", 
                                   len(embeddings), self.MIN_DOCS_FOR_IVF)
                    self.index_type = 'flat'
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info("FAISS IndexFlatL2 を次元数 %d で初期化しました。", self.dimension)
                
            elif self.index_type == 'ivf':
                # クラスタ数の調整: 設定されたn_clustersと、文書数/5（経験値）の小さい方を採用。
                # ただし、クラスタ数は文書総数以下である必要があり、最低1つ。
                optimal_n_clusters = min(self.n_clusters, max(1, len(embeddings) // 5, 1)) # 最低1クラスタを保証
                optimal_n_clusters = min(optimal_n_clusters, len(embeddings)) # 文書総数を超えないように
                
                if optimal_n_clusters <= 1:
                    logger.warning("IVFの最適なクラスタ数 (%d) が少なすぎます。Flatインデックスにフォールバックします。", optimal_n_clusters)
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self.index_type = 'flat'
                else:
                    self.n_clusters = optimal_n_clusters
                    quantizer = faiss.IndexFlatL2(self.dimension)
                    self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.n_clusters, faiss.METRIC_L2)
                    
                    if not self.index.is_trained: # 訓練済みでない場合のみ訓練
                        logger.info("FAISS IndexIVFFlat を %d 個のクラスタで訓練中 (%d 個の文書を使用)", 
                                   self.n_clusters, len(embeddings))
                        self.index.train(embeddings)
                    
                    # nprobeの設定: デフォルトはクラスタ数の10%。ただし最低1、最大でクラスタ数まで。
                    # 文書総数も考慮に入れる
                    self.nprobe = self.nprobe or max(1, min(int(self.n_clusters * self.DEFAULT_NPROBE_RATIO), self.n_clusters))
                    self.nprobe = min(self.nprobe, self.n_clusters) # nprobeはn_clustersを超えてはならない
                    
                    self.index.nprobe = self.nprobe
                    logger.info("nprobe を %d に設定しました。", self.nprobe)

            # エンベディングの追加
            if self.index:
                self.index.add(embeddings)
                logger.info("FAISSインデックスが正常に構築されました (%d 個のベクトル, タイプ: %s)", 
                           self.index.ntotal, self.index_type)
            else:
                raise RuntimeError("FAISSインデックスの初期化に失敗しました。")

            # インデックスとメタデータの保存
            faiss.write_index(self.index, str(self.index_path))
            self._save_metadata()
            logger.info("インデックスとメタデータを %s に保存しました。", self.current_index_dir)
            
        except Exception as e:
            logger.exception("インデックスの構築に失敗しました: %s", e)
            raise RuntimeError(f"インデックスの構築に失敗しました: {e}") from e

    def load_index(self) -> None:
        """
        現在のRAGデータディレクトリに保存済みのFAISSインデックスを読み込む。
        """
        if not self.index_path or not self.index_path.exists():
            raise FileNotFoundError(f"インデックスファイルが見つかりません: {self.index_path}。まずインデックスを構築してください。")
        
        try:
            self.index = faiss.read_index(str(self.index_path))
            
            # メタデータの読み込みと適用
            metadata = self._load_metadata()
            if metadata:
                self.index_type = metadata.get('index_type', self.index_type)
                self.n_clusters = metadata.get('n_clusters', self.n_clusters)
                self.nprobe = metadata.get('nprobe', self.nprobe)
                self.documents_hash = metadata.get('documents_hash', self.documents_hash)
                
                # IVFインデックスの場合、nprobeを再設定
                if self.index_type == 'ivf' and hasattr(self.index, 'nprobe'):
                    self.index.nprobe = self.nprobe
            
            logger.info("%s からFAISSインデックスをロードしました (タイプ: %s, ベクトル数: %d)", 
                       self.index_path, self.index_type, self.index.ntotal)
            
        except Exception as e:
            logger.exception("ファイル '%s' からインデックスをロードできませんでした: %s", self.index_path, e)
            raise RuntimeError(f"インデックスのロードに失敗しました: {e}") from e

    def clean_old_indices(self, keep_latest_n: int = 5) -> None:
        """
        古いインデックスディレクトリをクリーンアップ。
        Args:
            keep_latest_n (int): 保持する最新のディレクトリ数。
        """
        if keep_latest_n <= 0:
            logger.warning("keep_latest_n は正の値である必要があります。クリーンアップをスキップします。")
            return
        
        if not self.current_index_dir:
            logger.warning("現在のインデックスディレクトリが設定されていません。クリーンアップをスキップします。")
            return

        try:
            if not self.index_base_dir.exists():
                logger.warning("ベースディレクトリが存在しません: %s", self.index_base_dir)
                return
            
            # タイムスタンプ形式のディレクトリを取得
            subdirs = []
            for d in self.index_base_dir.iterdir():
                if d.is_dir() and len(d.name) >= 15:  # YYYYMMDD_HHMMSS_* 形式を期待
                    try:
                        datetime.strptime(d.name[:15], '%Y%m%d_%H%M%S')
                        subdirs.append(d)
                    except ValueError:
                        continue
            
            # 作成日時でソート（新しい順）
            subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 古いディレクトリを削除
            removed_count = 0
            for old_dir in subdirs[keep_latest_n:]:
                if old_dir != self.current_index_dir: # 現在使用中のディレクトリは削除しない
                    try:
                        shutil.rmtree(old_dir)
                        removed_count += 1
                        logger.info("古いインデックスディレクトリを削除しました: %s", old_dir)
                    except Exception as e:
                        logger.warning("古いインデックスディレクトリ %s の削除に失敗しました: %s", old_dir, e)
            
            if removed_count > 0:
                logger.info("%s 内の古いインデックスディレクトリを %d 件クリーンアップしました。", self.index_base_dir, removed_count)
            else:
                logger.info("%s にクリーンアップする古いディレクトリはありませんでした。", self.index_base_dir)
                
        except Exception as e:
            logger.exception("%s 内の古いインデックスのクリーンアップに失敗しました: %s", self.index_base_dir, e)

    def query(self, query_text: str, top_k: int = 2, similarity_threshold: Optional[float] = None) -> QueryResult:
        """
        クエリに対するRAG検索とプロンプト生成。
        Args:
            query_text (str): 検索クエリ。
            top_k (int): 取得する文書の数。
            similarity_threshold (float, optional): コサイン類似度の閾値 (0.0から1.0)。
                                                   この値より低い類似度の文書は除外される。
        Returns:
            QueryResult: プロンプト、取得文書、検索時間、詳細を含む辞書。
        """
        # 入力検証
        if not query_text or not query_text.strip():
            logger.warning("空または空白のみのクエリが提供されました。")
            return {
                "llm_prompt": "質問が入力されていません。有効な質問を入力してください。",
                "retrieved_documents": [],
                "search_time": 0.0,
                "details": []
            }
        
        query_text = query_text.strip()
        
        if self.index is None or self.index.ntotal == 0:
            logger.error("FAISSインデックスが初期化されていないか、空です。")
            return {
                "llm_prompt": "エラー: インデックスが初期化されていないか、文書がインデックス化されていません。文書をロードしてインデックスを構築してください。",
                "retrieved_documents": [],
                "search_time": 0.0,
                "details": []
            }
        
        if not self.documents:
            logger.error("文書がロードされていません。")
            return {
                "llm_prompt": "エラー: 文書がロードされていません。ロードされた文書がないと検索できません。",
                "retrieved_documents": [],
                "search_time": 0.0,
                "details": []
            }
        
        if top_k <= 0:
            logger.warning("top_k は正の値である必要があります。1に設定します。")
            top_k = 1
        
        top_k = min(top_k, self.index.ntotal) # 利用可能なインデックスの総数に制限

        if self.model is None:
            raise RuntimeError("SentenceTransformerモデルが初期化されていません。")

        try:
            query_embedding = self.model.encode(
                [query_text],
                convert_to_tensor=False,
                normalize_embeddings=True # 正規化された埋め込みを使用
            ).astype(np.float32)
            
            start_time = time.time()
            # FAISSはL2距離で検索するため、結果はL2距離
            distances, indices = self.index.search(query_embedding, top_k)
            search_time = time.time() - start_time
            
            logger.info("FAISS検索が %.4f 秒で完了しました。クエリ: '%.50s...'", 
                       search_time, query_text)

            retrieved_docs_text: List[str] = []
            retrieval_details: List[Dict[str, Union[str, float, int]]] = []

            for i, (l2_dist, idx) in enumerate(zip(distances[0], indices[0])):
                if not (0 <= idx < len(self.documents)):
                    logger.warning("FAISSによって返された無効なインデックス %d (総文書数: %d)。スキップします。", 
                                 idx, len(self.documents))
                    continue
                
                # 正規化された埋め込みの場合、L2距離 d とコサイン類似度 s の関係は d^2 = 2 - 2s
                # よって、s = 1 - d^2 / 2
                cosine_sim = 1 - (l2_dist**2) / 2.0 

                if similarity_threshold is not None and cosine_sim < similarity_threshold:
                    logger.debug("文書 #%d はフィルタリングされました (コサイン類似度: %.4f < %.4f)", 
                               i+1, cosine_sim, similarity_threshold)
                    continue
                
                doc_text = self.documents[idx]
                retrieved_docs_text.append(doc_text)
                retrieval_details.append({
                    "document": doc_text,
                    "l2_distance": float(l2_dist),
                    "cosine_similarity": float(cosine_sim),
                    "original_index": int(idx),
                    "rank": i + 1
                })
                
                logger.debug("取得した文書 #%d (インデックス: %d, ランク: %d): '%.50s...' (L2: %.4f, CosSim: %.4f)",
                           i+1, idx, i+1, doc_text, l2_dist, cosine_sim)

            if not retrieved_docs_text:
                logger.info("クエリ '%.50s...' に関連する文書は見つかりませんでした (top_k=%d, 類似度閾値=%.2f)", 
                            query_text, top_k, similarity_threshold or 0.0)
                llm_prompt = (f"質問: {query_text}\n\n回答: 申し訳ございませんが、関連する情報が見つかりませんでした。"
                              "別の表現で質問していただくか、より具体的な内容で質問してください。")
                return {
                    "llm_prompt": llm_prompt,
                    "retrieved_documents": [],
                    "search_time": search_time,
                    "details": []
                }

            context = "\n\n".join([f"[文書{i+1}]\n{doc}" for i, doc in enumerate(retrieved_docs_text)])
            llm_prompt = f"""以下の関連文書を参考にして、質問に正確かつ詳細に答えてください。

=== 関連文書 ===
{context}

=== 質問 ===
{query_text}

=== 回答 ===
上記の文書に基づいて回答します："""

            logger.info("クエリに対するLLMプロンプトを生成しました (取得文書数: %d): '%.50s...'", 
                       len(retrieved_docs_text), query_text)
            
            return {
                "llm_prompt": llm_prompt,
                "retrieved_documents": retrieved_docs_text,
                "search_time": search_time,
                "details": retrieval_details
            }
            
        except Exception as e:
            logger.exception("クエリ処理中にエラーが発生しました: '%.50s...'", query_text)
            return {
                "llm_prompt": f"エラー: クエリ処理中に問題が発生しました - {str(e)}",
                "retrieved_documents": [],
                "search_time": 0.0,
                "details": []
            }

    def get_stats(self) -> Dict[str, Union[int, str, float]]:
        """システムの統計情報を取得"""
        stats: Dict[str, Union[int, str, float]] = {
            "documents_count": len(self.documents),
            "index_type": self.index_type,
            "dimension": self.dimension if self.dimension is not None else 0,
            "index_total_vectors": self.index.ntotal if self.index else 0,
            "current_index_dir": str(self.current_index_dir) if self.current_index_dir else "N/A"
        }
        
        if self.index_type == 'ivf':
            stats.update({
                "n_clusters": self.n_clusters,
                "nprobe": self.nprobe if self.nprobe is not None else "N/A"
            })
        
        if self.index_path and self.index_path.exists():
            stats["index_file_size_mb"] = round(self.index_path.stat().st_size / (1024 * 1024), 2)
        
        return stats

if __name__ == "__main__":
    # サンプル文書データ
    sample_documents = [
        "RAG（Retrieval Augmented Generation）は、大規模言語モデルの課題、特に幻覚や情報鮮度の問題を解決するために考案された強力なAIフレームワークです。このシステムでは、外部のデータベースから関連情報を検索し、それを基に回答を生成します。",
        "LLMの幻覚（Hallucination）は、大規模言語モデルが事実と異なる情報を生成してしまう問題であり、RAGのようなフレームワークがその対策として注目されています。幻覚は訓練データにない情報や、古い情報に基づく回答で発生しやすいです。",
        "ベクトルデータベースは、高次元のベクトルデータを効率的に保存、管理、検索するための特殊なデータベースです。近似最近傍探索（ANN）アルゴリズムを使用して、類似度の高いベクトルを高速に検索できます。",
        "Faissは、Facebook AIが開発したオープンソースの効率的な類似性検索ライブラリで、特に大規模なベクトルデータセットにおいて高速な検索を実現します。CPU版とGPU版があり、様々なインデックス構造をサポートしています。",
        "プロンプトエンジニアリングは、大規模言語モデルから最適な出力を得るための技術であり、質問の仕方や文脈の与え方が重要です。適切なプロンプト設計により、モデルの性能を大幅に向上させることができます。",
        "RAGシステムの実運用における課題としては、計算コストとレイテンシ（応答遅延）が挙げられます。リアルタイム性が求められるアプリケーションでは、検索速度とエンベディング生成速度の最適化が重要になります。",
        "インデックスの最適化は、検索速度を向上させるための重要なステップです。適切なインデックス構造の選択、パラメータチューニング、データの前処理によって大幅な性能改善が期待できます。",
        "セマンティック検索は、意味的な類似性に基づいて情報を検索する技術です。従来のキーワードベースの検索とは異なり、文脈や意図を理解して関連度の高い結果を返すことができます。",
        "モデルの軽量化や量子化は、推論速度とメモリ使用量を改善する重要な技術です。精度を保ちながらモデルサイズを削減することで、リソース制約のある環境でも効率的に動作させることができます。",
        "キャッシュは、頻繁にアクセスされるデータを一時的に保存し、高速なアクセスを可能にする仕組みです。RAGシステムでは、エンベディングキャッシュや検索結果キャッシュによって応答速度を大幅に改善できます。",
        "自然言語処理（NLP）は、コンピュータが人間の言語を理解、解釈、生成する能力を研究するAIの分野です。機械翻訳やテキスト要約などが含まれます。",
        "トランスフォーマーモデルは、自然言語処理の分野で大きな進歩をもたらしたニューラルネットワークアーキテクチャです。Attention機構が特徴です。",
        "強化学習は、エージェントが環境と相互作用し、試行錯誤を通じて最適な行動戦略を学習する機械学習の一分野です。報酬を最大化するように学習します。",
        "教師あり学習は、入力データとそれに対応する正解ラベルのペアを用いてモデルを訓練する機械学習の最も一般的な手法です。分類や回帰問題に用いられます。",
        "非教師あり学習は、ラベルなしのデータからパターンや構造を自動的に見つけ出す機械学習の手法です。クラスタリングや次元削減などが含まれます。"
    ]

    # テスト用ディレクトリの作成
    test_base_dir = Path('test_rag_systems')
    if test_base_dir.exists():
        shutil.rmtree(test_base_dir) # 以前のテストデータを削除
    test_base_dir.mkdir(parents=True, exist_ok=True)
    
    # サンプル文書をJSONファイルに保存
    # 複数回使用することを想定し、固定のファイル名にする
    test_doc_file = test_base_dir / 'shared_sample_documents.json'
    with test_doc_file.open('w', encoding='utf-8') as f:
        json.dump({'documents': sample_documents}, f, ensure_ascii=False, indent=2)

    print("=== RAG System Test ===")

    # ケース1: 新しいRAGシステムを作成し、文書をロードしてインデックスを構築
    print("\n--- ケース1: 新しいIVFインデックスの構築 ---")
    try:
        rag_ivf_1 = RAGSystem(
            model_name='all-MiniLM-L6-v2', 
            index_type='ivf', 
            n_clusters=50, # 文書数が少ないので、実際には調整される
            index_base_dir=test_base_dir, 
            load_latest=False # 新しいディレクトリを作成
        )
        rag_ivf_1.load_documents(test_doc_file)
        rag_ivf_1.build_index()
        stats_1 = rag_ivf_1.get_stats()
        print(f"構築後の統計1: {stats_1}")

        query_result_1 = rag_ivf_1.query("RAGシステムの主な利点は何ですか？", top_k=3)
        print("\nクエリ1 (IVF):")
        print(f"LLMプロンプト:\n{query_result_1['llm_prompt']}")
        print(f"取得文書: {[d[:50] + '...' for d in query_result_1['retrieved_documents']]}")
        print(f"検索時間: {query_result_1['search_time']:.4f}秒")
        print(f"詳細: {query_result_1['details']}")

    except Exception as e:
        logger.error(f"ケース1でエラーが発生しました: {e}")

    # ケース2: 同じベースディレクトリで新しいRAGシステムを作成し、最新の状態をロード
    print("\n--- ケース2: 最新インデックスのロード ---")
    try:
        rag_ivf_2 = RAGSystem(
            model_name='all-MiniLM-L6-v2', 
            index_type='ivf', # この設定はロードされるメタデータで上書きされる可能性あり
            n_clusters=100, 
            index_base_dir=test_base_dir, 
            load_latest=True # 最新のディレクトリをロード
        )
        stats_2 = rag_ivf_2.get_stats()
        print(f"ロード後の統計2: {stats_2}")
        
        query_result_2 = rag_ivf_2.query("LLMの幻覚問題を解決する方法は？", top_k=2, similarity_threshold=0.8)
        print("\nクエリ2 (ロードされたIVF):")
        print(f"LLMプロンプト:\n{query_result_2['llm_prompt']}")
        print(f"取得文書: {[d[:50] + '...' for d in query_result_2['retrieved_documents']]}")
        print(f"検索時間: {query_result_2['search_time']:.4f}秒")
        print(f"詳細: {query_result_2['details']}")
    except Exception as e:
        logger.error(f"ケース2でエラーが発生しました: {e}")

    # ケース3: Flatインデックスのテスト（新しいディレクトリで）
    print("\n--- ケース3: 新しいFlatインデックスの構築 ---")
    try:
        rag_flat = RAGSystem(
            model_name='all-MiniLM-L6-v2', 
            index_type='flat', 
            index_base_dir=test_base_dir, 
            load_latest=False # 新しいディレクトリを作成
        )
        rag_flat.load_documents(test_doc_file)
        rag_flat.build_index()
        stats_3 = rag_flat.get_stats()
        print(f"構築後の統計3: {stats_3}")

        query_result_3 = rag_flat.query("ベクトルデータベースとは何ですか？", top_k=1)
        print("\nクエリ3 (Flat):")
        print(f"LLMプロンプト:\n{query_result_3['llm_prompt']}")
        print(f"取得文書: {[d[:50] + '...' for d in query_result_3['retrieved_documents']]}")
        print(f"検索時間: {query_result_3['search_time']:.4f}秒")
        print(f"詳細: {query_result_3['details']}")
    except Exception as e:
        logger.error(f"ケース3でエラーが発生しました: {e}")

    # ケース4: 存在しない文書ファイルをロードしようとする
    print("\n--- ケース4: 存在しないドキュメントファイルのロード ---")
    try:
        rag_err = RAGSystem(index_base_dir=test_base_dir, load_latest=False)
        rag_err.load_documents(Path('non_existent_file.json'))
    except FileNotFoundError as e:
        print(f"想定されたエラーを捕捉しました: {e}")
    except Exception as e:
        logger.error(f"ケース4で予期せぬエラーが発生しました: {e}")
    
    # ケース5: 空のクエリ
    print("\n--- ケース5: 空のクエリ ---")
    try:
        # 既存のRAGSystemをロード（最新のものを自動的に選択）
        rag_existing = RAGSystem(index_base_dir=test_base_dir, load_latest=True)
        query_result_empty = rag_existing.query("")
        print(f"空のクエリ結果LLMプロンプト: {query_result_empty['llm_prompt']}")
    except Exception as e:
        logger.error(f"ケース5で予期せぬエラーが発生しました: {e}")

    # ケース6: 古いインデックスのクリーンアップ
    print("\n--- ケース6: 古いインデックスのクリーンアップ ---")
    try:
        # 新しいインデックスをいくつか作成し、現在のもの以外の古いものを削除するようにする
        _ = RAGSystem(index_base_dir=test_base_dir, load_latest=False) # 新しいディレクトリ1
        time.sleep(0.1) # タイムスタンプが異なるように少し待つ
        _ = RAGSystem(index_base_dir=test_base_dir, load_latest=False) # 新しいディレクトリ2
        time.sleep(0.1)
        rag_clean = RAGSystem(index_base_dir=test_base_dir, load_latest=False) # 新しいディレクトリ3 (これが最新になる)
        rag_clean.clean_old_indices(keep_latest_n=2) # 最新2つを残して削除
        print("古いインデックスがクリーンアップされました。'test_rag_systems' ディレクトリを確認してください。")
    except Exception as e:
        logger.error(f"ケース6でエラーが発生しました: {e}")

    # テスト終了後のクリーンアップ（オプション）
    # shutil.rmtree(test_base_dir)
    # print(f"\nテストディレクトリ '{test_base_dir}' を削除しました。")
