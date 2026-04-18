import os
import faiss
import json
import numpy as np
from typing import List, Dict

from utils.logger import get_logger

logger = get_logger(__name__)

from rank_bm25 import BM25Okapi
import pickle

class VectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.texts = []
        self.metadata = []
        self.bm25 = None
        
    def _initialize_bm25(self):
        """Initializes the BM25 index from current texts."""
        if not self.texts:
            return
        tokenized_corpus = [doc.lower().split() for doc in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def add(self, embeddings: np.ndarray, texts: List[str], metadata: List[Dict]):
        if len(embeddings) != len(texts) or len(embeddings) != len(metadata):
            logger.error("Mismatch in lengths of embeddings, texts, and metadata")
            return
            
        logger.info(f"Adding {len(embeddings)} items to FAISS index.")
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.texts.extend(texts)
        self.metadata.extend(metadata)
        self._initialize_bm25()
        
    def search(self, query_embedding: np.ndarray, query_text: str, k: int = 5) -> Dict:
        """Hybrid Search with relevance scoring for domain gating."""
        if not self.texts:
            return {"results": [], "top_distance": 99.0}

        # 1. FAISS Search (Semantic)
        faiss_k = k * 10
        distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), faiss_k)
        
        # Capture the raw best distance for domain gating
        top_distance = float(distances[0][0]) if len(distances[0]) > 0 else 99.0

        faiss_results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                faiss_results.append(idx)

        # 2. BM25 Search (Keyword)
        tokenized_query = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:faiss_k].tolist()

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        rrf_constant = 60
        
        for rank, idx in enumerate(faiss_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_constant + rank + 1)
            
        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rrf_constant + rank + 1)

        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_indices = sorted_indices[:k]
        
        final_results = []
        for idx in top_indices:
            final_results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "rrf_score": rrf_scores[idx]
            })
            
        return {
            "results": final_results,
            "top_distance": top_distance
        }
        
    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
            
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        with open(os.path.join(path, "data.json"), "w", encoding="utf-8") as f:
            json.dump({"texts": self.texts, "metadata": self.metadata}, f, ensure_ascii=False, indent=2)
            
        # Also persist BM25 via pickle or just rebuild on load (rebuilding is fine for small acts)
        logger.info(f"Vector store saved to {path}")
        
    def load(self, path: str):
        index_path = os.path.join(path, "index.faiss")
        data_path = os.path.join(path, "data.json")
        
        if not os.path.exists(index_path) or not os.path.exists(data_path):
            logger.warning(f"Index or data not found at {path}")
            return False
            
        self.index = faiss.read_index(index_path)
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.texts = data["texts"]
            self.metadata = data["metadata"]
            
        self._initialize_bm25()
        logger.info(f"Vector store loaded from {path}. Contains {self.index.ntotal} vectors.")
        return True
