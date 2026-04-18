import os
import torch
from typing import List, Dict, Optional
from utils.logger import get_logger
from utils.retriever import Retriever
from utils.vector_store import VectorStore
from utils.ranker import Ranker
from utils.qa import QuestionAnswering
from utils.data_pipeline import process_pdfs

logger = get_logger(__name__)

class LegalRAGEngine:
    def __init__(self, index_path: str = "embeddings/", upload_dir: str = "data/raw/"):
        self.index_path = index_path
        self.upload_dir = upload_dir
        
        # Models (lazy loaded)
        self.retriever = None
        self.vector_store = None
        self.ranker = None
        self.qa_model = None
        
    def _ensure_loaded(self):
        """Lazy loads all models if not already initialized."""
        if self.retriever is None:
            logger.info("Initializing LegalRAGEngine models...")
            self.retriever = Retriever()
            self.vector_store = VectorStore(embedding_dim=self.retriever.get_embedding_dimension())
            
            # Load index if exists
            if os.path.exists(os.path.join(self.index_path, "index.faiss")):
                self.vector_store.load(self.index_path)
            
            self.ranker = Ranker()
            self.qa_model = QuestionAnswering()
            logger.info("LegalRAGEngine successfully initialized.")

    def ingest_document(self, file_path: str) -> Dict:
        """Processes a new PDF and adds it to the persistent index."""
        self._ensure_loaded()
        
        processed_data = process_pdfs(file_path)
        if not processed_data:
            return {"status": "error", "message": "Failed to process document."}
            
        texts = [item["text"] for item in processed_data]
        metadata = [item["metadata"] for item in processed_data]
        
        # Generate embeddings and update index
        embeddings = self.retriever.encode(texts)
        self.vector_store.add(embeddings, texts, metadata)
        self.vector_store.save(self.index_path)
        
        return {
            "status": "success",
            "chunks": len(texts),
            "source": os.path.basename(file_path)
        }

    def ask(self, query: str, state_filter: Optional[str] = "All") -> Dict:
        """Executes the full RAG cycle: Search -> Gate -> Rank -> Answer."""
        self._ensure_loaded()
        
        if not self.vector_store.texts:
            return {
                "answer": "The system has no documents indexed. Please upload a PDF first.",
                "confidence": 0.0,
                "sources": []
            }

        # 1. Hybrid Search
        query_emb = self.retriever.encode([query])[0]
        search_result = self.vector_store.search(query_emb, query, k=40)
        retrieved_docs = search_result["results"]
        top_distance = search_result["top_distance"]
        
        # 2. Domain Gating (Hallucination Prevention)
        # 1.65 is the L2 distance threshold for out-of-scope detection
        if top_distance > 1.65:
            return {
                "answer": "This query is unrelated to the provided legal documents. I can only assist with matters concerning the Forest Conservation Act and related cases.",
                "confidence": 0.0,
                "sources": []
            }

        # 3. Search Priority (Keyword Boosting)
        critical_keywords = ["penalty", "punishment", "imprisonment", "fine", "contravenes", "goal", "objective", "purpose"]
        query_lower = query.lower()
        if any(kw in query_lower for kw in critical_keywords):
            retrieved_docs.sort(key=lambda x: any(kw in x["text"].lower() for kw in critical_keywords), reverse=True)

        # 4. Deep Re-ranking
        top_docs = self.ranker.rank(query, retrieved_docs, top_k=5)
        
        # 5. Generative QA
        qa_result = self.qa_model.answer(query, top_docs)
        
        return {
            "answer": qa_result["answer"],
            "confidence": qa_result["confidence"],
            "sources": qa_result["sources"]
        }
