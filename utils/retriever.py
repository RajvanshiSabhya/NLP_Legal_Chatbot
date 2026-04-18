from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

class Retriever:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Returns embeddings for a list of texts."""
        logger.info(f"Encoding {len(texts)} texts...")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return embeddings

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
