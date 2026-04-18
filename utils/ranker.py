from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import List, Dict

from utils.logger import get_logger

logger = get_logger(__name__)

class Ranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading Cross-Encoder model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def rank(self, query: str, search_results: List[Dict], top_k: int = 5) -> List[Dict]:
        """Re-ranks search results using a cross-encoder."""
        if not search_results:
            return []
            
        logger.info(f"Re-ranking {len(search_results)} results for query: '{query}'")
        
        texts = [res["text"] for res in search_results]
        features = self.tokenizer([query] * len(texts), texts, padding=True, truncation=True, return_tensors="pt")
        
        features = {k: v.to(self.device) for k, v in features.items()}
        
        with torch.no_grad():
            scores = self.model(**features).logits.squeeze(-1).cpu().numpy()
            
        # Add scores to results and sort
        for res, score in zip(search_results, scores):
            res["score"] = float(score)
            
        search_results.sort(key=lambda x: x["score"], reverse=True)
        return search_results[:top_k]
