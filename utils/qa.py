import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, List
from utils.logger import get_logger
import os

# Suppress HF warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logger = get_logger(__name__)

class QuestionAnswering:
    def __init__(self, model_name: str = "google/flan-t5-large"):
        logger.info(f"Loading High-Capacity Generative QA model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def _detect_complexity(self, query: str) -> bool:
        """Determines if a query requires multi-part reasoning or a summary."""
        high_complexity_keywords = ["summarize", "scenario", "goal", "purpose", "why", "how", "compare", "describe", "explain", "objective"]
        query_lower = query.lower()
        return any(kw in query_lower for kw in high_complexity_keywords) or len(query.split()) > 10

    def answer(self, question: str, contexts: List[Dict]) -> Dict:
        """Generates a high-fidelity answer using a simplified RAG prompt."""
        if not contexts:
            return {
                "answer": "I don't have enough specific information to answer this.",
                "confidence": 0.0,
                "sources": []
            }
            
        logger.info(f"Generating answer using {len(contexts)} contexts...")
        
        # Build context
        context_text = ""
        for i, c in enumerate(contexts):
            context_text += f"[Doc {i+1}]: {c['text']}\n\n"
        
        is_complex = self._detect_complexity(question)
        
        # Standard Clean RAG Prompt - Works best for Flan-T5
        prompt = f"""Context from Forest Conservation Act documents:
{context_text}

Task: Answer the question below using only the context above. If the context doesn't contain the answer, say 'Unrelated' or 'I don't know'.
{'Provide a detailed response with 1-2 paragraphs.' if is_complex else 'Provide a direct 1-2 sentence response.'}

Question: {question}
Answer:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400 if is_complex else 150,
                    min_new_tokens=10 if is_complex else 0, # Low min token to allow short answers
                    num_beams=4,
                    repetition_penalty=1.2,
                    length_penalty=1.4 if is_complex else 1.0,
                    early_stopping=True
                )
                
            answer_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Refusal detection
            refusal_indicators = ["unrelated", "don't know", "not in the context", "can't answer"]
            is_refusal = any(ind in answer_text.lower() for ind in refusal_indicators)
            confidence = 0.1 if is_refusal else 0.9
            
            return {
                "answer": answer_text,
                "confidence": confidence,
                "sources": [c['metadata'] for c in contexts[:3]]
            }
            
        except Exception as e:
            logger.error(f"Error during generative QA: {e}")
            return {
                "answer": "An error occurred during answer generation.",
                "confidence": 0.0,
                "sources": []
            }
