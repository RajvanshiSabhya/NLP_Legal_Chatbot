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
        """Generates a high-fidelity answer using context-aware legal reasoning."""
        if not contexts:
            return {
                "answer": "I don't have enough specific legal data to reliably answer this.",
                "confidence": 0.0,
                "sources": []
            }
            
        logger.info(f"Generating answer using {len(contexts)} contexts...")
        
        # Build structured context with document type hints
        context_text = ""
        doc_types = set()
        for i, c in enumerate(contexts):
            dtype = c['metadata'].get('doc_type', 'LATENT')
            doc_types.add(dtype)
            source = c['metadata'].get('source', 'Unknown')
            context_text += f"[{dtype} Source {i+1} - {source}]: {c['text']}\n\n"
        
        is_complex = self._detect_complexity(question)
        is_judgment_heavy = "JUDGMENT" in doc_types
        
        # SOPHISTICATED LEGAL PROMPT
        prompt = f"""You are a specialized Legal AI assistant for Indian Forestry Law.
        
CONTEXT DOCUMENTS:
{context_text}

TASK: Provide a legally accurate answer to the question using ONLY the context provided above.
REASONING STEPS:
1. Identify if the answer is found in a STATUTE (Act) or a JUDGMENT (Case Law).
2. If it is a STATUTE, cite the specific Section number.
3. If it is a JUDGMENT, mention the Case Name or the outcome of the specific dispute.
4. If the question is unrelated to the context (e.g., general knowledge), say 'I don't know based on the provided documents as this is unrelated to forest law.'

{'Provide a detailed response with 1-2 paragraphs citing relevant sections or case outcomes.' if is_complex or is_judgment_heavy else 'Provide a direct 1-2 sentence legal response.'}

Question: {question}
Legal Answer:"""

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500 if is_complex else 200,
                    min_new_tokens=20 if is_complex else 5,
                    num_beams=4,
                    repetition_penalty=1.2,
                    length_penalty=1.4 if is_complex or is_judgment_heavy else 1.0,
                    early_stopping=True
                )
                
            answer_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Remove redundant "Legal Answer:" prefix if model generated it
            answer_text = re.sub(r'^Legal Answer:\s*', '', answer_text)
            
            # Refusal detection
            refusal_indicators = ["unrelated", "don't know", "not in the context", "can't answer", "not mentioned"]
            is_refusal = any(ind in answer_text.lower() for ind in refusal_indicators)
            confidence = 0.15 if is_refusal else 0.9
            
            return {
                "answer": answer_text,
                "confidence": confidence,
                "sources": [c['metadata'] for c in contexts[:3]]
            }
            
        except Exception as e:
            logger.error(f"Error during generative QA: {e}")
            return {
                "answer": f"An error occurred during legal reasoning: {str(e)}",
                "confidence": 0.0,
                "sources": []
            }
