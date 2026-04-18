import os
import shutil
import warnings
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Suppress warnings for clean production output
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from utils.logger import get_logger
from utils.engine import LegalRAGEngine

logger = get_logger(__name__)

# Core Application Setup
app = FastAPI(
    title="Legal-Aware Deforestation RAG Chatbot API",
    description="A production-grade RAG system for Indian Forest Law and Case Judgments.",
    version="2.0.0"
)

# Global Engine Instance
engine = LegalRAGEngine()

# CORS Configuration
raw_origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000,https://vanrakshakcm.vercel.app",
)
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https://.*\.hf\.space$|^https://.*\.vercel\.app$|^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    state: Optional[str] = "All"

@app.on_event("startup")
def startup():
    """Warms up the RAG engine on startup."""
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")
    # Lazy loading models early to reduce first-request latency
    engine._ensure_loaded()
    logger.info("API Startup complete.")

@app.get("/")
def root():
    return {
        "message": "Legal RAG API v2.0 Online",
        "documentation": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Uploads and processes a legal document (Act or Judgment)."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    upload_path = f"data/raw/{file.filename}"
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = engine.ingest_document(upload_path)
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
            
        return result
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(request: QueryRequest):
    """Orchestrates the full legal RAG cycle to provide accurate answers."""
    try:
        logger.info(f"Query received: {request.query}")
        result = engine.ask(request.query, state_filter=request.state)
        return {
            "question": request.query,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "sources": result["sources"]
        }
    except Exception as e:
        logger.error(f"QA Pipeline error: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred while processing your legal query.")

if __name__ == "__main__":
    import uvicorn
    # Respect platform ports (Railway/HuggingFace)
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
