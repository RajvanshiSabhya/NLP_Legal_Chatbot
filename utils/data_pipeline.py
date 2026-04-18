import os
import re
import fitz  # PyMuPDF
from typing import List, Dict

from utils.logger import get_logger

logger = get_logger(__name__)

def clean_text(text: str) -> str:
    """Removes extra whitespaces, normalize newlines, and noise from text."""
    # Remove surrogate characters (common in bad PDF extractions)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Collapse multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove common PDF artifacts like "Page X of Y" if they appear alone
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    return text.strip()

def chunk_blocks(blocks: List[tuple], max_chunk_size: int = 400) -> List[str]:
    """Combines logical blocks into chunks while respecting block boundaries."""
    chunks = []
    current_chunk = []
    current_length = 0
    
    for b in blocks:
        text = b[4].strip()
        if not text:
            continue
            
        block_length = len(text.split())
        
        # If a single block is too huge (rare in this doc), split it
        if block_length > max_chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split huge block into smaller sub-chunks
            words = text.split()
            for i in range(0, len(words), max_chunk_size):
                chunks.append(" ".join(words[i:i + max_chunk_size]))
            continue

        if current_length + block_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [text]
            current_length = block_length
        else:
            current_chunk.append(text)
            current_length += block_length
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

def extract_section_hint(text: str) -> str:
    """Heuristic to identify the primary Section number from a chunk."""
    match = re.search(r'Section\s*(\d+[A-Z]?)', text, re.IGNORECASE)
    if match:
        return match.group(1)
    return "Unknown"

def extract_metadata_from_filename(filename: str) -> Dict:
    """Heuristic to extract state or info from filename."""
    states = ['Delhi', 'Haryana', 'Punjab', 'Uttar Pradesh']
    found_state = 'Unknown'
    
    for state in states:
        if state.lower() in filename.lower():
            found_state = state
            break
            
    return {
        "source": filename,
        "state": found_state,
        "year": "Unknown",  # Could be parsed via regex if needed
        "document_type": "PDF"
    }

def detect_document_type(text: str) -> str:
    """Detects if the document is an 'Act/Statute' or a 'Court Judgment'."""
    judgment_keywords = ["judgment", "petitioner", "respondent", "versus", "appeal", "high court", "supreme court"]
    statute_keywords = ["act", "section", "chapter", "preamble", "gazette"]
    
    text_lower = text[:2000].lower()
    judgment_score = sum(1 for kw in judgment_keywords if kw in text_lower)
    statute_score = sum(1 for kw in statute_keywords if kw in text_lower)
    
    return "JUDGMENT" if judgment_score >= statute_score else "STATUTE"

def extract_legal_entities(text: str) -> Dict:
    """Extracts case names or section numbers as high-priority metadata."""
    case_match = re.search(r'(?:Judgment in|In the matter of)\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+vs\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
    section_match = re.search(r'Section\s*(\d+[A-Z]?)', text, re.IGNORECASE)
    
    return {
        "case_name": case_match.group(1) if case_match else "Unknown",
        "section_hint": section_match.group(1) if section_match else "Unknown"
    }

def process_pdfs(path: str) -> List[Dict]:
    """Loads PDFs with advanced legal structure awareness."""
    logger.info(f"Processing legal documents at: {path}")
    processed_data = []
    
    if not os.path.exists(path):
        return processed_data
        
    filepaths = []
    if os.path.isdir(path):
        filepaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(".pdf")]
    elif path.lower().endswith(".pdf"):
        filepaths = [path]
    
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        try:
            doc = fitz.open(filepath)
            full_text = ""
            all_blocks = []
            for page in doc:
                blocks = page.get_text("blocks")
                all_blocks.extend(blocks)
                full_text += page.get_text()
            doc.close()
            
            # 1. Detect Document Type
            doc_type = detect_document_type(full_text)
            logger.info(f"Detected {doc_type} for {filename}")
            
            # 2. Extract Metadata
            base_meta = extract_metadata_from_filename(filename)
            base_meta["doc_type"] = doc_type
            
            # 3. Create Semantic Chunks
            chunks = chunk_blocks(all_blocks, max_chunk_size=400)
            
            for i, chunk in enumerate(chunks):
                clean_chunk = clean_text(chunk)
                entities = extract_legal_entities(clean_chunk)
                
                meta = base_meta.copy()
                meta.update(entities)
                meta["chunk_id"] = i
                
                processed_data.append({
                    "text": clean_chunk,
                    "metadata": meta
                })
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
                
    return processed_data

if __name__ == "__main__":
    # Test locally
    data = process_pdfs("data/raw")
    if data:
        print(f"Sample Chunk: {data[0]['text'][:200]}...")
        print(f"Metadata: {data[0]['metadata']}")
