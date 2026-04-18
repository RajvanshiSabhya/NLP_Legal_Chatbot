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

def process_pdfs(path: str) -> List[Dict]:
    """Loads PDFs using structure-aware block extraction."""
    logger.info(f"Processing path: {path}")
    processed_data = []
    
    if not os.path.exists(path):
        logger.warning(f"Path {path} does not exist.")
        return processed_data
        
    filepaths = []
    if os.path.isdir(path):
        for filename in os.listdir(path):
            if filename.lower().endswith(".pdf"):
                filepaths.append(os.path.join(path, filename))
    elif os.path.isfile(path) and path.lower().endswith(".pdf"):
        filepaths.append(path)
    else:
        logger.warning(f"Path {path} is neither a directory nor a PDF file.")
        return processed_data
        
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        logger.info(f"Reading {filepath} with structure-aware extraction")
        try:
            doc = fitz.open(filepath)
            all_blocks = []
            for page in doc:
                # get_text("blocks") returns a list of (x0, y0, x1, y1, "text", block_no, block_type)
                blocks = page.get_text("blocks")
                all_blocks.extend(blocks)
            doc.close()
            
            # Use logical blocks to create chunks
            chunks = chunk_blocks(all_blocks, max_chunk_size=400)
            metadata = extract_metadata_from_filename(filename)
            
            for i, chunk in enumerate(chunks):
                meta = metadata.copy()
                meta["chunk_id"] = i
                meta["section_hint"] = extract_section_hint(chunk)
                processed_data.append({
                    "text": clean_text(chunk),
                    "metadata": meta
                })
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
                
    logger.info(f"Total structured chunks created: {len(processed_data)}")
    return processed_data

if __name__ == "__main__":
    # Test locally
    data = process_pdfs("data/raw")
    if data:
        print(f"Sample Chunk: {data[0]['text'][:200]}...")
        print(f"Metadata: {data[0]['metadata']}")
