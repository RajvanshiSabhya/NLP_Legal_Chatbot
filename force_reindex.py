import os
import shutil
import requests

# Configuration
BASE_URL = "http://localhost:7860"
INDEX_PATH = "embeddings/"
UPLOAD_DIR = "data/raw/"

def force_reindex():
    print("--- Starting Force Re-Indexing ---")
    
    # 1. Check if the server is running
    try:
        requests.get(f"{BASE_URL}/health")
    except:
        print(f"Error: The server is not running on {BASE_URL}. Please start 'main.py' first.")
        return

    # 2. Delete the old embeddings folder locally
    if os.path.exists(INDEX_PATH):
        print(f"Deleting old index at {INDEX_PATH}...")
        shutil.rmtree(INDEX_PATH)
        print("Old index deleted.")
    
    # 3. List all PDFs in the raw folder
    pdfs = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    
    if not pdfs:
        print(f"No PDFs found in {UPLOAD_DIR}. Please add some PDFs first.")
        return

    print(f"Found {len(pdfs)} PDFs. Re-ingesting now...")

    # 4. Re-ingest each PDF via the API
    for pdf in pdfs:
        print(f"Ingesting {pdf}...")
        pdf_path = os.path.join(UPLOAD_DIR, pdf)
        with open(pdf_path, "rb") as f:
            files = {"file": (pdf, f, "application/pdf")}
            response = requests.post(f"{BASE_URL}/ingest", files=files)
            
        if response.status_code == 200:
            print(f"Successfully re-ingested {pdf}")
        else:
            print(f"Failed to ingest {pdf}: {response.text}")

    print("\n--- Re-Indexing Complete ---")
    print("Please restart your main.py server now to ensure all models reload the fresh index.")

if __name__ == "__main__":
    force_reindex()
