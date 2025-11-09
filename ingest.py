# ingest.py
"""
Ingest files from ./data, chunk them, embed with sentence-transformers,
and upsert into Pinecone index 'mini-jarvis-index'.
Run: python ingest.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load env
load_dotenv()
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_KEY:
    print("Missing PINECONE_API_KEY in .env")
    sys.exit(1)

# Third-party imports (after env check)
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import tiktoken
from pdfminer.high_level import extract_text

# Config
DATA_DIR = Path("data")
INDEX_NAME = "mini-jarvis-index"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_TOKENS = 700
OVERLAP_TOKENS = 100

# Initialize clients
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(INDEX_NAME)
embedder = SentenceTransformer(EMBED_MODEL)
ENC = tiktoken.get_encoding("cl100k_base")  # token encoder for chunk sizing

# --- Helper functions (defined before use) ---
def text_from_file(path: Path) -> str:
    sfx = path.suffix.lower()
    if sfx == ".pdf":
        try:
            return extract_text(str(path))
        except Exception:
            return ""
    elif sfx in [".txt", ".md"]:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    else:
        return ""

def tokenize(text: str):
    return ENC.encode(text)

def detokenize(tokens):
    return ENC.decode(tokens)

def chunk_text(text: str, chunk_size: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS):
    """
    Return list of (start_token_index, end_token_index, chunk_text).
    Uses token-level chunking to keep approx token limits.
    """
    toks = tokenize(text)
    total = len(toks)
    if total == 0:
        return []
    chunks = []
    start = 0
    while start < total:
        end = min(start + chunk_size, total)
        chunk_toks = toks[start:end]
        chunk_str = detokenize(chunk_toks)
        chunks.append((start, end, chunk_str))
        if end == total:
            break
        start = end - overlap
    return chunks

# --- Ingest pipeline ---
def ingest():
    files = sorted([p for p in DATA_DIR.glob("*") if p.is_file()])
    if not files:
        print("No files found in ./data — put PDFs or .txt there and run again.")
        return

    batch = []
    id_counter = 0

    for f in files:
        text = text_from_file(f)
        if not text or len(text.strip()) == 0:
            print(f"Skipping (no text): {f.name}")
            continue

        chunks = chunk_text(text)
        print(f"File: {f.name} → {len(chunks)} chunks")

        for i, (s, e, chunk_str) in enumerate(chunks):
            meta = {
                "source": f.name,
                "chunk": i,
                "token_start": int(s),
                "token_end": int(e)
            }
            id_counter += 1
            vector_id = f"{f.stem}-{i}"

            # create embedding (single chunk at a time for memory-safety)
            emb = embedder.encode(chunk_str, show_progress_bar=False, convert_to_numpy=True).tolist()
            batch.append((vector_id, emb, meta))

            # upsert in small batches
            if len(batch) >= 50:
                index.upsert(vectors=batch)
                print(f"  Upserted {len(batch)} vectors (total so far: {id_counter})")
                batch = []

    # final upsert
    if batch:
        index.upsert(vectors=batch)
        print(f"  Upserted final {len(batch)} vectors (total: {id_counter})")

    print("Ingestion complete. Total vectors:", id_counter)

if __name__ == "__main__":
    ingest()
