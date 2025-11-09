# pinecone_setup.py
"""
Final working version: creates a Pinecone index (serverless) and inserts a test embedding.
"""

import os
import sys
import traceback
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
if not API_KEY:
    print("❌ Missing PINECONE_API_KEY in .env")
    sys.exit(1)

# Initialize Pinecone
try:
    pc = Pinecone(api_key=API_KEY)
    print("✅ Pinecone client initialized.")
except Exception:
    print("❌ Failed to initialize Pinecone.")
    traceback.print_exc()
    sys.exit(1)

INDEX_NAME = "mini-jarvis-index"
DIM = 384  # matches the all-MiniLM-L6-v2 model

try:
    existing = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"🆕 Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # << required spec
        )
        print(f"✅ Index '{INDEX_NAME}' created.")
    else:
        print(f"ℹ️ Index '{INDEX_NAME}' already exists.")
except Exception:
    print("❌ Error while creating index.")
    traceback.print_exc()
    sys.exit(1)

# Connect to index
index = pc.Index(INDEX_NAME)
print("✅ Connected to index.")

# Generate and upload a real embedding
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    text = "This is Jarvis' first memory vector."
    vec = embedder.encode([text], show_progress_bar=False, convert_to_numpy=True)[0].tolist()

    index.upsert(vectors=[("test-id-1", vec, {"text": text})])
    print("✅ Test vector inserted successfully.")
except Exception:
    print("❌ Failed to create/upsert embedding.")
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 All done! Pinecone index is ready and a test vector has been inserted.")
