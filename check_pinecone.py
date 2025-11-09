# check_pinecone.py
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

INDEX_NAME = "mini-jarvis-index"

print("🔑 PINECONE_API_KEY set:", bool(os.getenv("PINECONE_API_KEY")))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

print("\n📊 Index stats:")
stats = index.describe_index_stats()
print(stats)

# Verify embedding dimension matches Pinecone index dimension
model = SentenceTransformer("all-MiniLM-L6-v2")
print("\n🧠 Embedding model:", model._model_card or "all-MiniLM-L6-v2")
print("🧭 Embedding dim:", model.get_sentence_embedding_dimension())

# Quick test query (won't fail if empty; just shows matches length)
qvec = model.encode("test question", convert_to_numpy=True).tolist()
res = index.query(vector=qvec, top_k=3, include_metadata=True)
matches = res.get("matches") if isinstance(res, dict) else getattr(res, "matches", [])
print("\n🔎 Query returned", len(matches), "matches")
if matches:
    for m in matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
        print("•", meta.get("source"), "| score:", getattr(m, "score", m.get("score")))
