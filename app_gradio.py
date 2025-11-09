# app_gradio.py
"""
Gradio chat UI using:
 - Pinecone (index 'mini-jarvis-index')
 - Sentence-Transformers embeddings for retrieval
 - llama-cpp (GGUF) for local generation
Run: python app_gradio.py
"""

import os
import time
import traceback
from dotenv import load_dotenv

import gradio as gr
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from llama_cpp import Llama

# === Load environment variables ===
load_dotenv()

# === Config ===
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "mini-jarvis-index")
LLAMA_PATH = os.getenv("LLAMA_GGUF_PATH")

# Basic safety checks
if not PINECONE_KEY:
    raise SystemExit("❌ Missing: PINECONE_API_KEY in .env")
if not LLAMA_PATH or not os.path.isfile(LLAMA_PATH):
    raise SystemExit("❌ Missing or invalid LLAMA_GGUF_PATH in .env (path to .gguf model)")

# === Initialize Pinecone (new SDK syntax) ===
try:
    print("🔗 Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_KEY)

    # Check or create the index
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating index '{INDEX_NAME}' (dim=384, metric='cosine')...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # Embedding dimension for all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Give Pinecone a few seconds to finish creating
        time.sleep(3)

    index = pc.Index(INDEX_NAME)
    print(f"✅ Pinecone index '{INDEX_NAME}' is ready.")
except Exception as e:
    traceback.print_exc()
    raise SystemExit(f"❌ Failed to initialize Pinecone: {e}")

# === Initialize embedder ===
try:
    print("🔍 Loading SentenceTransformer...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedder loaded.")
except Exception as e:
    traceback.print_exc()
    raise SystemExit(f"❌ Failed to load SentenceTransformer: {e}")

# === Load LLaMA model ===
print("🦙 Loading LLaMA model... this may take a few minutes.")
try:
    llm = Llama(model_path=LLAMA_PATH, n_ctx=2048)
    print("✅ Model loaded successfully!")
except Exception as e:
    traceback.print_exc()
    raise SystemExit(f"❌ Failed to load LLaMA model: {e}")

# === Helper functions ===
def retrieve_top_k(question: str, k: int = 3):
    """Retrieve top-k most relevant text chunks from Pinecone."""
    if not question.strip():
        return []
    qvec = embedder.encode(question, convert_to_numpy=True).tolist()

    try:
        res = index.query(vector=qvec, top_k=k, include_metadata=True)
    except Exception as e:
        print("⚠️ Pinecone query error:", e)
        return []

    matches = []
    raw_matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
    for m in raw_matches:
        meta = m.get("metadata", {}) if isinstance(m, dict) else getattr(m, "metadata", {})
        text = meta.get("text", "(no text)")
        src = meta.get("source", "unknown")
        matches.append({"text": text, "source": src})
    return matches


def build_prompt(question: str, chunks):
    """Builds context + question prompt for the LLM."""
    context = "\n\n".join(f"[{c['source']}]: {c['text']}" for c in chunks)
    return (
        "You are a helpful assistant. Use only the provided context to answer the question.\n"
        "If the answer is not in the context, say 'I don't know.'\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer concisely and cite sources."
    )


def generate_answer(prompt: str, max_tokens: int = 256, temp: float = 0.2):
    try:
        out = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temp,
            stop=["</s>", "USER:", "ASSISTANT:"],  # safe stops
        )
        return out["choices"][0]["text"].strip()
    except Exception as e:
        return f"LLAMA generation error: {e}"



def answer_question(question: str, top_k: int = 3, max_tokens: int = 256, temp: float = 0.2):
    """Full pipeline: retrieve, build prompt, and generate."""
    if not question.strip():
        return "Please enter a question.", ""

    chunks = retrieve_top_k(question, k=int(top_k))
    if not chunks:
        return "No relevant information found.", ""

    prompt = build_prompt(question, chunks)
    answer = generate_answer(prompt, max_tokens=int(max_tokens), temp=float(temp))
    sources = "\n".join(sorted({c["source"] for c in chunks}))
    return answer, sources


# === Gradio UI ===
def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## 🤖 Mini Jarvis — Local LLaMA + Pinecone Search")

        with gr.Row():
            q = gr.Textbox(label="Ask your question", placeholder="Type your question here...", lines=2)
        with gr.Row():
            top_k = gr.Slider(1, 10, value=3, step=1, label="Top-k results")
            max_toks = gr.Slider(64, 2048, value=256, step=64, label="Max tokens")
            temp = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")

        with gr.Row():
            ans = gr.Textbox(label="Answer", lines=8)
            src = gr.Textbox(label="Sources", lines=8)

        ask = gr.Button("Ask")

        q.submit(fn=answer_question, inputs=[q, top_k, max_toks, temp], outputs=[ans, src])
        ask.click(fn=answer_question, inputs=[q, top_k, max_toks, temp], outputs=[ans, src])

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)

