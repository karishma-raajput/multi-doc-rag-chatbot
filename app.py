"""
Multi-Document RAG Chatbot - Flask Backend
--------------------------------------------
Pipeline:
1. User uploads PDF/DOCX/TXT files
2. Text is extracted and split into chunks
3. Chunks are embedded using a local sentence-transformers model (free, no API cost)
4. Embeddings are stored in a FAISS index (free, local, in-memory vector store)
5. User asks a question -> question is embedded -> top-k similar chunks retrieved from FAISS
6. Retrieved chunks + question are sent to an LLM (Groq free tier) to generate the final answer
"""

import os
from dotenv import load_dotenv
load_dotenv()

import faiss
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import docx
from groq import Groq

# Config

UPLOAD_FOLDER = "uploads"
STORAGE_FOLDER = "storage"
INDEX_PATH = os.path.join(STORAGE_FOLDER, "faiss.index")
CHUNKS_PATH = os.path.join(STORAGE_FOLDER, "chunks.pkl")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # free, local, runs on CPU
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 8

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "openai/gpt-oss-20b"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STORAGE_FOLDER, exist_ok=True)

# Load embedding model once at startup
print("Loading embedding model...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = embed_model.get_sentence_embedding_dimension()

if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
    print("Found existing knowledge base on disk — loading it...")
    faiss_index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunk_store = pickle.load(f)
    print(f"Loaded {len(chunk_store)} chunks from a previous session.")
else:
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    chunk_store = []   # list of {"text": ..., "source": ...}

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def save_knowledge_base():
    """Persist the FAISS index and chunk store to disk."""
    faiss.write_index(faiss_index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunk_store, f)


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple fixed-size character chunker with overlap (no external deps)."""
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # move forward, keeping some overlap

    return [c.strip() for c in chunks if c.strip()]


# Helpers
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(filepath, filename):
    ext = filename.rsplit(".", 1)[1].lower()
    text = ""

    if ext == "pdf":
        reader = PdfReader(filepath)
        for page in reader.pages:
            text += (page.extract_text() or "") + "\n"

    elif ext == "docx":
        doc = docx.Document(filepath)
        for para in doc.paragraphs:
            text += para.text + "\n"

    elif ext == "txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

    return text


def embed_texts(texts):
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype("float32")


def retrieve_context(query, k=TOP_K):
    if faiss_index.ntotal == 0:
        return []

    query_vec = embed_texts([query])
    distances, indices = faiss_index.search(query_vec, min(k, faiss_index.ntotal))

    results = []
    for idx in indices[0]:
        if idx != -1:
            results.append(chunk_store[idx])
    return results


def generate_answer(query, context_chunks):
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )

    prompt = f"""You are a helpful assistant answering questions based ONLY on the provided context.
If the answer isn't in the context, say you don't have enough information.

Context:
{context_text}

Question: {query}

Answer:"""

    if not groq_client:
        return (
            "[No GROQ_API_KEY set — showing raw retrieved context instead of a generated answer]\n\n"
            + context_text
        )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700,
    )
    return response.choices[0].message.content


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "files" not in request.files:
        return jsonify({"error": "No files provided"}), 400

    files = request.files.getlist("files")
    total_chunks = 0
    processed_files = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            text = extract_text(filepath, filename)
            if not text.strip():
                continue

            chunks = split_text(text)
            if not chunks:
                continue

            vectors = embed_texts(chunks)
            faiss_index.add(vectors)

            for chunk in chunks:
                chunk_store.append({"text": chunk, "source": filename})

            total_chunks += len(chunks)
            processed_files.append(filename)

    if processed_files:
        save_knowledge_base()

    return jsonify({
        "message": f"Processed {len(processed_files)} file(s), {total_chunks} chunks indexed.",
        "files": processed_files,
        "total_chunks_in_store": len(chunk_store),
    })


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "has_documents": len(chunk_store) > 0,
        "total_chunks": len(chunk_store),
        "sources": sorted(set(c["source"] for c in chunk_store)) if chunk_store else [],
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    if faiss_index.ntotal == 0:
        return jsonify({"answer": "Please upload at least one document first.", "sources": []})

    context_chunks = retrieve_context(query)
    answer = generate_answer(query, context_chunks)
    sources = sorted(set(c["source"] for c in context_chunks))

    return jsonify({"answer": answer, "sources": sources})


@app.route("/reset", methods=["POST"])
def reset():
    global faiss_index, chunk_store
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)
    chunk_store = []

    for path in (INDEX_PATH, CHUNKS_PATH):
        if os.path.exists(path):
            os.remove(path)

    return jsonify({"message": "Knowledge base cleared."})


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)