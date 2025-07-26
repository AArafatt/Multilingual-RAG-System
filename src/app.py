# app.py
import os
import re
import time
import numpy as np
import pytesseract
import faiss
from pdf2image import convert_from_path
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from pathlib import Path

# ───── 1) Explicitly locate & load .env ─────
BASE_DIR   = Path(__file__).resolve().parent
dotenv_path = BASE_DIR / ".env"
if not dotenv_path.exists():
    raise RuntimeError(f"⚠️ .env file not found at {dotenv_path}")
load_dotenv(dotenv_path)

# ───── 2) Read and validate API key ─────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("⚠️ OPENAI_API_KEY is not set in .env")

# ───── 3) Resolve your PDF path ─────
PDF_FILENAME = "HSC26-Bangla1st-Paper.pdf"
PDF_PATH     = BASE_DIR / PDF_FILENAME
if not PDF_PATH.exists():
    raise RuntimeError(f"⚠️ PDF not found at {PDF_PATH}")

# ───── 4) Local binaries ─────
POPPLER_PATH       = r"C:\poppler\Library\bin"
TESSERACT_CMD_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH

# ───── 5) Utility functions ─────
def extract_text_from_pdf(pdf_path: Path) -> str:
    images = convert_from_path(str(pdf_path), poppler_path=POPPLER_PATH)
    text = ""
    for img in images:
        raw = pytesseract.image_to_string(img, lang='ben', config='--psm 6')
        text += raw.replace('\x0c', '').strip() + "\n"
    return text

def split_into_chunks(text: str, chunk_size=1000, chunk_overlap=500) -> List[str]:
    sentences, merged, i = re.split(r'(।|\?|!|\n)', text), [], 0
    while i < len(sentences) - 1:
        s = sentences[i].strip() + sentences[i+1].strip()
        if s:
            merged.append(s)
        i += 2

    chunks, curr = [], ""
    for s in merged:
        if len(curr) + len(s) <= chunk_size:
            curr += s + " "
        else:
            chunks.append(curr.strip())
            curr = curr[-chunk_overlap:] + s + " "
    if curr:
        chunks.append(curr.strip())
    return chunks

def embed_chunks_openai(chunks: List[str], api_key: str, model_name="text-embedding-3-small"):
    client, embs = OpenAI(api_key=api_key), []
    for chunk in chunks:
        resp = client.embeddings.create(input=chunk, model=model_name)
        embs.append(resp.data[0].embedding)
        time.sleep(0.1)
    return np.array(embs, dtype='float32'), model_name

def build_faiss_index(emb_array: np.ndarray, metric='cosine'):
    dim = emb_array.shape[1]
    if metric == 'l2':
        idx = faiss.IndexFlatL2(dim)
    else:
        faiss.normalize_L2(emb_array)
        idx = faiss.IndexFlatIP(dim)
    idx.add(emb_array)
    return idx

def get_embedding(text: str, api_key: str, model_name: str, metric='cosine'):
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(input=text, model=model_name)
    vec = np.array(resp.data[0].embedding, dtype='float32').reshape(1, -1)
    if metric == 'cosine':
        faiss.normalize_L2(vec)
    return vec

def retrieve(question: str, idx, chunks: List[str], api_key: str, model_name: str, top_k=3):
    qv = get_embedding(question, api_key, model_name, metric='cosine')
    D, I = idx.search(qv, top_k)
    return [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]

def generate_answer(contexts: List[str], question: str, api_key: str) -> str:
    prompt = "\n\n".join(contexts)
    client = OpenAI(api_key=api_key)
    chat = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system", "content":"You are a helpful assistant for Bangla MCQs."},
            {"role":"user",   "content":f"Context:\n{prompt}\n\nQuestion: {question}"}
        ]
    )
    return chat.choices[0].message.content.strip()

# ───── 6) Build RAG index at startup ─────
print(f"Loading and OCR’ing PDF from: {PDF_PATH}")
_raw_text = extract_text_from_pdf(PDF_PATH)
_chunks    = split_into_chunks(_raw_text)
_emb_array, _emb_model = embed_chunks_openai(_chunks, OPENAI_API_KEY)
_index = build_faiss_index(_emb_array, metric='cosine')
print(f"Built FAISS index with {_emb_array.shape[0]} chunks")

# ───── 7) FastAPI app ─────
app = FastAPI(title="Bangla RAG API")

class Query(BaseModel):
    question: str
    top_k: int = 3

class Response(BaseModel):
    answer: str
    context: List[str]
    scores: List[float]

@app.get("/")
def read_root():
    return {
        "message": "API running. POST to /query",
        "example": {"question": "বিয়ের সময় কল্যাণীর প্রকত বয়স কত িছল?", "top_k":3}
    }

@app.post("/query", response_model=Response)
def query_rag(q: Query):
    hits = retrieve(q.question, _index, _chunks, OPENAI_API_KEY, _emb_model, top_k=q.top_k)
    contexts, scores = zip(*hits)
    ans = generate_answer(list(contexts), q.question, OPENAI_API_KEY)
    return Response(answer=ans, context=list(contexts), scores=list(scores))
