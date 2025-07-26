import os
import re
import time
import numpy as np
import pytesseract
import faiss
from pdf2image import convert_from_path
from openai import OpenAI
from dotenv import load_dotenv

# === Load API Key from .env ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_PATH = "HSC26-Bangla1st-Paper.pdf"

# === Explicit Poppler & Tesseract Paths ===
POPPLER_PATH = r"C:\poppler\Library\bin"
PYTESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_PATH

# === OCR Extraction ===
def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    text = ""
    for img in images:
        raw = pytesseract.image_to_string(img, lang='ben+eng', config='--psm 6')
        text += raw.replace('\x0c', '').strip() + "\n"
    return text

# === Chunking ===
def split_into_chunks(text, chunk_size=1000, chunk_overlap=500):
    sentences = re.split(r'(।|\?|!|\n)', text)
    merged = []
    i = 0
    while i < len(sentences) - 1:
        s = sentences[i].strip()
        if i + 1 < len(sentences):
            s += sentences[i + 1].strip()
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

# === Extract MCQ Answers ===
def extract_mcq_answers(text):
    inline_answers = re.findall(r'(\d+)[.|।] .*?[উত্তর|উ:]*[:：\-\s]+([কখগঘ])', text)
    table_answers = re.findall(r'(\d+)\s*\|\s*([কখগঘ])', text)
    all_answers = {int(num): opt for num, opt in inline_answers + table_answers}
    return all_answers

# === Embedding ===
def embed_chunks_openai(chunks, openai_api_key, model_name="text-embedding-3-small"):
    client = OpenAI(api_key=openai_api_key)
    embeddings = []
    for i, chunk in enumerate(chunks):
        try:
            res = client.embeddings.create(input=chunk, model=model_name)
            embeddings.append(res.data[0].embedding)
            time.sleep(0.1)
        except Exception as e:
            print(f"⚠️ Chunk {i} error: {e}")
            embeddings.append([0.0] * 1536)
    return embeddings, model_name

# === FAISS Index Builders ===
def build_faiss_index(embeddings, metric='l2'):
    dim = len(embeddings[0])
    data = np.array(embeddings).astype('float32')

    if metric == 'l2':
        index = faiss.IndexFlatL2(dim)
    elif metric == 'dot':
        index = faiss.IndexFlatIP(dim)
    elif metric == 'cosine':
        faiss.normalize_L2(data)
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError("Unknown metric.")

    index.add(data)
    return index

# === Querying ===
def get_embedding(query, api_key, model):
    client = OpenAI(api_key=api_key)
    res = client.embeddings.create(input=query, model=model)
    return np.array(res.data[0].embedding, dtype=np.float32)

def retrieve(query, index, chunks, api_key, model, k=5, metric='l2'):
    q = get_embedding(query, api_key, model)
    if metric == 'cosine':
        faiss.normalize_L2(q.reshape(1, -1))
    D, I = index.search(np.array([q]), k)
    return [chunks[i] for i in I[0]], D[0]

# === Answer Generator ===
def generate_answer(context_chunks, query, api_key):
    if not context_chunks:
        return "❌ প্রাসঙ্গিক কোনো তথ্য পাওয়া যায়নি।"

    context = "\n".join(context_chunks)
    client = OpenAI(api_key=api_key)
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": '''You are a helpful assistant for answering Bangla and English textbook MCQs.
You will be given:
- A user-submitted multiple-choice question.
- A set of MCQs from the same page that includes:
  - Question texts
  - Four options: (ক), (খ), (গ), (ঘ) or (A), (B), (C), (D)
  - Answer keys, either inline ("উত্তর: গ") or tabular ("6 | গ").

Your job is to return the **correct answer text only**, no letters, no explanations.

Rules:
- Use only the provided context.
- Never guess or add commentary.
'''
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )
    return res.choices[0].message.content.strip()

# === Main Initialization ===
def initialize_rag(pdf_path, api_key):
    print("🔍 OCR extracting...")
    text = extract_text_from_pdf(pdf_path)
    print("✂️ Chunking...")
    chunks = split_into_chunks(text)
    print(f"✅ {len(chunks)} chunks.")
    print("🧠 Embedding...")
    embeddings, model = embed_chunks_openai(chunks, api_key)

    print("⚙️ Building indices...")
    idx_l2 = build_faiss_index(embeddings, 'l2')
    idx_dot = build_faiss_index(embeddings, 'dot')
    idx_cos = build_faiss_index(embeddings, 'cosine')
    print("✅ All indices ready.")

    mcq_answers = extract_mcq_answers(text)
    return chunks, model, {"L2": idx_l2, "Dot": idx_dot, "Cosine": idx_cos}, mcq_answers, text

# === Query Interface ===
def comparative_query_loop(chunks, indices, model, api_key):
    print("🤖 টাইপ করুন প্রশ্ন (type `exit` to quit)\n")
    while True:
        query = input("প্রশ্ন: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 বিদায়!")
            break

        for name, index in indices.items():
            print(f"\n📡 {name} Retrieval")
            results, scores = retrieve(query, index, chunks, api_key, model, k=5, metric=name.lower())
            print(f"Top Score: {scores[0]:.4f}")
            print("📚 Context Preview:")
            for i, r in enumerate(results[:2]):
                cleaned_text = r[:200].replace('\n', ' ')
                print(f"  {i+1}. {cleaned_text}")

            answer = generate_answer(results, query, api_key)
            print(f"✅ Answer from {name}:\n{answer}\n")
            print("-" * 40)

# === Entry Point ===
if __name__ == "__main__":
    chunks, emb_model, all_indices, mcq_answers, full_text = initialize_rag(PDF_PATH, OPENAI_API_KEY)
    comparative_query_loop(chunks, all_indices, emb_model, OPENAI_API_KEY)
