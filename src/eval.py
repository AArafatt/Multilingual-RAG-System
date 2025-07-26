# eval.py
import os
import json
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# ───── Load env & artifacts ─────
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# These artifacts you generate once with:
#   build_artifacts.py → chunks.json, embeddings.npy, faiss.index
chunks: List[str]       = json.load(open("chunks.json", "r", encoding="utf-8"))
emb_array: np.ndarray   = np.load("embeddings.npy")
index: faiss.Index      = faiss.read_index("faiss.index")

# ───── Define your test set ─────
TEST_QUERIES: List[Dict[str,str]] = [
    {
      "question": "বাংলা ভাষার জাতীয় কবি কে?",
      "ground_truth": "কাজী নজরুল ইসলাম"
    },
    # add more cases...
]

# ───── Helpers ─────
def get_embedding(text: str, api_key: str,
                  model="text-embedding-3-small") -> np.ndarray:
    client = OpenAI(api_key=api_key)
    r = client.embeddings.create(input=text, model=model)
    vec = np.array(r.data[0].embedding, dtype='float32').reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec

# ───── Evaluation ─────
def evaluate(k: int = 5):
    total_sim, grounded_hits = 0.0, 0
    N = len(TEST_QUERIES)

    for case in TEST_QUERIES:
        qvec = get_embedding(case["question"], API_KEY)
        D, I = index.search(qvec, k)
        total_sim += D[0][0]
        retrieved = [chunks[i] for i in I[0]]
        if any(case["ground_truth"] in txt for txt in retrieved):
            grounded_hits += 1

    return {
        "avg_top1_sim": total_sim / N,
        f"grounded_recall@{k}": grounded_hits / N
    }

# ───── Main ─────
if __name__ == "__main__":
    metrics = evaluate(k=5)
    print("=== RAG Retrieval Evaluation ===")
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")
