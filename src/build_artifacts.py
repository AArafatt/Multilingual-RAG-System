# build_artifacts.py
import json
import numpy as np
import faiss
from main import (
    extract_text_from_pdf,
    split_into_chunks,
    embed_chunks_openai,
    build_faiss_index,
    OPENAI_API_KEY,
    PDF_PATH
)

# 1) Extract & chunk
text   = extract_text_from_pdf(PDF_PATH)
chunks = split_into_chunks(text)

# 2) Embed & index
emb_array, model = embed_chunks_openai(chunks, OPENAI_API_KEY)
index = build_faiss_index(emb_array, metric='cosine')

# 3) Persist
json.dump(chunks, open("chunks.json","w",encoding="utf-8"), ensure_ascii=False)
np.save("embeddings.npy", emb_array)
faiss.write_index(index, "faiss.index")
