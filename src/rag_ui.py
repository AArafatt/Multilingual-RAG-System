# rag_ui.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(page_title="RAG QA System", layout="centered")

st.title("📚 Multilingual RAG QA System")
st.write("Ask a question based on the uploaded textbook content.")

question = st.text_input("❓ Enter your question")
top_k = st.slider("🔍 Number of top documents to retrieve", min_value=1, max_value=5, value=3)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        payload = {"question": question, "top_k": top_k}
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()

            st.success("✅ Answer:")
            st.markdown(f"**{result['answer']}**")

            st.markdown("---")
            st.info("📄 Context used:")
            for i, (ctx, score) in enumerate(zip(result["context"], result["scores"]), start=1):
                st.markdown(f"**[{i}] Score: {score:.3f}**")
                st.write(ctx)
                st.markdown("---")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Failed to contact FastAPI: {e}")
