import streamlit as st
from rag_engine import CorrectiveRAG

st.set_page_config(page_title="Corrective RAG PDF Lab", page_icon="📚", layout="wide")
st.title("📚 Corrective RAG (CRAG) — PDF Question Answering Lab")
st.write(
    "Upload multiple PDFs, index their text, and ask questions. "
    "This student demo uses TF-IDF retrieval and a simple corrective retrieval step."
)

if "rag" not in st.session_state:
    st.session_state.rag = CorrectiveRAG()
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

with st.sidebar:
    st.header("Model settings")
    provider = st.radio("Choose model connection", ["Ollama", "Groq API"])
    if provider == "Ollama":
        model = st.text_input("Ollama model name", value="llama3.2:3b")
        st.caption("Requires Ollama running on this computer. Free local inference; speed depends on your hardware.")
        api_key = ""
    else:
        model = st.text_input("Groq model name", value="llama-3.3-70b-versatile")
        api_key = st.text_input("Groq API key", type="password", help="Use a key from Groq Console. Never commit it to GitHub.")
        st.caption("Free-tier availability and limits can change. This is not unlimited.")
    top_k = st.slider("Number of retrieved text chunks", min_value=2, max_value=8, value=4)
    st.divider()
    if st.button("Clear knowledge base"):
        st.session_state.rag = CorrectiveRAG()
        st.session_state.indexed_files = []
        st.success("Knowledge base cleared.")
        st.rerun()

st.subheader("1. Upload and index PDFs")
uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Build / update knowledge base", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF first.")
    else:
        try:
            with st.spinner("Reading PDFs and building the text index..."):
                added = st.session_state.rag.add_pdfs(uploaded_files)
                st.session_state.indexed_files = sorted(set(
                    st.session_state.indexed_files + [f.name for f in uploaded_files]
                ))
            st.success(f"Finished. Added {added} text chunks from the uploaded PDFs.")
        except Exception as exc:
            st.error(str(exc))

if st.session_state.rag.chunks:
    st.info(
        f"Knowledge base contains {len(st.session_state.rag.chunks)} text chunks. "
        f"Files indexed in this session: {', '.join(st.session_state.indexed_files)}"
    )
else:
    st.caption("No PDFs indexed yet. Upload files and click Build / update knowledge base.")

st.subheader("2. Ask a question")
question = st.text_input("Example: What are the main points discussed in these documents?")
if st.button("Get answer"):
    if not question.strip():
        st.warning("Type a question first.")
    elif not st.session_state.rag.chunks:
        st.warning("Upload and index PDFs before asking questions.")
    else:
        try:
            with st.spinner("Retrieving evidence and generating an answer..."):
                result = st.session_state.rag.answer(
                    question=question.strip(),
                    provider=provider,
                    model=model.strip(),
                    api_key=api_key,
                    top_k=top_k
                )
            st.markdown("### Answer")
            st.write(result["answer"])
            retrieval = result["retrieval"]
            st.markdown("### Retrieval check")
            st.write(f"Best TF-IDF match score: **{retrieval['score']:.3f}**")
            st.write(f"Corrective step: **{'Used' if retrieval['corrected'] else 'Not needed'}**")
            st.caption(retrieval["message"])
            with st.expander("Show retrieved evidence"):
                for idx, chunk in enumerate(retrieval["chunks"], start=1):
                    st.markdown(f"**Result {idx}: {chunk['source']} — page {chunk['page']} (score {chunk['score']:.3f})**")
                    st.write(chunk["text"])
                    st.divider()
        except Exception as exc:
            st.error(str(exc))

st.divider()
st.caption(
    "Learning note: this is an educational CRAG-style implementation, not a full research-grade CRAG reproduction. "
    "TF-IDF scores are heuristic relevance signals, not calibrated probabilities. PDFs are processed in memory for this session."
)
