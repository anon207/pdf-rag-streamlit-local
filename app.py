import streamlit as st 
from sentence_transformers import SentenceTransformer
from pdf_utils import extract_pages 
from rag import build_chunks, embed_texts, make_faiss_index, retrieve, answer_question_context_only

filename = None

st.set_page_config(page_title="PDF RAG (Local)", layout="wide") 
st.title("PDF Q&A (Local RAG, context-only)")
@st.cache_resource 

def load_embed_model(): 
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

if "ready" not in st.session_state: 
    st.session_state.ready = False

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

col1, col2 = st.columns(2)

with col1: 
    chunk_size = st.slider("Chunk size (chars)", 600, 1600, 1000, 100) 
with col2: 
    overlap = st.slider("Overlap (chars)", 0, 400, 200, 50)

if uploaded is not None: 
    if st.button("Process PDF"): 
        filename = uploaded.name

    with st.spinner("Extracting text..."):
        pages = extract_pages(uploaded)

    with st.spinner("Chunking..."):
        chunks = build_chunks(pages, filename=filename, chunk_size=chunk_size, overlap=overlap)

    with st.spinner("Embedding + indexing..."):
        embs = embed_texts(embed_model, [c["text"] for c in chunks])
        index = make_faiss_index(embs)

    st.session_state.filename = filename
    st.session_state.pages = pages
    st.session_state.chunks = chunks
    st.session_state.index = index
    st.session_state.ready = True

    st.success(f"Ready: indexed {len(chunks)} chunks from {filename}")

    st.divider()

    question = st.text_input("Ask a question about the PDF")

    k = st.slider("Top-k passages", 2, 10, 5, 1)

    if st.button("Answer", disabled=not st.session_state.ready or not question.strip()): 
        chunks = st.session_state.chunks 
        index = st.session_state.index

    with st.spinner("Retrieving..."):
        hits = retrieve(question, embed_model, index, chunks, k=k)

    with st.spinner("Generating answer (local LLM)..."):
        answer = answer_question_context_only(question, hits)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Citations (retrieved context)")
    for h in hits:
        with st.expander(f"{h['source']} — page {h['page']} (score {h['score']:.3f})"):
            st.write(h["text"])

