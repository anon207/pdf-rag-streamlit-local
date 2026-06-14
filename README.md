# PDF Q&A (Local RAG, Context-Only)

Ask questions about a PDF **using only the PDF’s content** (no web search, no paid APIs). This app runs fully locally with:
- **Streamlit** (UI)
- **PyPDF** (PDF text extraction)
- **Sentence-Transformers** (embeddings)
- **FAISS** (vector search / retrieval)
- **Ollama** (local LLM generation)

If the PDF doesn’t contain the answer, the model will respond:
> **"I can’t find that in the PDF."**

---

## Demo

1) Upload a PDF  
2) Click **Process PDF** (chunks + embeddings + FAISS index)  
3) Ask a question  
4) Get an answer **plus citations** (file + page + retrieved passages)

---

## Why I built this

This project demonstrates an end-to-end **RAG (Retrieval-Augmented Generation)** pipeline:
- Document ingestion → chunking
- Embedding + vector indexing
- Top‑k retrieval with citations
- Context-only prompt construction
- Local LLM inference via Ollama

---

## Requirements

- **Python 3.10+** (recommended: 3.11)
- **Ollama** installed: https://ollama.com
- A pulled model (default):
  ```bash
  ollama pull llama3:8b


Setup (Windows)
Create and activate a virtual environment:
python -m venv .venv
.\.venv\Scripts\activate

Install dependencies:
pip install streamlit pypdf sentence-transformers faiss-cpu requests

Start the Ollama server (if it isn’t already running):
ollama serve

Run the app:
streamlit run app.py

Open the local URL Streamlit prints (usually http://localhost:8501).

Project structure

app.py - Streamlit UI (upload, process, ask, view citations)
pdf_utils.py - PDF extraction (per-page text + page numbers)
rag.py - chunking, embeddings, FAISS indexing, retrieval, Ollama prompting


How it works (high level)

Extract text per page from the PDF
Chunk the text with overlap (improves recall)
Convert chunks into embeddings (vectors)
Store vectors in a FAISS index for similarity search
For each question:
embed the question
retrieve the top‑k most relevant chunks
build a context-only prompt with citations
generate an answer using Ollama (local LLM)


Notes / limitations

This is context-only to reduce hallucinations. If the PDF doesn’t contain the info, the app refuses to answer.
PDF extraction quality depends on the PDF (scanned/image PDFs may extract poorly).
Very large PDFs may take time to embed and index (chunking parameters affect speed).


Roadmap (possible next steps)

Support image/scanned PDFs via OCR (Tesseract)
Add lightweight evaluation (saved Q/A set + retrieval checks)
Add reranking for better retrieval precision
Support multiple PDFs + persistent indexes


Screenshots

Pictures/upload.png
Pictures/first_example.png
Pictures/Second_example.png


License
MIT
```