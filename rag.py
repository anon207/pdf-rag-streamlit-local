import requests 
import numpy as np 
import faiss

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]: 
    if chunk_size <= overlap: raise ValueError("chunk_size must be > overlap") 
    chunks = [] 
    start = 0 
    n = len(text) 
    while start < n: 
        end = min(start + chunk_size, n) 
        chunk = text[start:end].strip() 
        if chunk: chunks.append(chunk) 
        start += chunk_size - overlap 
    return chunks

def build_chunks(pages: list[dict], filename: str, chunk_size: int = 1000, overlap: int = 200) -> list[dict]: 
    out = [] 
    for p in pages: 
        for ch in chunk_text(p["text"], chunk_size=chunk_size, overlap=overlap): 
            out.append({ "text": ch, "source": filename, "page": p["page_num"], }) 
    return out

def embed_texts(model, texts: list[str]) -> np.ndarray: 
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False) 
    return np.asarray(vecs, dtype="float32")

def make_faiss_index(embeddings: np.ndarray): 
    dim = embeddings.shape[1] 
    index = faiss.IndexFlatIP(dim) # cosine sim because normalized 
    index.add(embeddings) 
    return index

def retrieve(query: str, model, index, chunks: list[dict], k: int = 5) -> list[dict]: 
    q = embed_texts(model, [query]) 
    scores, ids = index.search(q, k) 
    results = [] 
    for score, idx in zip(scores[0].tolist(), ids[0].tolist()): 
        if idx == -1: continue 
        item = dict(chunks[idx]) 
        item["score"] = float(score) 
        results.append(item) 
    return results

def call_ollama_generate(model: str, prompt: str) -> str: 
    r = requests.post( "http://localhost:11434/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=300 ) 
    r.raise_for_status() 
    return (r.json().get("response") or "").strip()

def answer_question_context_only(question: str, contexts: list[dict], ollama_model: str = "llama3:8b") -> str: 
    context_block = "\n\n".join( 
        [
        f"[Source: {c['source']} | Page: {c['page']}]\n{c['text']}" 
        for c in contexts
        ] 
    )

    prompt = f"""
    You are a careful assistant. 
    You MUST answer using only the provided context from the PDF. 
    If the context does not contain the answer, reply exactly: "I can’t find that in the PDF."
    Context: {context_block}
    Question: {question}
    Answer (use only the context, be concise):""" 

    return call_ollama_generate(ollama_model, prompt)