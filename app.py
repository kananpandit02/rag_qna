# app.py
import os
import re
import torch
import faiss
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_documents(folder_path):
    all_texts = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    all_texts.append(content)
        elif file.endswith(".pdf"):
            reader = PdfReader(path)
            text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
            if text.strip():
                all_texts.append(text)
    return all_texts

def fallback_sent_tokenize(text):
    return re.split(r'(?<=[.?!])\s+', text.strip())

def chunk_text(text, max_words=100):
    sentences = fallback_sent_tokenize(text)
    chunks, current, count = [], [], 0
    for sent in sentences:
        words = sent.split()
        if count + len(words) > max_words:
            if current:
                chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks

def embed_chunks(chunks, model):
    return model.encode(chunks, show_progress_bar=False)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def generate_answer(query, retriever, passages, embeddings, index, gen_model, tokenizer, top_k=3):
    q_embed = retriever.encode([query])
    scores, indices = index.search(q_embed, top_k)
    top_chunks = sorted(set([i for i in indices[0] if i < len(passages)]))
    context = "\n".join([f"[Chunk {i}]: {passages[i]}" for i in top_chunks])
    prompt = f"question: {query} context: {context}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = gen_model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True), top_chunks

def main():
    st.set_page_config(page_title="🧠 RAG Q&A App", layout="wide")
    st.title("📚 Ask Questions from PDF/TXT using RAG")

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    uploaded_files = st.file_uploader("📤 Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    question = st.text_input("❓ Enter your question:")

    if st.button("🧠 Get Answer") and uploaded_files and question:
        for file in uploaded_files:
            with open(os.path.join(upload_folder, file.name), "wb") as f:
                f.write(file.read())

        st.info("⏳ Loading models...")
        retriever = SentenceTransformer("all-MiniLM-L6-v2")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        docs = load_documents(upload_folder)
        chunks = [chunk for doc in docs for chunk in chunk_text(doc)]

        st.info("🔍 Embedding & indexing...")
        chunk_embeddings = embed_chunks(chunks, retriever)
        faiss_index = build_faiss_index(torch.tensor(chunk_embeddings).numpy())

        st.success("🧠 Generating Answer...")
        answer, evidence_ids = generate_answer(
            question, retriever, chunks, torch.tensor(chunk_embeddings).numpy(),
            faiss_index, gen_model, tokenizer)

        st.subheader("✅ Answer:")
        st.success(answer)

        st.subheader("📄 Evidence:")
        for i in evidence_ids:
            st.markdown(f"**Chunk {i}:** {chunks[i]}")

if __name__ == "__main__":
    main()

import streamlit as st
from pypdf import PdfReader
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Load documents
def load_documents(folder_path):
    all_texts = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    all_texts.append(content)
        elif file.endswith(".pdf"):
            reader = PdfReader(path)
            text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
            if text.strip():
                all_texts.append(text)
    return all_texts

def fallback_sent_tokenize(text):
    return re.split(r'(?<=[.?!])\s+', text.strip())

def chunk_text(text, max_words=100):
    sentences = fallback_sent_tokenize(text)
    chunks, current, count = [], [], 0
    for sent in sentences:
        words = sent.split()
        if count + len(words) > max_words:
            if current:
                chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sent)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks

def embed_chunks(chunks, model):
    instruction = "Represent the passage for retrieval:"
    inputs = [[instruction, chunk] for chunk in chunks]
    return model.encode(inputs, show_progress_bar=False)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def generate_answer(query, retriever, passages, embeddings, index, gen_model, tokenizer, top_k=3):
    q_embed = retriever.encode([["Represent the question for retrieving supporting documents", query]])
    scores, indices = index.search(q_embed, top_k)
    top_chunks = sorted(set([i for i in indices[0] if i < len(passages)]))
    context = "\n".join([f"[Chunk {i}]: {passages[i]}" for i in top_chunks])
    prompt = f"question: {query} context: {context}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = gen_model.generate(**inputs, max_length=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True), top_chunks

def main():
    st.set_page_config(page_title="🧠 RAG Q&A App", layout="wide")
    st.title("📚 Ask Questions from PDF/TXT using RAG")

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    uploaded_files = st.file_uploader("📤 Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    question = st.text_input("❓ Enter your question:")

    if st.button("🧠 Get Answer") and uploaded_files and question:
        for file in uploaded_files:
            with open(os.path.join(upload_folder, file.name), "wb") as f:
                f.write(file.read())

        st.info("⏳ Loading models...")
        retriever = INSTRUCTOR("hkunlp/instructor-base")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        docs = load_documents(upload_folder)
        chunks = [chunk for doc in docs for chunk in chunk_text(doc)]

        st.info("🔍 Embedding & indexing...")
        chunk_embeddings = embed_chunks(chunks, retriever)
        faiss_index = build_faiss_index(torch.tensor(chunk_embeddings).numpy())

        st.success("🧠 Generating Answer...")
        answer, evidence_ids = generate_answer(
            question, retriever, chunks, torch.tensor(chunk_embeddings).numpy(),
            faiss_index, gen_model, tokenizer)

        st.subheader("✅ Answer:")
        st.success(answer)

        st.subheader("📄 Evidence:")
        for i in evidence_ids:
            st.markdown(f"**Chunk {i}:** {chunks[i]}")

if __name__ == "__main__":
    main()
