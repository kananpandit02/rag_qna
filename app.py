#for multiple question and summary
# app.py
import os
import re
import torch
import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from InstructorEmbedding import INSTRUCTOR
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login

# ========== 0. Hugging Face Token Login ==========
if "HUGGINGFACE_HUB_TOKEN" in st.secrets:
    login(st.secrets["HUGGINGFACE_HUB_TOKEN"])
elif os.getenv("HUGGINGFACE_HUB_TOKEN"):
    login(os.getenv("HUGGINGFACE_HUB_TOKEN"])
else:
    st.error("❌ HuggingFace token not found. Set it in Streamlit secrets or environment.")
    st.stop()

# ========== 1. Load Documents ==========
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

# ========== 2. Sentence Tokenizer ==========
def fallback_sent_tokenize(text):
    return re.split(r'(?<=[.?!])\s+', text.strip())

# ========== 3. Chunk Text ==========
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

# ========== 4. Embed Chunks ==========
def embed_chunks(chunks, model):
    instruction = "Represent the passage for retrieval:"
    inputs = [[instruction, chunk] for chunk in chunks]
    embeddings = model.encode(inputs, show_progress_bar=False)
    return np.array(embeddings).astype("float32")

# ========== 5. Build FAISS Index ==========
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# ========== 6. Generate Answer or Summary ==========
def generate_answer(query, retriever, passages, embeddings, index, gen_model, tokenizer, top_k=3):
    is_summary = "summary" in query.lower() or "summarize" in query.lower()

    if is_summary:
        context = "\n".join(passages)[:1500]  # truncate for memory
        prompt = f"Summarize the following document:\n\n{context}"
    else:
        q_embed = retriever.encode([["Represent the question for retrieving supporting documents", query]])
        scores, indices = index.search(q_embed.astype("float32"), top_k)
        top_chunks = sorted(set([i for i in indices[0] if i < len(passages)]))
        context = "\n".join([f"[Chunk {i}]: {passages[i]}" for i in top_chunks])[:1500]
        prompt = f"question: {query} context: {context}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = gen_model.generate(**inputs, max_length=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, top_chunks if not is_summary else []

# ========== 7. Streamlit App ==========
def main():
    st.set_page_config(page_title="🧠 RAG Q&A App", layout="wide")
    st.title("📚 Ask Questions from PDF/TXT using RAG")

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    uploaded_files = st.file_uploader("📤 Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            file_path = os.path.join(upload_folder, file.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(file.read())
        st.success("✅ Files uploaded!")

    # Model loading once per session
    if "retriever" not in st.session_state:
        with st.spinner("⏳ Loading models..."):
            st.session_state.retriever = INSTRUCTOR("hkunlp/instructor-base")
            st.session_state.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            st.session_state.gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

    # Preprocess files and cache chunks
    if uploaded_files and "chunks" not in st.session_state:
        with st.spinner("📄 Processing documents..."):
            docs = load_documents(upload_folder)
            st.session_state.chunks = [chunk for doc in docs for chunk in chunk_text(doc)]

        with st.spinner("🔍 Embedding & indexing..."):
            st.session_state.embeddings = embed_chunks(st.session_state.chunks, st.session_state.retriever)
            st.session_state.index = build_faiss_index(st.session_state.embeddings)

    question = st.text_input("❓ Enter your question or ask for a summary:")

    if st.button("🧠 Get Answer") and question:
        with st.spinner("🧠 Generating Answer..."):
            answer, evidence_ids = generate_answer(
                question,
                st.session_state.retriever,
                st.session_state.chunks,
                st.session_state.embeddings,
                st.session_state.index,
                st.session_state.gen_model,
                st.session_state.tokenizer
            )

            st.subheader("✅ Answer:")
            st.success(answer)

            if evidence_ids:
                st.subheader("📄 Evidence Chunks:")
                for i in evidence_ids:
                    st.markdown(f"**Chunk {i}:** {st.session_state.chunks[i]}")

if __name__ == "__main__":
    main()











#for one question 
# # app.py
# import os
# import re
# import torch
# import faiss
# import streamlit as st
# from pypdf import PdfReader
# from InstructorEmbedding import INSTRUCTOR
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # 1. Load documents
# def load_documents(folder_path):
#     all_texts = []
#     for file in os.listdir(folder_path):
#         path = os.path.join(folder_path, file)
#         if file.endswith(".txt"):
#             with open(path, "r", encoding="utf-8") as f:
#                 content = f.read().strip()
#                 if content:
#                     all_texts.append(content)
#         elif file.endswith(".pdf"):
#             reader = PdfReader(path)
#             text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
#             if text.strip():
#                 all_texts.append(text)
#     return all_texts

# def fallback_sent_tokenize(text):
#     return re.split(r'(?<=[.?!])\s+', text.strip())

# def chunk_text(text, max_words=100):
#     sentences = fallback_sent_tokenize(text)
#     chunks, current, count = [], [], 0
#     for sent in sentences:
#         words = sent.split()
#         if count + len(words) > max_words:
#             if current:
#                 chunks.append(" ".join(current))
#             current, count = [], 0
#         current.append(sent)
#         count += len(words)
#     if current:
#         chunks.append(" ".join(current))
#     return chunks

# def embed_chunks(chunks, model):
#     instruction = "Represent the passage for retrieval:"
#     inputs = [[instruction, chunk] for chunk in chunks]
#     return model.encode(inputs, show_progress_bar=False)

# def build_faiss_index(embeddings):
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(embeddings)
#     return index

# def generate_answer(query, retriever, passages, embeddings, index, gen_model, tokenizer, top_k=3):
#     q_embed = retriever.encode([["Represent the question for retrieving supporting documents", query]])
#     scores, indices = index.search(q_embed, top_k)
#     top_chunks = sorted(set([i for i in indices[0] if i < len(passages)]))
#     context = "\n".join([f"[Chunk {i}]: {passages[i]}" for i in top_chunks])
#     prompt = f"question: {query} context: {context}"
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
#     outputs = gen_model.generate(**inputs, max_length=256)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True), top_chunks

# def main():
#     st.set_page_config(page_title="🧠 RAG Q&A App", layout="wide")
#     st.title("📚 Ask Questions from PDF/TXT using RAG")

#     upload_folder = "uploads"
#     os.makedirs(upload_folder, exist_ok=True)

#     # ⬆️ Avoid duplicate element error with `key`
#     uploaded_files = st.file_uploader(
#         "📤 Upload PDF or TXT files",
#         type=["pdf", "txt"],
#         accept_multiple_files=True,
#         key="file_uploader"
#     )

#     question = st.text_input("❓ Enter your question:", key="question_input")

#     if st.button("🧠 Get Answer", key="get_answer"):
#         if not uploaded_files:
#             st.warning("⚠️ Please upload at least one PDF or TXT file.")
#             return
#         if not question.strip():
#             st.warning("⚠️ Please enter a question.")
#             return

#         for file in uploaded_files:
#             with open(os.path.join(upload_folder, file.name), "wb") as f:
#                 f.write(file.read())

#         st.info("⏳ Loading models...")
#         retriever = INSTRUCTOR("hkunlp/instructor-base")
#         tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
#         gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

#         docs = load_documents(upload_folder)
#         chunks = [chunk for doc in docs for chunk in chunk_text(doc)]

#         st.info("🔍 Embedding & indexing...")
#         chunk_embeddings = embed_chunks(chunks, retriever)
#         faiss_index = build_faiss_index(torch.tensor(chunk_embeddings).numpy())

#         st.success("🧠 Generating Answer...")
#         answer, evidence_ids = generate_answer(
#             question, retriever, chunks, torch.tensor(chunk_embeddings).numpy(),
#             faiss_index, gen_model, tokenizer
#         )

#         st.subheader("✅ Answer:")
#         st.success(answer)

#         st.subheader("📄 Evidence:")
#         for i in evidence_ids:
#             st.markdown(f"**Chunk {i}:** {chunks[i]}")

# if __name__ == "__main__":
#     main()




