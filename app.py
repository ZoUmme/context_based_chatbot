import os

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# PDF TO TEXT
def read_pdfs(pdf_files):
    all_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text
    return all_text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_text(text)


@st.cache_resource
def create_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_vector_store(chunks):
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = FAISS.from_documents(documents, create_embeddings())
    vector_store.save_local(INDEX_PATH)
    return vector_store


def load_vector_store():
    return FAISS.load_local(
        INDEX_PATH,
        create_embeddings(),
        allow_dangerous_deserialization=True,
    )


def retrieve_chunks(question, vector_store):
    return vector_store.similarity_search(question, k=10)


def get_prompt_and_llm():
    prompt_template = """
You are an AI assistant.
Answer the question using only the context below.
If the answer is not present in the context, say exactly:
"THE ANSWER IS NOT AVAILABLE IN THE PROVIDED CONTEXT."
The answer has to be in bullet point wise, each point not exceeding 200 words.
The answer has to be in a way that a CLASS 10TH GRADE STUDENT understands it.

Context:
{context}
Question:
{question}

Answer:
"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.7-flash",
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )
    return prompt, llm


def answer_question(question, vector_store):
    docs = retrieve_chunks(question, vector_store)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt, llm = get_prompt_and_llm()
    final_prompt = prompt.format(context=context, question=question)
    return llm.invoke(final_prompt).content


def main():
    st.set_page_config(page_title="Chat With Your PDFs", page_icon="📚")
    st.title("Chat With Your PDFs")
    st.caption("Upload a PDF, then ask questions answered from its content.")

    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY is not set. Add it to your .env file and restart the app.")
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Build knowledge base", type="primary"):
        with st.spinner("Reading PDFs and building the index..."):
            text = read_pdfs(uploaded_files)
            if not text.strip():
                st.error("No readable text was found in the uploaded PDFs.")
                st.stop()
            chunks = split_text(text)
            st.session_state.vector_store = build_vector_store(chunks)
        st.success(f"Knowledge base ready with {len(chunks)} text chunks.")

    vector_store = st.session_state.get("vector_store")
    index_files_exist = all(
        os.path.exists(os.path.join(INDEX_PATH, filename))
        for filename in ("index.faiss", "index.pkl")
    )
    if vector_store is None and index_files_exist:
        vector_store = load_vector_store()
        st.session_state.vector_store = vector_store

    if vector_store is None:
        st.info("Upload a PDF and build the knowledge base to begin.")
        return

    question = st.text_input("Ask a question about your PDF")
    if question:
        with st.spinner("Searching the document..."):
            answer = answer_question(question, vector_store)
        st.markdown(answer)


if __name__ == "__main__":
    main()