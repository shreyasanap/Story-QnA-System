import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import CHROMA_DIR, EMBED_MODEL

_cached_db = None

def get_vector_db(paths):
    """Load or build a Chroma store for the supplied PDFs."""
    global _cached_db
    if _cached_db:
        return _cached_db

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        _cached_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL))
        return _cached_db

    docs = []
    for p in paths:
        docs.extend(PyPDFLoader(p).load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100).split_documents(docs)
    _cached_db = Chroma.from_documents(chunks, embedding=HuggingFaceEmbeddings(model_name=EMBED_MODEL), persist_directory=CHROMA_DIR)
    return _cached_db
