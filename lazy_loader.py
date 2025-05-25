from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def lazy_load_and_chunk_pdfs(folder_path, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            print(f"📄 Loading: {filename}")
            loader = PyPDFLoader(path)
            docs = loader.load()
            chunks = splitter.split_documents(docs)

            for chunk in chunks:
                yield chunk
