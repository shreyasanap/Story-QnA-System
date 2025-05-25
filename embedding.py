from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from lazy_loader import lazy_load_and_chunk_pdfs

def build_faiss_index(folder_path, index_save_path="faiss_index"):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = None

    for chunk in lazy_load_and_chunk_pdfs(folder_path):
        if vectorstore is None:
            vectorstore = FAISS.from_documents([chunk], embedding_model)
        else:
            vectorstore.add_documents([chunk])

    vectorstore.save_local(index_save_path)
    print(f"FAISS index saved at: {index_save_path}")
    return vectorstore

if __name__ == "__main__":
    folder = "pdfs"  
    build_faiss_index(folder, "faiss_index")
