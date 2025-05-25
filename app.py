import os
from dotenv import load_dotenv
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# Load env variables from .env file
load_dotenv()

# Initialize Gemini API key from env
API_KEY = os.getenv("GEMINI_API_KEY")

# Load FAISS vector store and embedding model once
@st.cache_resource
def load_faiss_and_embeddings():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    return vector_store

vector_store = load_faiss_and_embeddings()

# Retrieve relevant docs with MMR search
def retrieve_relevant_docs(query, k=3):
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 10})
    docs = retriever.get_relevant_documents(query)
    return docs

# Build context text for prompt
def build_context_text(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Call Gemini API with context + user query
def generate_answer_with_gemini(query, context_text):
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    chat_session = model.start_chat()

    system_prompt = (
        "You are a witty AI storyteller 🤡. "
        "Answer the question based on the story snippets below, in a funny tone. "
        "If unrelated, say 'I don't know...' in a funny way.\n\n"
        f"Context:\n{context_text}"
    )
    chat_session.send_message(system_prompt)
    response = chat_session.send_message(query)
    return response.text

# Streamlit UI
st.title("The Pun Intended Ai🤖")

if API_KEY is None or API_KEY == "":
    st.error("API key not found. Please set GEMINI_API_KEY in your .env file.")
else:
    user_query = st.text_input("Ask a question about the stories:")

    if st.button("Get Funny Answer"):
        if not user_query.strip():
            st.warning("Please enter a question to ask.")
        else:
            with st.spinner("Fetching answer..."):
                try:
                    docs = retrieve_relevant_docs(user_query)
                    if not docs:
                        st.info("No relevant documents found.")
                    else:
                        context = build_context_text(docs)
                        answer = generate_answer_with_gemini(user_query, context)
                        st.markdown(f"### 🤡 Funny Answer:\n{answer}")
                except Exception as e:
                    st.error(f"Oops! Something went wrong: {e}")
