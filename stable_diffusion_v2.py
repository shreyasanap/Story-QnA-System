import fitz
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from diffusers import StableDiffusionPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings

def load_and_split_stories(folder):
    chunks, metadatas = [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for pdf_path in Path(folder).glob("*.pdf"):
        print(f"📄 Reading: {pdf_path}")
        doc = fitz.open(pdf_path)
        full_text = "".join(page.get_text() for page in doc)

        if full_text.strip():
            chunk_list = splitter.split_text(full_text)
            chunks.extend(chunk_list)
            metadatas.extend([{"source": pdf_path.name}] * len(chunk_list))
        else:
            print(f"⚠️ No text extracted from {pdf_path}")

    return chunks, metadatas

def store_in_chroma(chunks, metadatas, embed_model):
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name="story_chunks")

    print("🔍 Embedding and storing in Chroma...")
    embeddings = embed_model.encode(chunks, show_progress_bar=True).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return collection

def load_llm():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def generate_funny_response(llm, story_chunk, query):
    prompt = (
        f"You are a silly, sarcastic storyteller bot. Here's a part of a fairy tale:\n\n"
        f"{story_chunk[:500]}\n\n"
        f"A human asked: \"{query}\"\n"
        f"Give a short, funny response using the story above. End with an emoji:"
    )
    result = llm(prompt, max_new_tokens=50, do_sample=False)
    return result[0]["generated_text"]

def generate_image(pipe, prompt):
    print("🖼️ Generating image...")
    image = pipe(prompt).images[0]
    image.save("generated_image3.png")
    print("🖼️ Image saved as 'generated_image2.png'")

def main():
    print("🧠 Funny StoryBot CLI - Type 'exit' to quit.")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name="story_chunks")

    if collection.count() == 0:
        chunks, metadatas = load_and_split_stories("pdfs")
        if not chunks:
            return
        collection = store_in_chroma(chunks, metadatas, embed_model)

    llm = load_llm()
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break

        q_embedding = embed_model.encode([query])[0].tolist()
        results = collection.query(query_embeddings=[q_embedding], n_results=1)

        if results["documents"] and results["distances"][0][0] < 1.5:
            matched_chunk = results["documents"][0][0]
            source = results["metadatas"][0][0]["source"]
            response = generate_funny_response(llm, matched_chunk, query)
            print(f"\n📚 From: {source}")
            print("🤖:", response)

            try:
                generate_image(pipe, query)
            except Exception as e:
                print(f"⚠️ Image generation failed: {e}")
        else:
            print("\n🤖: That story must be in another galaxy... 🪐")

if __name__ == "__main__":
    main()
