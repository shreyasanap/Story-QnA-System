import fitz  # PyMuPDF
import faiss
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from diffusers import StableDiffusionPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_split_stories(folder):
    all_chunks = []
    source_map = []  # Maps chunks back to full story
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for pdf_path in Path(folder).glob("*.pdf"):
        print(f"📄 Reading: {pdf_path}")
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        if full_text.strip():
            chunks = splitter.split_text(full_text)
            all_chunks.extend(chunks)
            source_map.extend([pdf_path.name] * len(chunks))  # Track source file
        else:
            print(f"⚠️ No text extracted from {pdf_path}")

    return all_chunks, source_map

def embed_chunks(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings, chunks, model

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
    image.save("generated_image2.png")
    print("🖼️ Image saved as 'generated_image2.png'")

def main():
    print("🧠 Funny StoryBot CLI - Type 'exit' to quit.")
    chunks, sources = load_and_split_stories("pdfs")
    if not chunks:
        return

    index, embeddings, texts, emb_model = embed_chunks(chunks)
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

        q_embed = emb_model.encode([query])
        D, I = index.search(q_embed, k=1)
        similarity = D[0][0]

        if similarity < 1.5:  # tighter threshold for chunk-level match
            matched_chunk = texts[I[0][0]]
            source = sources[I[0][0]]
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
