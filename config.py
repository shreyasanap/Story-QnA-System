from key import api_key as TOGETHER_API_KEY
import os


CHROMA_DIR      = "./chroma_db"
PDF_PATHS       = [
    "pdfs/Alice_In_Wonderland.pdf",
    "pdfs/Gullivers_Travels.pdf",
    "pdfs/The_Arabian_Nights.pdf",
]
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_GEN_MODEL = "black-forest-labs/FLUX.1-schnell"


