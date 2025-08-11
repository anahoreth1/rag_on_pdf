import faiss
import fitz
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from consts import *
from model_answering import answer_with_model
from private_consts import GEMINI_API_KEY


def pdf_to_pages(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text().strip())
    return pages


def pdf_to_text(pdf_path):
    pages = pdf_to_pages(pdf_path)
    full_text = ""
    for page in pages:
        full_text += page
    return full_text


def chunk_text(text, chunk_size=CHUNK_SIZE):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # Simple L2 distance index
    index.add(embeddings)
    return index


def init_for_rag():
    # define models
    genai.configure(api_key=GEMINI_API_KEY)
    answering_model = genai.GenerativeModel(MODEL_NAME)
    st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # read saved data
    index = faiss.read_index(DATA_FAISS_INDEX)
    with open(DATA_CHUNKS_METADATA, "r", encoding="utf-8") as f:
        metadata = [line.strip() for line in f]
    with open(DATA_CHUNKS_TEXT, "r", encoding="utf-8") as f:
        chunks = [line.strip() for line in f]

    return (st_model, index, chunks, metadata, answering_model)
