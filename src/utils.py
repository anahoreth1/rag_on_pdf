import faiss
import fitz

from consts import CHUNK_SIZE


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
