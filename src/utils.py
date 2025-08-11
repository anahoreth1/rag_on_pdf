import faiss
import fitz
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from consts import *
from private_consts import GEMINI_API_KEY
from typing import List, Tuple, Any


def pdf_to_pages(pdf_path: str) -> List[str]:
    """
    Extracts text from each page of a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[str]: List of page texts.
    """
    doc = fitz.open(pdf_path)
    pages: List[str] = []
    for page in doc:
        pages.append(page.get_text().strip())
    return pages


def pdf_to_text(pdf_path: str) -> str:
    """
    Extracts and concatenates text from all pages of a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Full text of the PDF.
    """
    pages: List[str] = pdf_to_pages(pdf_path)
    full_text: str = ""
    for page in pages:
        full_text += page
    return full_text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Splits text into fixed-size chunks.

    Args:
        text (str): The text to split.
        chunk_size (int): Size of each chunk in characters.

    Returns:
        List[str]: List of text chunks.
    """
    chunks: List[str] = []
    start: int = 0
    while start < len(text):
        end: int = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def build_faiss_index(embeddings) -> faiss.IndexFlatL2:
    """
    Builds a FAISS index from embeddings.

    Args:
        embeddings: Numpy array of shape (n_samples, embedding_dim).

    Returns:
        faiss.IndexFlatL2: FAISS index for similarity search.
    """
    dim: int = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # Simple L2 distance index
    index.add(embeddings)
    return index


def init_for_rag() -> Tuple[Any, faiss.IndexFlatL2, List[str], List[str], Any]:
    """
    Initializes models and loads data for the RAG pipeline.

    Returns:
        Tuple containing:
            - st_model: SentenceTransformer model for embeddings.
            - index: FAISS index for retrieval.
            - chunks: List of document chunks.
            - metadata: List of metadata for chunks.
            - answering_model: Generative model for answer generation.
    """
    # Configure generative model
    genai.configure(api_key=GEMINI_API_KEY)
    answering_model = genai.GenerativeModel(MODEL_NAME)
    st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Load FAISS index and chunk data
    index = faiss.read_index(DATA_FAISS_INDEX)
    with open(DATA_CHUNKS_METADATA, "r", encoding="utf-8") as f:
        metadata: List[str] = [line.strip() for line in f]
    with open(DATA_CHUNKS_TEXT, "r", encoding="utf-8") as f:
        chunks: List[str] = [line.strip() for line in f]

    return (st_model, index, chunks, metadata, answering_model)
