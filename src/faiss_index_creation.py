import faiss
import os
from sentence_transformers import SentenceTransformer

from consts import *
from utils import *


def create_faiss_index():
    # 1. Load the sentence transformer embedding model
    st_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 2. Read and process all PDFs in the folder
    all_chunks = []
    metadata = []  # To store the source filename for each chunk

    file_names = os.listdir(PDF_FOLDER)
    if IS_TEST:
        file_names = [TEST_FILE_NAME]

    for filename in file_names:
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(PDF_FOLDER, filename)
        text = pdf_to_text(path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        metadata.extend([filename] * len(chunks))

    print(f"Total chunks: {len(all_chunks)}")

    # 3. Generate embeddings for all text chunks
    embeddings = st_model.encode(
        all_chunks, show_progress_bar=True, convert_to_numpy=True
    )

    # 4. Build the FAISS index
    index = build_faiss_index(embeddings)
    print("FAISS index built.")

    # 5. Save the data
    os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)
    faiss.write_index(index, DATA_FAISS_INDEX)
    with open(DATA_CHUNKS_METADATA, "w", encoding="utf-8") as f:
        for meta in metadata:
            f.write(meta + "\n")
    with open(DATA_CHUNKS_TEXT, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk.replace("\n", " ") + "\n")
    print("The data was saved")


if __name__ == "__main__":
    create_faiss_index()
