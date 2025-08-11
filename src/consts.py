PDF_FOLDER = "../data/domain_data"  # Folder containing PDF files
TEST_DATA_FOLDER = "../data/test_data"
TEST_DATA_CSV = f"{TEST_DATA_FOLDER}/test_dataset.csv"


MODEL_NAME = "gemini-2.0-flash"  # Model name for Gemini API

CHUNK_SIZE = 500
TOP_K = 3

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PROCESSED_DATA_FOLDER = "../data/processed_data"
DATA_FAISS_INDEX = f"{PROCESSED_DATA_FOLDER}/faiss_index.bin"
DATA_CHUNKS_METADATA = f"{PROCESSED_DATA_FOLDER}/chunks_metadata.txt"
DATA_CHUNKS_TEXT = f"{PROCESSED_DATA_FOLDER}/chunks_text.txt"


# For testing purposes
IS_TEST = True  # Set to False for production use
TEST_FILE_NAME = "0a7bc290-5559-4b68-bd36-6d5834c8363a.pdf"
TEST_PAGE_NUMBER = 5
