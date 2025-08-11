import json
import os
import pandas as pd
import google.generativeai as genai
import tqdm

from consts import *
from private_consts import GEMINI_API_KEY
from utils import pdf_to_pages


def define_model():
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
    return model


def generate_prompt(chunk_text):
    prompt = f"""
        You are creating a test dataset for evaluating a RAG (Retrieval-Augmented Generation) system.
        Read the text below and generate 3 to 5 factual question-answer pairs.
        Only use the information explicitly stated in the text.
        Output in JSON array format: each element should be {{"question": "...", "answer": "..."}}.
        Do NOT include code fences, backticks, or any extra text.
        
        Text:
        \"\"\"{chunk_text}\"\"\"
        """

    return prompt


def generate_qa_from_chunk(
    chunk_text: str, model, chunk_id: int, pdf_name: str
) -> list[dict]:
    """
    Generates Q&A pairs for a given chunk of text using the provided model.

    Args:
        chunk_text (str): The text chunk to generate questions and answers from.
        model: The LLM model object with a .generate_content() method.
        chunk_id (int): The index of the chunk in the PDF.
        pdf_name (str): The name of the PDF file.

    Returns:
        list[dict]: List of dictionaries with keys 'question', 'answer', 'chunk_id', 'pdf_name'.
                    If parsing fails, returns a list with one dict containing the raw response.
    """
    prompt: str = generate_prompt(chunk_text)
    response = model.generate_content(prompt)
    raw_text: str = response.text.strip()

    # Remove optional code block markers like ```json ... ```
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")  # remove backticks
        # Sometimes format is like: json\n[ ... ]
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:].strip()

    try:
        qa_list: list = json.loads(raw_text)
        # Ensure it's a list of dicts with "question" and "answer"
        qa_list = [
            {
                "question": qa.get("question", "").strip(),
                "answer": qa.get("answer", "").strip(),
                "chunk_id": chunk_id,
                "pdf_name": pdf_name,
            }
            for qa in qa_list
            if isinstance(qa, dict)
        ]
        return qa_list
    except Exception:
        # If parsing failed, wrap raw text in a fallback
        return [
            {
                "question": "PARSE_ERROR",
                "answer": response.text.strip(),
                "chunk_id": chunk_id,
                "pdf_name": pdf_name,
            }
        ]


def generate_test_dataset():
    file_names = os.listdir(PDF_FOLDER)
    if IS_TEST:
        file_names = [TEST_FILE_NAME]

    model = define_model()

    all_entries = []
    for pdf_file in tqdm.tqdm(file_names):
        if pdf_file.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            pages = pdf_to_pages(pdf_path)
            if IS_TEST:
                pages = [pages[TEST_PAGE_NUMBER]]

            for idx, page in tqdm.tqdm(enumerate(pages)):
                try:
                    entries = generate_qa_from_chunk(page, model, idx, pdf_file)
                    all_entries.extend(entries)
                except Exception as e:
                    print(f"[ERROR] {pdf_file} chunk {idx}: {e}")

    all_entries_df = pd.DataFrame(all_entries)
    all_entries_df.to_csv(TEST_DATA_CSV, index=False, encoding="utf-8")
    print(f"Dataset saved to {TEST_DATA_CSV}, total {len(all_entries_df)} Q&A pairs")


if __name__ == "__main__":
    generate_test_dataset()
