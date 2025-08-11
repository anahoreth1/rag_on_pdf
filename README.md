# Project Documentation

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for answering questions using a dataset of PDF documents. The workflow includes automatic test dataset generation, semantic search, answer generation with LLMs, and a Streamlit-based user interface.

## Project Structure

- `src/`
  - `ui.py` — Streamlit UI for interactive question answering.
  - `utils.py` — Utility functions for PDF processing, chunking, embedding, and model initialization.
  - `model_answering.py` — Logic for retrieving relevant chunks and generating answers (see code for details).
  - `consts.py`, `private_consts.py` — Configuration constants and API keys.
  - `test_dataset_generation.py` — Script for generating test datasets from PDFs using Gemini API.

- `data/`
  - `domain_data/` — Source PDF files.
  - `test_data/` — Generated test datasets.
- `notebooks/`
  - `research.ipynb` — All experiments, code tests, and pipeline prototyping.
- `README.md` — Project documentation.


## 1. Test Dataset Generation

**Purpose:**  
Automatically generate factual question–answer pairs from PDF documents for model evaluation.

**How it works:**  
- Extracts text from each PDF page.
- Prompts Google Gemini API to generate 3–5 factual Q&A pairs per chunk.
- Cleans and parses the output to JSON.
- Saves the dataset as a CSV file (`TEST_DATA_CSV`).

**Usage:**  
Run the script:
```bash
python test_dataset_generation.py
```
See results in `data/test_data/test_dataset.csv`.


## 2. RAG System

**Purpose:**  
Answer user questions by retrieving relevant document chunks and generating answers using LLMs.

**Components:**
- **Embedding Model:** Used for semantic search (e.g., SentenceTransformer).
- **FAISS Index:** Fast similarity search over chunk embeddings.
- **Generative Model:** Google Gemini or other LLM for answer generation.

**Pipeline:**
1. User question is embedded.
2. Relevant chunks are retrieved from FAISS index.
3. Chunks and question are passed to the LLM to generate an answer.

**Initialization:**  
Handled by `init_for_rag()` in `utils.py`.


## 3. UI

**Purpose:**  
Provide a simple local interface for testing the RAG system.

**How to run:**  
From the `src` folder, execute:
```bash
streamlit run ui.py
```

**Features:**
- Input field for user questions.
- Button to generate answers.
- Display of model-generated answers.

## 4. Notebooks

All stages and experiments are documented in `notebooks/research.ipynb`.  
Use this notebook for prototyping, debugging, and step-by-step exploration.

## Configuration

- **API Keys:**  
  Store your Gemini API key in `private_consts.py`.
- **Model Names and Paths:**  
  Set in `consts.py`. 
- **Debugging:**  
  Flag `IS_TEST` enables testing and debugging mode by limiting the RAG pipeline to a single document. This helps speed up development and troubleshooting without processing the entire knowledge base.

## Requirements

- Python 3.10+
- `streamlit`
- `faiss`
- `fitz` (PyMuPDF)
- `sentence-transformers`
- `google-generativeai`
- Other dependencies as listed in the `requirements.txt`

## How to Use

1. **Configuration:** 
    * Install dependencies from `requirements.txt` 
    * Store your Gemini API key in `private_consts.py`
    * Extract source pdf files to the `data/domain_data` folder 
2. **Generate test dataset:**  
   Run `test_dataset_generation.py` to create Q&A pairs from PDFs.
3. **Start UI:**  
   Run `streamlit run ui.py` and ask questions.
4. **Experiment:**  
   Use `notebooks/research.ipynb` for custom tests and development.

## Future enchancements
* Improve retrieval quality with better embeddings, hybrid search, and smarter chunking.
* Add semantic evaluation metrics for more accurate answer validation.
* Enhance UI with file uploads, batch testing, and user feedback collection.
* Enable multilingual support for broader usability.