


# Overview 

In this project RAG architecture is used for getting answers from the pdf files dataset.


1. Generation of the test dataset, which will be used for comparing the models, is the next step.

2. RAG System

3. UI 

All the stages are tested in the notebook located in `notebooks/research.ipynb`

To 

# 1. Creating the test dataset

The logic is in `test_dataset_generation.py` script. This script generates a test dataset for evaluating RAG systems using the Google Gemini API. It processes PDF documents, extracts their text, and prompts Gemini to produce factual question–answer pairs strictly from each page. The output is cleaned, converted to JSON and saved into `.csv` file (`TEST_DATA_CSV`).


# 2. RAG System

TBA

# 3. UI 

TBA