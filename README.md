


# Overview 

# Overview

This project uses the RAG architecture to obtain answers from a dataset of PDF files.

1. The next step is the generation of the test dataset, which will be used to compare the models.

2. RAG System.

3. A UI based on Streamlit is provided to demonstrate the functionality. To run the UI, execute the Streamlit script inside the `src` folder:  
```streamlit run ui.py```


All stages have been tested in the notebook located at  `notebooks/research.ipynb`


# 1. Creating the test dataset

The logic is implemented in the `test_dataset_generation`.py script. This script generates a test dataset for evaluating RAG systems using the Google Gemini API. It processes PDF documents, extracts their text, and prompts Gemini to produce factual question–answer pairs strictly from each page. The output is cleaned, converted to JSON format, and saved as a `.csv` file (`TEST_DATA_CSV`).


# 2. RAG System

TBA

# 3. UI 

The UI provides a simple way to test the functionality locally.
Streamlit script is used.
