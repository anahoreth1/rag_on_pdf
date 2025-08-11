import streamlit as st

from model_answering import answer_with_model
from utils import init_for_rag


def get_answer(
    question: str, st_model, index, chunks, metadata, answering_model
) -> str:
    """
    Generates an answer to the user's question using the provided models and data.

    Args:
        question (str): The user's question.
        st_model: The semantic search model.
        index: The vector index for retrieval.
        chunks: The document chunks.
        metadata: Metadata for the chunks.
        answering_model: The LLM for answer generation.

    Returns:
        str: The generated answer.
    """
    model_answer: str = answer_with_model(
        question, st_model, index, chunks, metadata, answering_model
    )
    return model_answer


def main() -> None:
    """
    Streamlit UI entry point for the Knowledge Base QA System.
    Initializes models and handles user interaction.
    """
    # Initialize models and data for RAG pipeline
    st_model, index, chunks, metadata, answering_model = init_for_rag()

    st.title("Knowledge Base QA System")

    # Input field for user's question
    user_question: str = st.text_input("Enter your question:")

    # Button to trigger answer generation
    if st.button("Get Answer"):
        if not user_question.strip():
            st.warning("Please enter a question before submitting.")
        else:
            answer: str = get_answer(
                user_question, st_model, index, chunks, metadata, answering_model
            )
            st.markdown(f"**Answer:** {answer}")


if __name__ == "__main__":
    main()
