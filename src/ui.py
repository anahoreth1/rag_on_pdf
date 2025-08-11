import streamlit as st

from model_answering import answer_with_model
from utils import init_for_rag


def get_answer(
    question: str, st_model, index, chunks, metadata, answering_model
) -> str:
    model_answer = answer_with_model(
        question, st_model, index, chunks, metadata, answering_model
    )

    return model_answer


def main():
    st_model, index, chunks, metadata, answering_model = init_for_rag()

    st.title("Knowledge Base QA System")

    user_question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if not user_question.strip():
            st.warning("Please enter a question before submitting.")
        else:
            answer = get_answer(
                user_question, st_model, index, chunks, metadata, answering_model
            )
            st.markdown(f"**Answer:** {answer}")


if __name__ == "__main__":
    main()
