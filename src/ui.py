import streamlit as st


def get_answer(question: str) -> str:
    """
    Placeholder function for answer generation.
    Will be replaced this with the RAG pipeline inference later.
    """
    # For now, just retranslate the
    return question


def main():
    st.title("Knowledge Base QA System")

    user_question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if not user_question.strip():
            st.warning("Please enter a question before submitting.")
        else:
            answer = get_answer(user_question)
            st.markdown(f"**Answer:** {answer}")


if __name__ == "__main__":
    main()
