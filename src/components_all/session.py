import streamlit as st
from typing import Dict, List, Any

def init_session_state() -> None:
    """Initializes the chat history and other session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {
            "Upload and Query (FAISS)": [],
            "Query Existing Collection (Milvus)": [],
        }
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "Upload and Query (FAISS)"
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "vector_store" not in st.session_state:  # Add this line
        st.session_state.vector_store = None



def display_chat(mode: str) -> None:
    """
    Displays the chat history in the Streamlit UI.

    Args:
        mode (str): The current mode ("Upload and Query (FAISS)" or
            "Query Existing Collection (Milvus)").
    """
    if not isinstance(mode, str):
        raise TypeError("mode must be a string.")
    if "chat_history" in st.session_state:
        for chat in st.session_state.chat_history[mode]:
            with st.chat_message("user"):
                st.markdown(f"**🧑 You:**\n{chat['user']}")
            with st.chat_message("ai"):
                st.markdown(f"**🤖 Gemini:**\n{chat['bot']}")


def construct_prompt(user_question: str, chat_history: List[dict]) -> str:
    """Construct a prompt that includes chat history."""
    history_text = "\n".join(
        [f"User: {entry['user']}\nBot: {entry['bot']}" for entry in chat_history]
    )
    prompt = f"{history_text}\nUser: {user_question}\nBot:"
    return prompt
