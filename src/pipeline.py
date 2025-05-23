import streamlit as st

from components_all.ingestion import ingest_pdf_data

from components_all.processing import process_text_data

from components_all.vector_store import *

from components_all.session import construct_prompt, init_session_state, display_chat
 
# Constants for modes
MODE_FAISS = "Upload and Query (FAISS)"
MODE_MILVUS = "Query Existing Collection (Milvus)"
 
 
def main():
    """Main function to run the Streamlit application, orchestrating the pipeline."""
    st.set_page_config("Chat with PDF - Gemini")
    st.header("📄 Chat with PDF using Gemini")
 
    st.sidebar.title("⚙️ Configuration")
    mode = st.sidebar.radio("Select Mode:", [MODE_FAISS, MODE_MILVUS])
 
    init_session_state()
    st.session_state.current_mode = mode
 
    # Display the subheaders at the top, based on the selected mode
    if mode == MODE_FAISS:
        st.subheader("📤 Upload PDFs and Start Chatting")
    elif mode == MODE_MILVUS:
        st.subheader("🔍 Query from Milvus Collection")
 
 
 
    if mode == MODE_FAISS:
        pdf_docs = st.sidebar.file_uploader(
            "Upload your PDFs", accept_multiple_files=True
        )
 
        if pdf_docs:
            # Store the names of the uploaded files
            st.session_state.uploaded_files = [pdf.name for pdf in pdf_docs]
 
        if st.sidebar.button("Process and Save to FAISS"):
            if pdf_docs:
                try:
                    with st.spinner("Processing PDFs..."):
                        raw_text = ingest_pdf_data(pdf_docs)  # Stage 1: Ingestion
                        text_chunks = process_text_data(raw_text)  # Stage 2: Processing
                        vector_store = create_vector_store(
                            text_chunks
                        )  # Stage 3: Vector Store Creation
                        st.session_state.vector_store = (
                            vector_store
                        )  # Store for later queries
                        st.success("Documents processed and saved to FAISS!")
                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    # No return, allow user to try again.
            else:
                st.sidebar.warning("Please upload at least one PDF.")

        
 
        if st.sidebar.button("🧹 Clean Up FAISS Index"):
            with st.spinner("Deleting FAISS index..."):
                cleanup_vector_store()  # Stage 5: Cleanup
                st.session_state.pop(
                    "vector_store", None
                )  # Remove from session state
                st.rerun()

 
    if st.sidebar.button("🧼 Reset Chat"):
        st.session_state.chat_history[mode] = []
        st.success("Chat history cleared.")
        st.rerun()
    
    user_question = st.chat_input("Ask something...")
    if user_question:
        with st.spinner("Thinking..."):
            chat_history = st.session_state.chat_history[mode]
            prompt = construct_prompt(user_question, chat_history)
            
            if mode == MODE_FAISS:
                vector_store = st.session_state['vector_store']
                answer, duration = query_vector_store(vector_store, prompt)
            elif mode == MODE_MILVUS:
                answer, duration = query_milvus(prompt)
            
            # Update chat history
            st.session_state.chat_history[mode].append({"user": user_question, "bot": answer})
        display_chat(mode)

 
if __name__ == "__main__":
    main()
