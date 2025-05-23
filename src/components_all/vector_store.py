import os
import shutil
import streamlit as st
from typing import Optional, List
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pymilvus import connections, Collection, utility
from components_all.llm_chain import get_conversational_chain
from components.retriever import Retriever
from langchain_core.prompts import load_prompt
import time
from dotenv import load_dotenv

load_dotenv()

# Constants
FAISS_INDEX_DIR = "faiss_index"
MILVUS_COLLECTION_NAME = "pdf_embeddings1"
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
EMBEDDING_MODEL_NAME = "models/embedding-001"  # Define the embedding model name


# FAISS Functions
def create_vector_store(text_chunks: List[str]) -> FAISS:
    """
    Creates a FAISS index from text chunks.

    Args:
        text_chunks (List[str]): List of text chunks.

    Returns:
        FAISS: The created FAISS index.

    Raises:
        TypeError: If text_chunks is not a list.
        ValueError: If text_chunks is empty.
    """
    if not isinstance(text_chunks, list):
        raise TypeError("text_chunks must be a list.")
    if not text_chunks:
        raise ValueError("text_chunks cannot be empty.")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_DIR)
    return vector_store


def load_vector_store(index_dir: str = FAISS_INDEX_DIR) -> Optional[FAISS]:
    """
    Loads the FAISS index from disk.

    Args:
        index_dir (str, optional): Directory where the FAISS index is stored.
            Defaults to FAISS_INDEX_DIR.

    Returns:
        Optional[FAISS]: Loaded FAISS index, or None if it doesn't exist.

     Raises:
        TypeError: If index_dir is not a string
    """
    if not isinstance(index_dir, str):
        raise TypeError("index_dir must be a string")

    if not os.path.exists(os.path.join(index_dir, "index.faiss")):
        return None
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)



def query_vector_store(vector_store: FAISS, user_question: str) -> str:
    """
    Queries the FAISS index and retrieves an answer.

    Args:
        vector_store (FAISS): FAISS index to query.
        user_question (str): The query.

    Returns:
        str: The answer from the FAISS index.

    Raises:
        TypeError: If user_question is not a string.
        ValueError: If user_question is empty.
    """
    if not isinstance(user_question, str):
        raise TypeError("user_question must be a string.")
    if not user_question:
        raise ValueError("user_question cannot be empty.")
    if vector_store is None:
        return "Vector DB not found. Please upload documents first.", 0

    docs = vector_store.similarity_search(user_question)
    if not docs:
        return "No matching documents found.", 0

    chain = get_conversational_chain() #moved to llm_chain.py
    start = time.time()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    duration = time.time() - start
    return response["output_text"], duration


def cleanup_vector_store(index_dir: str = FAISS_INDEX_DIR) -> None:
    """
    Cleans up the FAISS index files.

    Args:
        index_dir (str, optional): Directory where the FAISS index is stored.
            Defaults to FAISS_INDEX_DIR.
    """
    if not isinstance(index_dir, str):
        raise TypeError("index_dir must be a string.")

    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
        st.success(f"Deleted {index_dir} directory successfully.")
    else:
        st.warning(f"{index_dir} directory does not exist.")


# Milvus Functions
def cleanup_milvus(collection_name: str = MILVUS_COLLECTION_NAME) -> None:
    """
    Cleans up the Milvus collection.

    Args:
        collection_name (str, optional): Name of the Milvus collection to delete.
            Defaults to MILVUS_COLLECTION_NAME.
    """
    if not isinstance(collection_name, str):
        raise TypeError("collection_name must be a string.")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        collection.drop()
        st.success(f"Deleted collection '{collection_name}' successfully.")
    else:
        st.warning(f"Collection '{collection_name}' does not exist.")



def query_milvus(query: str) -> str:
    """
    Queries the Milvus collection and retrieves an answer.

    Args:
        query (str): The query.

    Returns:
        str: The answer from the Milvus collection.

    Raises:
        TypeError: If query is not a string.
        ValueError: If query is empty.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string.")
    if not query:
        raise ValueError("query cannot be empty.")

    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = MILVUS_COLLECTION_NAME
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 20}}
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    r = Retriever(collection, embedding_model, search_params)
    content, metadata = r.search(query, 20)
    metadata_str = ", ".join([f"{k}: {v}" for k, v in metadata.items()]) if isinstance(metadata, dict) else str(metadata)

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3) #moved to llm_chain.py
    system_prompt = load_prompt("template.json") #moved to llm_chain.py
    prompt = system_prompt.invoke({'retrieved_info': content, 'query': query, 'metadata': metadata_str}) #moved to llm_chain.py
    start = time.time()
    result = model.invoke(prompt)
    duration = time.time() - start
    return result.content, duration