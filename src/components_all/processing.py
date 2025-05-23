from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from utils import clean_text  # Import the clean_text function
from components_all.utils import clean_text

def process_text_data(text: str) -> List[str]:
    """
    Processes the ingested text data.  This includes cleaning and chunking.

    Args:
        text (str): The raw text data.

    Returns:
        List[str]: A list of text chunks.

    Raises:
        TypeError: If text is not a string.
        ValueError: If text is empty.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not text:
        raise ValueError("text cannot be empty.")

    cleaned_text = clean_text(text)  # Use the clean_text function
    text_chunks = split_text_into_chunks(cleaned_text)
    return text_chunks



def split_text_into_chunks(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[str]:
    """
    Splits text into smaller chunks.

    Args:
        text (str): Input text.
        chunk_size (int, optional): Maximum size of each chunk. Defaults to 10000.
        chunk_overlap (int, optional): Overlap between adjacent chunks. Defaults to 1000.

    Returns:
        List[str]: List of text chunks.

     Raises:
        TypeError: If text is not a string, or chunk_size/chunk_overlap are not integers
        ValueError: If chunk_size is not greater than 0, or chunk_overlap is negative
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not isinstance(chunk_size, int) or not isinstance(chunk_overlap, int):
        raise TypeError("chunk_size and chunk_overlap must be integers.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)