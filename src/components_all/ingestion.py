from typing import List, Any
from PyPDF2 import PdfReader
import streamlit as st
# from utils import clean_text  # Import the clean_text function
from components_all.utils import clean_text


def ingest_pdf_data(pdf_docs: List[Any]) -> str:
    """
    Ingests PDF data from uploaded files. This is the first stage of the pipeline.

    Args:
        pdf_docs (List[Any]): List of uploaded PDF files. Use 'Any' for Streamlit file type.

    Returns:
        str: Extracted text from all PDF documents.  Returns empty string if no valid PDFs.

    Raises:
        TypeError: If pdf_docs is not a list.
        ValueError: If pdf_docs is empty.
    """
    if not isinstance(pdf_docs, list):
        raise TypeError("pdf_docs must be a list.")
    if not pdf_docs:
        raise ValueError("pdf_docs cannot be empty.")

    text = ""
    for pdf in pdf_docs:
        try:
            # Check file type before attempting to read with PdfReader
            if not pdf.name.lower().endswith(".pdf"):
                st.error(f"File '{pdf.name}' is not a PDF file. Please upload only PDF files.")
                continue  # Skip to the next file
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += clean_text(page_text)  # Use the clean_text function
        except Exception as e:
            st.error(f"An unexpected error occurred while processing file '{pdf.name}': {e}")
            raise  # Re-raise other exceptions to stop the pipeline
    return text

# def ingest_pdf_data(pdf_docs: List[Any]) -> str:
#     """
#     Ingests PDF data from uploaded files.  This is the first stage of the pipeline.

#     Args:
#         pdf_docs (List[Any]): List of uploaded PDF files. Use 'Any' for Streamlit file type.

#     Returns:
#         str: Extracted text from all PDF documents.

#     Raises:
#         TypeError: If pdf_docs is not a list.
#         ValueError: If pdf_docs is empty.
#     """
#     if not isinstance(pdf_docs, list):
#         raise TypeError("pdf_docs must be a list.")
#     if not pdf_docs:
#         raise ValueError("pdf_docs cannot be empty.")

#     text = ""
#     for pdf in pdf_docs:
#         try:
#             pdf_reader = PdfReader(pdf)
#             for page in pdf_reader.pages:
#                 page_text = page.extract_text() or ""
#                 text += clean_text(page_text)  # Use the clean_text function
#         except Exception as e:
#             st.error(f"Error processing PDF file {pdf.name}: {e}")
#             raise  # Re-raise the exception to stop the pipeline.
#     return text








