import re,fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter



class Preprocessor:
    """
    A class to preprocess a PDF document by extracting text page-wise and chunking it into smaller sections.

    Attributes:
    - pdf_document (str): Path to the PDF document.
    - pdf_text (dict): Dictionary containing extracted text for each page.
    - page_chunks (dict): Dictionary containing chunks for each page after splitting.
    """

    def __init__(self, pdf_text):
        """
        Initializes the Preprocessor class with the PDF document path.

        Args:
        - pdf_document (str): Path to the PDF document.
        """
        self.pdf_text = pdf_text  # Initialize with the dictionary extracted using Reader class in reader.py
        self.page_chunks = {}  # Initialize an empty dictionary for page chunks

    

    def clean_text(self):
        """
        Cleans the text by removing extra whitespace, newlines, and special characters,
        while retaining single quotes (') and double quotes (").
        Updates self.pdf_text with the cleaned text for each page.

        Returns:
        - dict: The updated self.pdf_text dictionary with cleaned text.

        Raises:
        - ValueError: If self.pdf_text is not a dictionary or contains invalid values.
        """
        if not isinstance(self.pdf_text, dict):
            raise ValueError("self.pdf_text must be a dictionary. Call `extract_text` first.")

        for page_number, text in self.pdf_text.items():
            try:
                if not isinstance(text, str):
                    raise ValueError(f"Text for page {page_number} is not a string: {type(text)} found.")

                # Clean the text
                text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
                text = text.strip()  # Remove leading and trailing spaces
                text = re.sub(r'[^\w\s\.\'"]', ' ', text)  # Remove special characters except ., ', and "
                
                # Update the dictionary
                self.pdf_text[page_number] = text

            except Exception as e:
                print(f"Error processing page {page_number}: {e}")
                self.pdf_text[page_number] = ""  # Replace problematic text with an empty string

        return self.pdf_text
    


    def text_splitting(self, chunk_size, chunk_overlap):
        """
        Splits the extracted text into smaller chunks using a RecursiveCharacterTextSplitter.

        Args:
        - chunk_size (int, optional): The maximum size of each chunk. Defaults to 500.
        - chunk_overlap (int, optional): The overlap size between chunks. Defaults to 50.

        Returns:
        - page_chunks (dict): Dictionary containing chunks for each page.
        """
        
        self.pdf_text = self.clean_text()
        if not self.pdf_text:
            raise ValueError("Text has not been extracted. Call `extract_text` first.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for page, text in self.pdf_text.items():
            chunks = text_splitter.split_text(text)  # Split the text into chunks
            self.page_chunks[page] = chunks

        return self.page_chunks







