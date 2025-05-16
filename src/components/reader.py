import os
import fitz
from pymilvus import Collection, connections,utility

class Reader:

    def __init__(self,pdf_document):
        self.pdf_document = pdf_document


    def isthere(self):
        # Step 1: Connect to Milvus
        connections.connect("default", host="localhost", port="19530")

        # Step 2: Check if Collection Exists
        collection_name = "pdf_embeddings"  # Replace with your Milvus collection name
        if collection_name not in utility.list_collections():
            print(f"Collection '{collection_name}' does not exist.")
            return False  # Collection does not exist

        # Step 3: Access the Collection
        collection = Collection(collection_name)

        # Step 4: Query for Metadata
        expr = "1 < page < 8"  # Modify this condition as needed
        try:
            results = collection.query(
                expr=expr,  # Query condition
                output_fields=["pdf"],  # Specify the "pdf" field
            )
        except Exception as e:
            print(f"Error querying collection: {e}")
            return False

        # Step 5: Extract Unique PDF Values
        pdf_values = {result["pdf"] for result in results if "pdf" in result}
        return os.path.basename(self.pdf_document) in pdf_values


    def extract_text(self):
        """
        Extracts text from a PDF, excluding page numbers typically found in headers or footers,
        and organizes the text by pages.

        Args:
        - pdf_document (str): Path to the PDF document.

        Returns:
        - pdf_text (dict): Dictionary with page numbers as keys and text content as values.
        """
        #.......
        "Note...some problem with the page extraction and formating here!"

        # Open the PDF file
        document = fitz.open(self.pdf_document)

        pdf_text = {}
        for page_number in range(document.page_count):
            page = document.load_page(page_number)

            # Get the text and its bounding boxes
            text_instances = page.get_text("dict")["blocks"]

            page_text = []
            for block in text_instances:
                bbox = block.get("bbox", [])
                text = block.get("lines", [])

                # Exclude text near the top or bottom (likely headers/footers)
                if bbox and (bbox[1] < 50 or bbox[3] > page.rect.height - 50):
                    continue

                # Collect text from the block
                for line in text:
                    page_text.append(" ".join([span["text"] for span in line["spans"]]))

            # Join all lines for the page and store in the dictionary
            pdf_text[page_number + 1] = "\n".join(page_text)

        # Close the document
        document.close()

        return pdf_text