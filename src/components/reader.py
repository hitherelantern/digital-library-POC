import os
import fitz
from pymilvus import Collection, connections

class Reader:

    def __init__(self,pdf_document):
        self.pdf_document = pdf_document


    def isthere(self):
        # Step 1: Connect to Milvus
        connections.connect("default", host="localhost", port="19530")

        # Step 2: Access the Collection
        collection_name = "pdf_embeddings"  # Replace with your Milvus collection name
        collection = Collection(collection_name)

        # Step 3: Query for Metadata
        # Retrieve all metadata for the "pdf" field

        expr = "1 < page < 8"
        results = collection.query(
            expr=expr,  # No filtering condition, fetch all documents
            output_fields=["pdf"] , # Specify the "pdf" field
        )

        # Step 4: Extract Unique PDF Values
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