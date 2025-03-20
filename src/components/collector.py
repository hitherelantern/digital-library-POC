import os 

class InfoCollector:
    """
    A class to embed text chunks.

    Attributes:
    - model (obj): Embedding model to be used.
    - all_embeddings (list): List of embeddings for all chunks.
    - all_metadata (list): List of metadata dictionaries corresponding to the chunks.
    """

    def __init__(self, model):
        """
        Initializes the Embedder class with an embedding model.

        Args:
        - model (str): Model name to be used for embedding.
        """
        self.model = model
        self.all_metadata = []

    def collect_metadata(self, page_chunks,pdf_document):
        """
        Embeds the text chunks using the specified model.

        Args:
        - page_chunks (dict): Dictionary where keys are page numbers and values are lists of chunks.

        Returns:
        - all_embeddings (list): List of embeddings for all chunks.
        - all_metadata (list): List of metadata dictionaries corresponding to the chunks.
        """
        # Initialize the embedding model
        # embedding_model = self.model
        # Generate embeddings and prepare metadata
        for page, chunks in page_chunks.items():
            # chunk_embeddings = embedding_model.embed_documents(chunks)
            chunk_metadata = [{"chunk_text": chunk, "page": page,'pdf':os.path.basename(pdf_document)} for chunk in chunks]

            self.all_metadata.extend(chunk_metadata)

        return self.all_metadata