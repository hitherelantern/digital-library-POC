from langchain_community.vectorstores import Milvus
from langchain.schema import Document
import time


class Writer:
    """
    A class to save embeddings and metadata into a Milvus vector database.

    Attributes:
    - collection_name (str): Name of the collection in the Milvus database.
    - embedding_model (HuggingFaceEmbeddings): The embedding model instance used for creating vector embeddings.
    """

    def __init__(self, collection_name, embedding_model,index_params):
        """
        Initializes the Writer class with a collection name and embedding model.

        Args:
        - collection_name (str): Name of the collection in the vector database.
        - embedding_model (HuggingFaceEmbeddings): The embedding model instance.
        """

        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.index_params = index_params or {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",  # Specified here for index creation
            "params": {"nlist": 128},
        }


    def save_to_vector_db(self,metadata):
        """
        Saves embeddings and metadata to a Milvus vector database.

        Args:
        - embeddings (list): List of chunk embeddings.
        - metadata (list): List of metadata dictionaries for each chunk.

        Returns:
        - vectorstore (Milvus): The Milvus vector store instance with the saved data.

        Raises:
        - ValueError: If embeddings or metadata are invalid.
        - RuntimeError: If saving to the database fails.
        """
        # Validate inputs
        # if not embeddings or not metadata:
        #     raise ValueError("Embeddings and metadata must not be empty.")

        # if len(embeddings) != len(metadata):
        #     raise ValueError("Embeddings and metadata must have the same length.")
        
        try:
            connection_args = {
                "host": "localhost",
                "port": "19530",
            }

            # Initialize the Milvus vector store
            vectorstore = Milvus(
                embedding_function = self.embedding_model,
                collection_name=self.collection_name,
                connection_args=connection_args,
                index_params=self.index_params,
                auto_id=True
            )

            # Convert metadata and embeddings into the required format
            documents = [
                Document(page_content=meta.pop("chunk_text"), metadata=meta)
                for meta in metadata
            ]

            # Add data to the vector store
            vectorstore.add_documents(documents)

            print(f"Successfully added {len(documents)} embeddings to collection '{self.collection_name}'.")
            return vectorstore

        except Exception as e:
            raise RuntimeError(f"Failed to save data to vector database: {e}")
        

    def delete(self,collection_name):

        """
        Removes a collection from the database.

        Args:
        - collection_name(str):Name of the collection to be deleted

        Returns:
        - None
        """

        # Initialize the Milvus vector store
        vectorstore = Milvus(collection_name=collection_name)

        # Perform deletion
        vectorstore.delete_collection()