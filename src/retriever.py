
from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
import time



class Retriever:
    def __init__(self, collection_name, embedding_model,search_params):
        

        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.search_params = search_params


    def search(self,query,k):
        
        
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
                search_params=self.search_params
            )
            start = time.time()
            result = vectorstore.similarity_search(query,k=k)
            end = time.time()
            metadata_list = [{'page': doc.metadata['page'], 'pdf': doc.metadata['pdf']} for doc in result]
            page_content_list = [doc.page_content for doc in result]
            print(f"Successfully retrieved {k} documents")
            print(f"Time taken to retrieve {k} documents is {end-start} seconds")
            return page_content_list,metadata_list
        
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve data from the db: {e}")
        



        
if __name__ == "__main__":


    collection = 'pdf_embeddings'
    search_params = {
        "metric_type": "COSINE",  # Use the same metric as during insertion
        "params": {"nprobe": 10}  # IVF search parameter
    }
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    r = Retriever(collection,embedding_model,search_params)


    query = "What is the middle class mindset?"
    content,metadata = r.search(query,1)
    print(f"content:{content}")
    print(f"metadata:{metadata}")