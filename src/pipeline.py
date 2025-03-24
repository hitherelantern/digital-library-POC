import time
from langchain_huggingface import HuggingFaceEmbeddings
from components.reader import Reader
from components.preprocessor import Preprocessor
from components.collector import InfoCollector
from components.writer import Writer
import pandas as pd,os

def pipeline(pdf_path):
    start = time.time()
    # Step 1: Read the PDF
    print("Reading the PDF...")
    reader = Reader(pdf_path)
    if reader.isthere():
        print(f"{os.path.basename(pdf_path)} is already in the Database")
        return 
    pdf_text = reader.extract_text()

    # Step 2: Preprocess the text
    print("Preprocessing the text...")
    preprocessor = Preprocessor(pdf_text)
    pdf_text = preprocessor.clean_text()
    page_chunks = preprocessor.text_splitting(chunk_size=1000,chunk_overlap=200)
  
    # chunk_data = []
    # for page, chunks in page_chunks.items():
    #     for chunk in chunks:
    #         chunk_data.append({"page": page, "chunk": chunk})

    # Embedding model to be used.
    embedding_model_instance = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 3: Embed the text
    print("Embedding the text...")
    collector = InfoCollector(model=embedding_model_instance)
    metadata = collector.collect_metadata(page_chunks,pdf_path)

    # for i in range(len(chunk_data)):
    #     chunk_data[i]["embedding"] = embeddings[i]
    
    # # Create a DataFrame
    # df = pd.DataFrame(chunk_data)
    # df.to_excel('chunks.xlsx', index=False)


    # Step 4: Write to database
    print("Writing to the database...")
    
    writer = Writer(
        collection_name="pdf_embeddings",
        embedding_model=embedding_model_instance,
        index_params=None
    )
    
    writer.save_to_vector_db(metadata)
    end = time.time()

    print("Pipeline completed successfully!")
    print(f"pipeline completed within {end - start:.4f} seconds")

# Execute the pipeline
if __name__ == "__main__":
    path = os.path.join(r"..\Business,finance and economics")
    files = os.listdir(path)
    
    for file in range(len(files)):
        print(os.path.basename(files[file]))
        # print(os.path.join(path, files[file]))
        pipeline(os.path.join(path, files[file]))
        

    

