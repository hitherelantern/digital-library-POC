import re
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from pymilvus import connections, Collection,utility
from dotenv import load_dotenv
from langchain_core.prompts import load_prompt


from retriever import Retriever


load_dotenv()




import atexit
import shutil

# Function to delete the FAISS index directory
def cleanup():
    index_dir = "faiss_index"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
        st.success(f"Deleted {index_dir} directory successfully.")
    else:
        st.warning(f"{index_dir} directory does not exist.")



def cleanup_milvus(collection_name):
    # Connect to the Milvus server
    connections.connect("default", host="localhost", port="19530")
    
    # Check if the collection exists using utility.list_collections()
    existing_collections = utility.list_collections()
    if collection_name in existing_collections:
        collection = Collection(collection_name)
        collection.drop()  # Drops the collection and deletes its data
        st.success(f"Deleted collection '{collection_name}' successfully.")
    else:
        st.warning(f"Collection '{collection_name}' does not exist.")



def clean_text(text):
    """
    Cleans the text by removing or replacing problematic characters.
 
    Args:
        text (str): The text to clean.
 
    Returns:
        str: The cleaned text.
    """
    # 1. Remove surrogate pairs:
    text = text.encode('utf-8', 'replace').decode('utf-8')
 
    # 2. Remove any non-BMP characters (including emojis and some special symbols):
    text = ''.join(c for c in text if ord(c) <= 0xFFFF)
 
    # 3. Remove control characters
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
   
    return text
 
 
 
def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF files.
 
    Args:
        pdf_docs (list): A list of uploaded PDF files.
 
    Returns:
        str: The combined text from all PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += clean_text(page_text)  # Clean the text extracted from each page
    return text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])



def milvus_logic(query):

    collection = 'pdf_embeddings'
    search_params = {
        "metric_type": "COSINE",  # Use the same metric as during insertion
        "params": {"nprobe": 10}  # IVF search parameter
    }
    embedding_model = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    r = Retriever(collection,embedding_model,search_params)


    content,metadata = r.search(query,2)


    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

    system_prompt = load_prompt('template.json')


    prompt = system_prompt.invoke(
        {
            'retrieved_info':content,
            'query':query,
            'metadata':metadata
        }

    )

    result = model.invoke(prompt)


    st.write("Reply: ", result.content)


def main():
    st.set_page_config("Chat PDF - App")
    st.header("Chat with PDF using Gemini💁")

    # Main sidebar menu
    with st.sidebar:
        st.title("Select Mode:")
        mode = st.radio("Choose an option:", ["Upload and Query (FAISS)", "Query Existing Collection (Milvus)"])

    # Option 1: Upload Document and Query (FAISS)
    if mode == "Upload and Query (FAISS)":
        st.subheader("Upload a Document and Ask Questions")
        
        # File uploader for PDF
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        if st.button("Process and Save to FAISS"):
            if pdf_docs:
                with st.spinner("Processing and saving to FAISS..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing Complete! You can now query your uploaded document.")
            else:
                st.warning("Please upload at least one PDF file.")

        # Input for querying FAISS
        user_question = st.text_input("Ask a Question from Your Uploaded Document")
        if user_question:
            with st.spinner("Querying FAISS..."):
                user_input(user_question)

        # Option to clean up FAISS index
        if st.button("Clean Up FAISS Index"):
            with st.spinner("Cleaning up FAISS..."):
                cleanup()

    # Option 2: Query Existing Milvus Collection
    elif mode == "Query Existing Collection (Milvus)":
        st.subheader("Query Existing Collection in Milvus")
        
        milvus_collection_name = "pdf_embeddings"  # Replace with your actual collection name
        
        # Input for querying Milvus
        user_question = st.text_input("Ask a Question from the Milvus Collection")
        if user_question:
            with st.spinner("Querying Milvus..."):
                milvus_logic(user_question)  # Implement query logic here
        
        





if __name__ == "__main__":
    main()
