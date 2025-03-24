# from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from retriever import Retriever
from langchain_core.prompts import load_prompt
from dotenv import load_dotenv



if __name__ == "__main__":

    load_dotenv()


    collection = 'pdf_embeddings'
    search_params = {
        "metric_type": "COSINE",  # Use the same metric as during insertion
        "params": {"nprobe": 10}  # IVF search parameter
    }
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    r = Retriever(collection,embedding_model,search_params)


    query = input("Have a query!.....\n")
    content,metadata = r.search(query,1)


    model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')

    system_prompt = load_prompt('template.json')


    prompt = system_prompt.invoke(
        {
            'retrieved_info':content,
            'query':query,
            'metadata':metadata
        }

    )

    result = model.invoke(prompt)

    print(f"The answer is \n {result.content}")

    

