from langchain.chains import RetrievalQA,retrieval_qa
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import load_prompt
from components.retriever import Retriever
from dotenv import load_dotenv

load_dotenv()
EMBEDDING_MODEL_NAME = "models/embedding-001"  # Define the embedding model name





def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context.

    Context:
    {context}
    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)
