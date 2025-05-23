# digital-library
A digital library(Proof of Concept/ Prototype) using RAG Architecture ! 




# Chat PDF with Dual Vector Databases

## Overview

The **Chat PDF** application allows users to interact with PDF documents using natural language queries. The app leverages the power of two different vector databases—**FAISS** and **Milvus**—to provide flexible and efficient querying capabilities. Users can either upload their own documents for querying or interact with an existing collection of PDF embeddings.

---

## Features

1. **Upload and Query PDFs (FAISS)**  
   - Upload your own PDF documents.  
   - The application processes the PDFs, creates embeddings, and stores them in a local FAISS database.  
   - You can query the uploaded documents for specific information.

2. **Query Existing Collection (Milvus)**  
   - Query a pre-existing collection of PDF embeddings stored in a Milvus vector database.  
   - Ideal for datasets where embeddings are precomputed and shared.

---

## Tech Stack

- **Backend**: Python, [LangChain](https://www.langchain.com/), [Milvus](https://milvus.io/)  
- **Frontend**: [Streamlit](https://streamlit.io/)  
- **Vector Databases**: [FAISS](https://github.com/facebookresearch/faiss), Milvus  
- **Embedding Models**: Google Generative AI (Gemini)

---


