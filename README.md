# RAG-Aided Document Chatbot with LLMs

A powerful document question-answering system that uses Retrieval-Augmented Generation (RAG) to process documents and generate answers based on their content, leveraging large language models (LLMs).

![Screenshot (295)](https://github.com/user-attachments/assets/fb821916-e45d-4662-8de1-23bfd2bfafce)

![Screenshot (292)](https://github.com/user-attachments/assets/59c6da47-f4e6-4a90-b11e-6d2166f2cf9c)

![Screenshot (293)](https://github.com/user-attachments/assets/2497519b-1a94-4fc2-89b8-e5d6cd05ed32)

![Screenshot (294)](https://github.com/user-attachments/assets/6f436e43-55eb-47e8-a9b9-52af709a4393)




## Features
- Upload documents in **PDF**, **PPTX**, or **DOCX** formats.
- **RAG (Retrieval-Augmented Generation)** to extract relevant information from documents.
- Uses **Cohere** LLM for answering questions based on the provided document content.
- Vectorization of document content using **HuggingFaceEmbeddings** and **FAISS**.
- Text processing and chunking for efficient information retrieval from large documents.

## Technologies
- **Streamlit** – Interactive web framework for building the app.
- **LangChain** – Toolkit for chaining together various components in LLMs, vector stores, and prompt engineering.
- **FAISS** – Efficient similarity search and clustering of document embeddings.
- **Cohere** – API for powerful language models to generate answers based on the document context.
- **HuggingFace Embeddings** – For creating dense vector embeddings of text.
- **Python-docx, PyPDF2, python-pptx** – Libraries for extracting text from DOCX, PDF, and PPTX files.

## Installation

### Prerequisites
Before running the app, make sure you have Python 3.8 or higher installed.

### 1. Clone the repository:
```bash
git clone https://github.com/samolubukun/RAG-Document-Chatbot.git
cd rag-aided-doc-chatbot
```

### 2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables:
Create a `.env` file in the project directory and add your API keys for the services used. Example:

```
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
GOOGLE_API_KEY=your_google_api_key
COHERE_API_KEY=your_cohere_api_key
```

### 5. Run the app:
```bash
streamlit run app.py
```

Your browser should automatically open the Streamlit app.

## How to Use
1. **Upload documents**: Click the upload button to upload multiple documents in PDF, PPTX, or DOCX formats.
2. **Process documents**: After uploading, click "Process Documents" to process and index the content.
3. **Ask questions**: Enter your question in the input box, and the system will provide answers based on the content of the uploaded documents.

