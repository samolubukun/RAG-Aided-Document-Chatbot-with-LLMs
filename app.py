import streamlit as st
import os
from docx import Document
from PyPDF2 import PdfReader
from pptx import Presentation
from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the variables
import os
os.environ['HuggingFaceHub_API_Token'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['cohere_api_key'] = os.getenv('COHERE_API_KEY')

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

def extract_text_from_pptx(pptx_file):
    ppt_text = ""
    prs = Presentation(pptx_file)
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                ppt_text += shape.text + '\n'
    return ppt_text

def extract_text_from_docx(docx_file):
    doc_text = ""
    doc = Document(docx_file)
    for paragraph in doc.paragraphs:
        doc_text += paragraph.text + '\n'
    return doc_text

def process_documents(uploaded_files):
    all_text = ""
    for file in uploaded_files:
        if file.name.endswith('.pdf'):
            all_text += extract_text_from_pdf(file)
        elif file.name.endswith('.pptx'):
            all_text += extract_text_from_pptx(file)
        elif file.name.endswith('.docx'):
            all_text += extract_text_from_docx(file)
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=['\n', '\n\n', ' ', '']
    )
    chunks = text_splitter.split_text(text=all_text)
    
    # Create embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def generate_answer(question, vectorstore):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    
    prompt_template = """Answer the question as precise as possible using the provided context. If the answer is
                    not contained in the context, say "answer not available in context" \n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:"""
    
    prompt = PromptTemplate.from_template(template=prompt_template)
    
    cohere_llm = Cohere(model="command", temperature=0.1, cohere_api_key=os.getenv('cohere_api_key'))
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | cohere_llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(question)

# Streamlit UI
st.title("RAG-Aided Document Chatbot with LLMs")

# File upload
uploaded_files = st.file_uploader("Upload your documents (PDF, PPTX, DOCX)", 
                                 type=['pdf', 'pptx', 'docx'],
                                 accept_multiple_files=True)

# Process button
if uploaded_files and st.button("Process Documents"):
    with st.spinner("Processing documents..."):
        st.session_state.vectorstore = process_documents(uploaded_files)
    st.success("Documents processed successfully!")

# Question input
question = st.text_input("Ask a question about your documents:")

# Generate answer
if question and st.session_state.vectorstore is not None:
    with st.spinner("Generating answer..."):
        answer = generate_answer(question, st.session_state.vectorstore)
    st.write("Answer:", answer)
elif question:
    st.warning("Please upload and process documents first.")