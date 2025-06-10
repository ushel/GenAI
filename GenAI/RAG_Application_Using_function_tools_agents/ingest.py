from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.document_loaders import PyPDFLoader

def ingest():
    # texts = [
    #     Document(page_content="In 2024, Tesla launched a new robotaxi service in California."),
    #     Document(page_content="Tesla improved its Dojo supercomputer in 2024."),
    #     Document(page_content="The Cybertruck was announced in 2023 by Tesla.")
    # ]
    
    loader1 = PyPDFLoader("/Users/apple/Desktop/GenAI/GenAI/RAG_Application_Using_function_tools_agents/Data/attention.pdf")
    loader2 = PyPDFLoader("/Users/apple/Desktop/GenAI/GenAI/RAG_Application_Using_function_tools_agents/Data/temp.pdf")
    
    doc1 = loader1.load()
    doc2 = loader2.load()
    
    docs = doc1 + doc2

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    db = Chroma.from_documents(split_docs, embedding=OpenAIEmbeddings(), persist_directory="./db")
    db.persist()
    print("✅ Ingestion complete.")

    return db
