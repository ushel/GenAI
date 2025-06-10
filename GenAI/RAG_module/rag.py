import os
import requests
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Prompts
tag_prompt = ChatPromptTemplate.from_template("""
Extract structured metadata as JSON from the question below:

{query}
""")

qa_prompt = ChatPromptTemplate.from_template("""
Use only the context below to answer the question.
If the answer is not directly in the context, use reasoning to infer a likely but informed answer.

Context:
{context}

Question: {input}
""")

def download_pdf(url: str, local_path: str):
    response = requests.get(url)
    with open(local_path, 'wb') as f:
        f.write(response.content)

def get_answer(query: str) -> str:
    # Step 1: Download and Load PDF
    # url = "https://openreview.net/pdf?id=VtmBAGCN7o"
    # local_pdf = "temp.pdf"
    # download_pdf(url, local_pdf)
    loader1 = PyPDFLoader("/Users/apple/Desktop/GenAI/GenAI/Tagging_extraction_rag/attention.pdf")
    loader2 = PyPDFLoader("/Users/apple/Desktop/GenAI/GenAI/Tagging_extraction_rag/temp.pdf")
    docs1 = loader1.load()
    docs2 = loader2.load()
    
    docs = docs1 + docs2

    # Step 2: Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    # Step 3: Embed and persist
    persist_dir = "./db"
    db = Chroma.from_documents(split_docs, embedding=OpenAIEmbeddings(), persist_directory=persist_dir)
    db.persist()
    retriever = db.as_retriever(search_kwargs={"k": 6})

    # Step 4: Setup LLM
    llm = ChatOpenAI(temperature=0)

    # Step 5: Tag extraction (Optional, for debugging or metadata)
    tag_response = llm(tag_prompt.format_prompt(query=query).to_messages())
    tags_text = tag_response.content.strip()
    print("Extracted tags:", tags_text)

    # Step 6: Document retrieval
    relevant_docs = retriever.get_relevant_documents(query)
    combined_context = "\n".join(doc.page_content for doc in relevant_docs)

    # Step 7: QA
    qa_response = llm(qa_prompt.format_prompt(input=query, context=combined_context).to_messages())

    # Clean up
    # os.remove(local_pdf)

    return qa_response.content.strip()
