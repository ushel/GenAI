from langchain.tools import tool
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import json
def get_retriever_tool(vectordb):
    retriever = vectordb.as_retriever()

# retriever = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings()).as_retriever()

    @tool
    def retrieve_docs(query: str) -> str:
        """Retrieve relevant documents for a given query."""
        docs = retriever.get_relevant_documents(query)
        return "\n".join([d.page_content for d in docs])
    
    return retrieve_docs


