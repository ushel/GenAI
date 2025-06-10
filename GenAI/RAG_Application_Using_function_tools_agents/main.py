# main.py

import os
from ingest import ingest
from tools import get_retriever_tool
from agent import create_agent
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
import streamlit as st

if __name__ == "__main__":
    # Define your PDF paths
    # pdf_paths = ["/Users/apple/Desktop/GenAI/GenAI/RAG_Application_Using_function_tools_agents/Data/attention.pdf", "/Users/apple/Desktop/GenAI/GenAI/RAG_Application_Using_function_tools_agents/Data/temp.pdf"]
    # if not all(os.path.exists(p) for p in pdf_paths):
    #     print("❌ One or more PDF files not found in ./papers/")
    #     exit()

    load_dotenv()
    os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    with st.spinner("🔎 Loading documents..."):
    # Step 1: Ingest PDFs and create vector store
        vectordb = ingest()

    # Step 2: Get retrieval tool
        retrieve_tool = get_retriever_tool(vectordb)

    # Step 3: Initialize agent
        agent = create_agent(retrieve_tool,memory=st.session_state.memory)


    st.subheader("💬 Ask something about the documents:")
    query = st.chat_input("Type your question here...")
    # Step 4: Ask question
    # query = input("\n💬 Ask a question about the papers: ")
    # response = agent.run(query)

    # print("\n🧠 Response:\n", response)
    if query:
        st.chat_message("user").write(query)
        with st.spinner("🤖 Thinking..."):
            response = agent.run(query)
        st.chat_message("ai").write(response)
