from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"


# Prompt Template
prompt = ChatPromptTemplate.from_messages(
  [
   ("system","You are a helpful assistant. Please response to the quries"),
   ("user","Question:{question}")
  ]
 )

#streamlit framework

st.title("Langchain Demo with Llama3.1")
input_text = st.text_input("Search topic you want")

#LLM Model
llm = Ollama(model="llama3.2")
output_parser = StrOutputParser()

chain = prompt|llm|output_parser


if input_text:
    st.write(chain.invoke({'question':input_text}))