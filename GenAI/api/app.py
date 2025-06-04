
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain.llms import OpenAI

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
app=FastAPI(
    title="Langchain Server",
    version="1.0",
    decsription="A simple API Server"

)
# app = FastAPI(
#     title="My Project",
#     openapi_url=f"/api/v1/openapi.json",
#     # docs_url=f"/api/v1/docs",
#     docs_url="/documentation",
#     # redoc_url=f"/api/v1/redoc",
#     redoc_url = None,
#     root_path="/app.py" # <------ This root_path fix the problem
# )
# app = FastAPI(
#     title="My App",
#     description="Description of my app.",
#     version="1.0",
#     docs_url='/docs',
#     openapi_url='/openapi.json', # This line solved my issue, in my case it was a lambda function
#     redoc_url=None,
#     root_path="/LLM/api/app"
# )


add_routes(
    app,
    OpenAI(),
    path="/openai"
)
model=OpenAI()
##ollama llama2
llm=Ollama(model="llama3.2")

prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2=ChatPromptTemplate.from_template("Write me an poem about {topic} for a 5 years child with 100 words")

add_routes(
    app,
    prompt1|model,
    path="/essay"

)

add_routes(
    app,
    prompt2|llm,
    path="/poem"


)


if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)
