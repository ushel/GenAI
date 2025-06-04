from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langserve import add_routes
import uvicorn
from langchain_community.llms import Ollama
from langchain.llms import OpenAI
from pydantic import BaseModel
import os



os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

# Define the request body models
class InputText(BaseModel):
    topic: str

# Initialize OpenAI model
model = OpenAI()

# Initialize Ollama model
llm = Ollama(model="llama3.2")

# Define prompts for essay and poem
prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} for a 5-year-old child with 100 words")

# Routes for essay and poem generation
@app.post("/essay/invoke")
async def generate_essay(input: InputText):
    response = await model.agenerate([prompt1.format(topic=input.topic)])
    return {"output": response[0]}

@app.post("/poem/invoke")
async def generate_poem(input: InputText):
    response = await llm.agenerate([prompt2.format(topic=input.topic)])
    return {"output": response[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)