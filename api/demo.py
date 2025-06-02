# Set OpenAI API key from environment

# Set OpenAI API key from environment variable
# Set OpenAI API key from environment variable
from fastapi import FastAPI
from langserve import add_routes
from langchain.llms import OpenAI
import os

# Ensure OpenAI API key is set

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

# FastAPI app
app = FastAPI()

# @app.post("/openai")
# async def openai_invoke(prompt: str):
#     openai_model = OpenAI()
#     result = openai_model(prompt)
#     return {"response": result}

# add_routes(app, OpenAI(), path="/openai")
@app.post("/openai/invoke")
async def openai_invoke(prompt: str):
    openai_model = OpenAI()
    result = openai_model(prompt)
    return {"response": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)