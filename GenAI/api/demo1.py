from fastapi import FastAPI, Body
# from langserve.validation import OpenAIInvokeRequest  # Assuming it's properly imported
from langchain.chat_models import ChatOpenAI
import os

# Set OpenAI API key (make sure it's loaded correctly)

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
app = FastAPI()

@app.post("/openai")
async def openai_invoke(request: OpenAIInvokeRequest = Body(...)):
    # Now you can access the prompt, max_tokens, and temperature from the request
    prompt = request.prompt
    max_tokens = request.max_tokens
    temperature = request.temperature
    
    # Your OpenAI processing logic here (e.g., ChatOpenAI model)
    
    return {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}