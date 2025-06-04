from fastapi import FastAPI

app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Hello World"}

@app.get("/openapi.json")
def custom_openapi():
    return {"message": "Custom OpenAPI"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)