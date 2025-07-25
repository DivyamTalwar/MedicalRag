from fastapi import FastAPI

app = FastAPI(title="RAG Chatbot API")

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Chatbot API"}
