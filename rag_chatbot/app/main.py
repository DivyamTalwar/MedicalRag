from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from llama_index.core.base.llms.types import ChatMessage
from app.services.query_engine.engine import QueryEngine

class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []

app = FastAPI(title="CIVIE RAG CHATBOT")

query_engine = QueryEngine()

@app.get("/")
def read_root():
    return {"message": "Welcome to the CIVIE RAG CHATBOT"}

@app.post("/chat")
def chat(request: ChatRequest):
    final_answer = query_engine.process_query(request.question, request.history)
    return {"answer": final_answer}
