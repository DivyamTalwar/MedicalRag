import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, message_to_dict, messages_from_dict

from rag_chatbot.app.services.query_engine.engine import QueryEngine

app = FastAPI(title="Medical RAG Chatbot", version="5.0.0")

query_engine = QueryEngine()

chat_history_store: List[Dict[str, Any]] = []

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    final_answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global chat_history_store
    try:
        conversation_history = messages_from_dict(chat_history_store)
        
        final_answer = await query_engine.process_query(request.question, chat_history=conversation_history)
        
        updated_history = conversation_history + [
            HumanMessage(content=request.question),
            AIMessage(content=final_answer)
        ]
        chat_history_store = [message_to_dict(msg) for msg in updated_history]
        
        return ChatResponse(
            final_answer=final_answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Medical RAG Chatbot",
    }

@app.get("/")
async def root():
    return {
        "message": "Medical RAG Chatbot API",}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
