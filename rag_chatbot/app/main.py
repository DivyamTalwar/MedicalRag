import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, message_to_dict, messages_from_dict

from rag_chatbot.app.services.agent.builder import build_medical_rag_agent
from rag_chatbot.app.services.agent.state import AgentState

app = FastAPI(title="Medical RAG Chatbot", version="4.0.1")

agent = build_medical_rag_agent()

class ChatRequest(BaseModel):
    question: str
    chat_history: List[Dict[str, Any]] = []

class ChatResponse(BaseModel):
    final_answer: str
    chat_history: List[Dict[str, Any]]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        conversation_history = messages_from_dict(request.chat_history)

        initial_state = {
            "query_state": {
                "original_query": request.question,
                "condensed_query": "",
                "chat_history": conversation_history,
                "medical_entities": {}
            },
            "search_state": {
                "dense_results": [],
                "sparse_results": [],
                "merged_candidates": [],
                "reranked_chunks": []
            },
            "context_state": {
                "parent_chunks": [],
                "assembled_context": "",
                "context_sufficiency": False,
                "medical_metadata": {}
            },
            "generation_state": {
                "final_answer": "",
                "is_streaming": False, # Changed to False for non-streaming response
            },
            "performance_state": {
                "node_timings": {},
                "total_duration": 0.0
            },
            "error_state": {
                "error_message": None,
                "failed_node": None
            },
            "sub_queries": []
        }
        
        result = agent.run(initial_state)
        
        final_answer = result.get("generation_state", {}).get("final_answer", "I'm sorry, but I couldn't generate a response.")
        
        # Update conversation history
        updated_history = conversation_history + [
            HumanMessage(content=request.question),
            AIMessage(content=final_answer)
        ]
        
        return ChatResponse(
            final_answer=final_answer,
            chat_history=[message_to_dict(msg) for msg in updated_history]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Medical RAG Chatbot",
        "version": "4.0.1",
        "agent_initialized": agent is not None
    }

@app.get("/")
async def root():
    return {
        "message": "Medical RAG Chatbot API",
        "version": "4.0.1",
        "endpoints": {
            "chat": "/chat",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
