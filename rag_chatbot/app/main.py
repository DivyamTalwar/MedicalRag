import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

from rag_chatbot.app.services.agent.builder import build_medical_rag_agent
from rag_chatbot.app.services.agent.state import AgentState

app = FastAPI(title="Medical RAG Chatbot", version="4.0.0")

agent = build_medical_rag_agent()

class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]] = []
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        chat_history = []
        for msg in request.chat_history:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
        
        initial_state = {
            "query_state": {
                "original_query": request.question,
                "condensed_query": "",
                "chat_history": chat_history,
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
                "rich_citations": [],
                "is_streaming": request.stream,
            },
            "performance_state": {
                "node_timings": {},
                "total_duration": 0.0
            },
            "error_state": {
                "error_message": None,
                "failed_node": None
            },
            "conversation_history": []
        }
        
        result = agent.run(initial_state)
        
        if request.stream:
            return StreamingResponse(
                result["generation_state"]["streaming_response"],
                media_type="text/event-stream"
            )
        
        final_answer = result["generation_state"]["final_answer"]
        citations = result["generation_state"]["rich_citations"]
        
        metadata = {
            "processing_time": result["performance_state"].get("total_duration", 0),
            "performance_metrics": result["performance_state"].get("node_timings", {})
        }
        
        return ChatResponse(
            answer=final_answer,
            citations=citations,
            metadata=metadata
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Medical RAG Chatbot",
        "version": "4.0.0",
        "agent_initialized": agent is not None
    }

@app.get("/")
async def root():
    return {
        "message": "Medical RAG Chatbot API",
        "version": "4.0.0",
        "endpoints": {
            "chat": "/chat",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
