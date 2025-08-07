import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from rag_chatbot.app.services.agent.builder import build_medical_rag_agent
from rag_chatbot.app.services.agent.state import AgentState

app = FastAPI(title="Medical RAG Chatbot", version="4.0.0")

conversation_history: List[BaseMessage] = []

agent = build_medical_rag_agent()

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
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
                "rich_citations": [],
                "is_streaming": True,
            },
            "performance_state": {
                "node_timings": {},
                "total_duration": 0.0
            },
            "error_state": {
                "error_message": None,
                "failed_node": None
            }
        }
        
        result = agent.run(initial_state)
        
        # Update conversation history
        conversation_history.append(HumanMessage(content=request.question))
        final_answer = result.get("generation_state", {}).get("final_answer", "")
        if final_answer:
            conversation_history.append(AIMessage(content=final_answer))

        streaming_response = result.get("generation_state", {}).get("streaming_response")
        
        if not streaming_response:
            # Fallback for non-streaming results
            async def string_generator(text):
                yield text
            return StreamingResponse(string_generator(final_answer), media_type="text/plain")

        return StreamingResponse(
            streaming_response,
            media_type="text/event-stream"
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
