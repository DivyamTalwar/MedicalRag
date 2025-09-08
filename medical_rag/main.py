"""
MAIN - FastAPI server for Medical RAG System
Ultra-simple production server with real API integrations
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import tempfile
import os

# Import our simple modules
from chat import ChatService
from search import SearchService
from pdf import PDFExtractor

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    sources_used: int

class StatusResponse(BaseModel):
    status: str
    message: str
    components: Dict[str, str]

# Initialize FastAPI app
app = FastAPI(
    title="Medical RAG System",
    description="Ultra-simple medical RAG with real API integrations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
print("[STARTUP] Initializing Medical RAG services...")

try:
    chat_service = ChatService()
    pdf_extractor = PDFExtractor()
    print("[OK] All services initialized successfully")
except Exception as e:
    print(f"[ERROR] Failed to initialize services: {e}")
    chat_service = None
    pdf_extractor = None

@app.get("/")
async def root():
    """System overview"""
    return {
        "system": "Medical RAG System",
        "version": "1.0.0",
        "status": "operational",
        "description": "Ultra-simple medical RAG with real API integrations",
        "endpoints": {
            "query": "POST /query - Ask medical questions with RAG",
            "chat": "POST /chat - Simple chat without RAG", 
            "upload": "POST /upload-pdf - Upload and index PDF",
            "status": "GET /status - System status",
            "docs": "GET /docs - API documentation"
        },
        "features": [
            "Real embedding API integration",
            "Real Pinecone vector database", 
            "Real LLM API responses",
            "PDF extraction and indexing",
            "Medical RAG pipeline"
        ]
    }

@app.get("/status")
async def get_status():
    """System status check"""
    components = {}
    overall_status = "healthy"
    
    # Check chat service
    if chat_service:
        try:
            # Test search service (part of chat service)
            stats = chat_service.search_service.get_stats()
            components["vector_database"] = f"Connected ({stats['total_vectors']} vectors)"
            components["chat_service"] = "Online"
        except:
            components["vector_database"] = "Error"
            components["chat_service"] = "Error"
            overall_status = "degraded"
    else:
        components["chat_service"] = "Offline"
        overall_status = "degraded"
    
    # Check PDF extractor
    components["pdf_extractor"] = "Ready" if pdf_extractor else "Error"
    
    return StatusResponse(
        status=overall_status,
        message="Medical RAG System Status",
        components=components
    )

@app.post("/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    """Main RAG query endpoint"""
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not available")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = chat_service.rag_query(request.question, request.top_k)
        
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            sources_used=result["sources_used"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/chat")
async def simple_chat(request: QueryRequest):
    """Simple chat without RAG"""
    if not chat_service:
        raise HTTPException(status_code=503, detail="Chat service not available")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        response = chat_service.simple_chat(request.question)
        return {"question": request.question, "answer": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and index PDF file"""
    if not pdf_extractor or not chat_service:
        raise HTTPException(status_code=503, detail="PDF service not available")
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Extract and index PDF
        success = pdf_extractor.extract_and_index(tmp_file_path, chat_service.search_service)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        if success:
            return {
                "message": f"Successfully processed {file.filename}",
                "filename": file.filename,
                "status": "indexed"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to process PDF")
            
    except Exception as e:
        # Clean up temp file if it exists
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not chat_service:
        return {"error": "Services not available"}
    
    try:
        stats = chat_service.search_service.get_stats()
        return {
            "vector_database": stats,
            "system_status": "operational",
            "endpoints_available": 6
        }
    except Exception as e:
        return {"error": f"Failed to get stats: {str(e)}"}

# Run server function
def run_server():
    print("\n" + "="*60)
    print("[MEDICAL RAG SYSTEM - ULTRA-SIMPLE EDITION]")
    print("="*60)
    print("[OK] Real Embedding API")
    print("[OK] Real Pinecone Database") 
    print("[OK] Real LLM API")
    print("[OK] PDF Processing")
    print("[OK] Complete RAG Pipeline")
    print("="*60)
    print("[STARTING] Server on http://localhost:8000")
    print("[DOCS] API docs: http://localhost:8000/docs")
    print("="*60)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    run_server()