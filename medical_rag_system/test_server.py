#!/usr/bin/env python3
"""
MINIMAL TEST SERVER
==================
Basic FastAPI server without heavy system initialization
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Core imports
from app.core.config import settings

# Initialize minimal FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Medical RAG System - Minimal Test Server", "status": "running"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "environment": settings.ENVIRONMENT
    }

def run_server():
    """Run the server"""
    print(f"Starting minimal test server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(
        "test_server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        access_log=True
    )

if __name__ == "__main__":
    run_server()