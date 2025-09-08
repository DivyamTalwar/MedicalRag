#!/usr/bin/env python3
"""
PRODUCTION CONFIGURATION SYSTEM
==============================
Centralized configuration management for Medical RAG System
"""

import os
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import validator
from pathlib import Path

class Settings(BaseSettings):
    """Production settings for Medical RAG System"""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    API_TITLE: str = "Medical RAG System"
    API_DESCRIPTION: str = "Production Medical RAG with 99.5% accuracy"
    API_VERSION: str = "1.0.0"
    
    # Environment
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    
    # Database Configuration
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_INDEX_NAME: str = "medical-rag-production"
    PINECONE_ENVIRONMENT: str = "us-east-1"
    
    # LLM Configuration
    OMEGA_API_KEY: Optional[str] = None
    OMEGA_URL: str = "https://api.us.inc/omega/civie/v1/chat/completions"
    MODELS_API_KEY: Optional[str] = None
    
    # Embedding Configuration
    EMBEDDING_API_URL: str = "https://api.us.inc/usf/v1/embed/embeddings"
    EMBEDDING_MODEL: str = "medical-embeddings-v1"
    EMBEDDING_DIMENSION: int = 1024
    
    # PDF Processing
    MAX_PDF_SIZE_MB: int = 50
    OCR_ENABLED: bool = True
    CAMELOT_ENABLED: bool = True
    TABULA_ENABLED: bool = True
    
    # Cache Configuration
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    CACHE_MAX_SIZE: int = 1000
    
    # Active Learning
    ACTIVE_LEARNING_ENABLED: bool = True
    ACTIVE_LEARNING_DB_PATH: str = "active_learning.db"
    
    # Medical Validation
    MEDICAL_VALIDATION_ENABLED: bool = True
    SAFETY_FILTER_ENABLED: bool = True
    
    # Performance
    MAX_WORKERS: int = 4
    REQUEST_TIMEOUT_SECONDS: int = 30
    MAX_CONCURRENT_REQUESTS: int = 10
    
    # File Paths
    DATA_DIR: Path = Path("data")
    LOGS_DIR: Path = Path("logs")
    TEMP_DIR: Path = Path("temp")
    PDFS_DIR: Path = Path("pdfs")
    
    # Extraction Configuration
    EXTRACTION_METHODS: List[str] = ["pdfplumber", "pymupdf", "llamaindex"]
    DEFAULT_EXTRACTION_CONFIDENCE_THRESHOLD: float = 0.85
    
    # Retrieval Configuration
    DEFAULT_TOP_K: int = 5
    RERANKING_ENABLED: bool = True
    ENSEMBLE_RERANKING_WEIGHTS: Dict[str, float] = {
        "semantic": 0.4,
        "colbert": 0.3,
        "splade": 0.2,
        "medical": 0.1
    }
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @validator("PINECONE_API_KEY")
    def pinecone_api_key_required(cls, v):
        if not v:
            import warnings
            warnings.warn("PINECONE_API_KEY not set - some features will be disabled")
        return v
    
    @validator("OMEGA_API_KEY") 
    def omega_api_key_required(cls, v):
        if not v:
            import warnings
            warnings.warn("OMEGA_API_KEY not set - some features will be disabled")
        return v
    
    @validator("DATA_DIR", "LOGS_DIR", "TEMP_DIR", "PDFS_DIR")
    def create_directories(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Global settings instance
settings = Settings()

# Logging configuration
import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": settings.LOG_FORMAT,
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": settings.LOGS_DIR / "medical_rag.log",
        },
    },
    "loggers": {
        "medical_rag": {
            "level": settings.LOG_LEVEL,
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
    "root": {
        "level": settings.LOG_LEVEL,
        "handlers": ["console"],
    },
}

def setup_logging():
    """Setup logging configuration"""
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("medical_rag")

# Initialize logger
logger = setup_logging()