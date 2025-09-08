#!/usr/bin/env python3
"""
CUSTOM EXCEPTIONS
================
Medical RAG System specific exceptions
"""

class MedicalRAGException(Exception):
    """Base exception for Medical RAG System"""
    pass

class PDFExtractionError(MedicalRAGException):
    """Raised when PDF extraction fails"""
    pass

class EmbeddingError(MedicalRAGException):
    """Raised when embedding generation fails"""
    pass

class RetrievalError(MedicalRAGException):
    """Raised when retrieval fails"""
    pass

class ValidationError(MedicalRAGException):
    """Raised when medical validation fails"""
    pass

class ConfigurationError(MedicalRAGException):
    """Raised when configuration is invalid"""
    pass

class APIError(MedicalRAGException):
    """Raised when external API calls fail"""
    pass

class CacheError(MedicalRAGException):
    """Raised when caching operations fail"""
    pass

class DatabaseError(MedicalRAGException):
    """Raised when database operations fail"""
    pass