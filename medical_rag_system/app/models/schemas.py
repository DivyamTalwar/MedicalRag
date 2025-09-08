#!/usr/bin/env python3
"""
PYDANTIC SCHEMAS
==============
Data models for Medical RAG System API
"""

from pydantic import BaseModel, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

# Enums
class MedicalFormType(str, Enum):
    LAB_REPORT = "lab_report"
    RADIOLOGY = "radiology"
    PATHOLOGY = "pathology"
    DISCHARGE_SUMMARY = "discharge_summary"
    CLINICAL_NOTE = "clinical_note"
    PRESCRIPTION = "prescription"
    INSURANCE_FORM = "insurance_form"
    UNKNOWN = "unknown"

class ExtractionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ValidationSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# Request Models
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    use_cache: Optional[bool] = True
    use_reranking: Optional[bool] = True
    medical_context: Optional[Dict[str, Any]] = None
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    @validator('top_k')
    def top_k_positive(cls, v):
        if v <= 0:
            raise ValueError('top_k must be positive')
        return v

class PDFUploadRequest(BaseModel):
    filename: str
    extract_tables: Optional[bool] = True
    extract_images: Optional[bool] = True
    use_ocr: Optional[bool] = True
    medical_type: Optional[MedicalFormType] = MedicalFormType.UNKNOWN

class BatchProcessRequest(BaseModel):
    pdf_paths: List[str]
    extraction_methods: Optional[List[str]] = None
    parallel_processing: Optional[bool] = True

# Response Models
class ExtractionResultResponse(BaseModel):
    text: str
    confidence: float
    extraction_methods: List[str]
    processing_time: float
    form_type: MedicalFormType
    sections_detected: int
    tables_extracted: int
    images_found: int
    
class MedicalSectionResponse(BaseModel):
    type: str
    title: str
    content: str
    confidence: float
    page_num: int

class ExtractedTableResponse(BaseModel):
    headers: List[str]
    rows: List[List[str]]
    confidence: float
    page_num: int
    table_type: str

class ValidationIssueResponse(BaseModel):
    severity: ValidationSeverity
    category: str
    message: str
    location: str
    suggestion: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    response: str
    confidence: float
    sources: List[Dict[str, Any]]
    processing_time: float
    systems_used: List[str]
    medical_validation: Optional[Dict[str, Any]] = None
    cached: bool = False

class RetrievalResultResponse(BaseModel):
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    medical_type: Optional[MedicalFormType] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    systems_online: int
    systems_detail: Dict[str, str]
    database_status: str
    cache_status: str

# Medical-specific Models
class LabValueResponse(BaseModel):
    test_name: str
    value: float
    unit: str
    normal_range: str
    status: str  # normal, low, high, critical
    confidence: float

class MedicationResponse(BaseModel):
    name: str
    dose: float
    unit: str
    frequency: str
    route: Optional[str] = None
    category: Optional[str] = None
    safety_score: float

class MedicalEntityResponse(BaseModel):
    entity: str
    type: str  # symptom, disease, medication, procedure
    confidence: float
    context: str

class KnowledgeGraphResponse(BaseModel):
    entities: List[MedicalEntityResponse]
    relationships: List[Dict[str, Any]]
    confidence: float

# Batch Processing Models
class BatchJobStatus(BaseModel):
    job_id: str
    status: ExtractionStatus
    total_files: int
    completed_files: int
    failed_files: int
    progress_percentage: float
    estimated_completion: Optional[datetime] = None
    results: Optional[List[ExtractionResultResponse]] = None

class BatchJobRequest(BaseModel):
    files: List[str]
    extraction_config: Optional[Dict[str, Any]] = None
    callback_url: Optional[str] = None

# System Configuration Models
class SystemStatsResponse(BaseModel):
    documents_processed: int
    average_accuracy: float
    total_queries: int
    cache_hit_rate: float
    active_learning_improvements: int
    system_uptime: str

class CacheStatsResponse(BaseModel):
    cache_size: int
    hit_rate: float
    miss_rate: float
    evictions: int
    memory_usage: str

class ActiveLearningStatsResponse(BaseModel):
    total_interactions: int
    feedback_received: int
    accuracy_improvement: float
    model_updates: int
    last_update: Optional[datetime] = None

# Error Response Models
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = datetime.now()
    request_id: Optional[str] = None

# Feedback Models
class FeedbackRequest(BaseModel):
    query: str
    response: str
    rating: int  # 1-5
    comments: Optional[str] = None
    correction: Optional[str] = None
    
    @validator('rating')
    def rating_valid(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Rating must be between 1 and 5')
        return v

class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str
    improvement_applied: bool
    message: str

# Analytics Models
class AnalyticsRequest(BaseModel):
    start_date: datetime
    end_date: datetime
    metrics: Optional[List[str]] = None
    group_by: Optional[str] = None

class AnalyticsResponse(BaseModel):
    period: str
    metrics: Dict[str, Any]
    trends: Dict[str, List[float]]
    insights: List[str]

# Configuration Models
class SystemConfigRequest(BaseModel):
    extraction_methods: Optional[List[str]] = None
    confidence_threshold: Optional[float] = None
    reranking_weights: Optional[Dict[str, float]] = None
    cache_settings: Optional[Dict[str, Any]] = None

class SystemConfigResponse(BaseModel):
    current_config: Dict[str, Any]
    pending_changes: Optional[Dict[str, Any]] = None
    last_updated: datetime
    version: str