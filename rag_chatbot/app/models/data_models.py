from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum

class EnhancedMedicalMetadata(BaseModel):
    doc_id: UUID
    chunk_id: Optional[str] = None 
    parent_id: Optional[UUID] = None
    
    pdf_name: str
    page_no: int
    order_idx: int
    
    chunk_type: str
    section_title: Optional[str] = None
    
    medical_entities: List[str] = Field(default_factory=list)
    numerical_data: List[Dict[str, str]] = Field(default_factory=list)
    references_table: bool = False
    table_type: Optional[str] = None
    primary_topics: List[str] = Field(default_factory=list)
    searchable_terms: List[str] = Field(default_factory=list)
    
    medical_accuracy_score: float = 1.0

class DocumentChunk(BaseModel):
    text: str
    metadata: EnhancedMedicalMetadata

class Document(BaseModel):
    id: str = Field(..., description="Unique identifier for the document.")
    text: str = Field(..., description="The text content of the document.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the document.")

class ParentSummary(BaseModel):
    id: str = Field(..., description="Unique identifier for the parent summary.")
    summary: str = Field(..., description="The summary of the parent document.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the summary.")

class ParentDoc(BaseModel):
    id: str = Field(..., description="Unique identifier for the parent document.")
    text: str = Field(..., description="The text content of the parent document.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata associated with the document.")
