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

class Citation(BaseModel):
    document_id: str = Field(..., description="The ID of the document being cited.")
    source_name: str = Field(..., description="The name of the source document.")
    page_number: Optional[int] = Field(None, description="The page number of the citation.")
    section_title: Optional[str] = Field(None, description="The section title of the citation.")
    content: str = Field(..., description="The cited content.")
    confidence_score: float = Field(1.0, description="Confidence score for the citation.")
