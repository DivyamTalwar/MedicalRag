from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4


class EnhancedMedicalMetadata(BaseModel):
    doc_id: UUID
    chunk_id: Optional[str] = None 
    parent_id: Optional[UUID] = None
    
    # Document context
    pdf_name: str
    page_no: int
    order_idx: int
    
    chunk_type: str
    section_title: Optional[str] = None
    
    medical_entities: List[str] = Field(default_factory=list)
    numerical_data: List[Dict[str, str]] = Field(default_factory=list)
    contains_phi: bool = False
    references_table: bool = False
    table_type: Optional[str] = None
    primary_topics: List[str] = Field(default_factory=list)
    searchable_terms: List[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    text: str
    metadata: EnhancedMedicalMetadata
