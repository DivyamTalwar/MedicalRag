from pydantic import BaseModel, Field
from uuid import UUID, uuid4

class ChunkMetadata(BaseModel):
    doc_id: UUID
    parent_id: UUID | None = None
    pdf_name: str
    page_no: int
    order_idx: int # Order of chunk on the page
    chunk_type: str # "paragraph", "table", "list"
    section_title: str | None = None

class DocumentChunk(BaseModel):
    chunk_id: UUID = Field(default_factory=uuid4)
    text: str
    metadata: ChunkMetadata
