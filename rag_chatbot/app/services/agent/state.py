from typing import TypedDict, List, Optional, Dict, Any, Union
from langchain_core.messages import BaseMessage
from rag_chatbot.app.models.data_models import Document, Citation

# State for query processing
class QueryState(TypedDict):
    original_query: str
    condensed_query: str
    chat_history: List[BaseMessage]
    medical_entities: Dict[str, Any]

# State for search and retrieval
class SearchState(TypedDict):
    dense_results: List[Document]
    sparse_results: List[Document]
    merged_candidates: List[Document]
    reranked_chunks: List[Document]

# State for context management
class ContextState(TypedDict):
    parent_chunks: List[Document]
    assembled_context: str 
    context_sufficiency: bool
    medical_metadata: Dict[str, Any]

# State for answer generation
class GenerationState(TypedDict):
    final_answer: str
    rich_citations: List[Citation]
    is_streaming: bool

class PerformanceState(TypedDict):
    node_timings: Dict[str, float]
    total_duration: float

class ErrorState(TypedDict):
    error_message: Optional[str]
    failed_node: Optional[str]

# The overall state of the agent
class AgentState(TypedDict):
    query_state: QueryState
    search_state: SearchState
    context_state: ContextState
    generation_state: GenerationState
    performance_state: PerformanceState
    error_state: ErrorState
    conversation_history: List[Dict[str, Any]]
