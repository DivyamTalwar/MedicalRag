from typing import TypedDict, List, Optional, Dict, Any, Union
from langchain_core.messages import BaseMessage
from rag_chatbot.app.models.data_models import Document, Citation

class QueryState(TypedDict):
    original_query: str
    condensed_query: str
    chat_history: List[BaseMessage]
    medical_entities: Dict[str, Any]

class SearchState(TypedDict):
    dense_results: List[Document]
    sparse_results: List[Document]
    merged_candidates: List[Document]
    reranked_chunks: List[Document]

class ContextState(TypedDict):
    parent_chunks: List[Document]
    assembled_context: str 
    context_sufficiency: bool
    medical_metadata: Dict[str, Any]

class GenerationState(TypedDict):
    final_answer: str
    is_streaming: bool
    sub_query_responses: Optional[List[Dict[str, Any]]]

class PerformanceState(TypedDict):
    node_timings: Dict[str, float]
    total_duration: float

class ErrorState(TypedDict):
    error_message: Optional[str]
    failed_node: Optional[str]

class AgentState(TypedDict):
    query_state: QueryState
    search_state: SearchState
    context_state: ContextState
    generation_state: GenerationState
    performance_state: PerformanceState
    error_state: ErrorState
    conversation_history: List[Dict[str, Any]]
    sub_queries: Optional[List[str]]
