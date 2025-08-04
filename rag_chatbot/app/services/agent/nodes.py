import time
import re
import logging
from typing import Dict, List, Any, Optional
from .state import AgentState
from rag_chatbot.app.models.data_models import Document
from rag_chatbot.app.services.query_engine.transformation import (
    QueryCondenser, 
    MedicalQueryExpander, 
    HyDEGenerator
)
from rag_chatbot.app.services.query_engine.search import (
    DenseSearchEngine, 
    SparseSearchEngine, 
    ResultMerger, 
    CrossEncoderReranker
)
from rag_chatbot.app.services.query_engine.generation import AnswerGenerator
from rag_chatbot.app.services.query_engine.context import ContextAssembler, ContextManager
from rag_chatbot.app.core.extractor import MedicalEntityExtractor

class CondenseQuestionNode:
    def __init__(self, query_condenser: QueryCondenser):
        self.query_condenser = query_condenser

    def run(self, state: AgentState) -> Dict[str, Any]:
        start_time = time.time()
        
        query = state["query_state"]["original_query"]
        chat_history = state["query_state"]["chat_history"]
        
        condensed_query = self.query_condenser.condense(query, chat_history)
        
        processing_time = time.time() - start_time
        
        return {
            "query_state": {
                "condensed_query": condensed_query,
            },
            "performance_state": {
                "node_timings": {"condense_question": processing_time}
            }
        }

class DecomposeQueryNode:
    def __init__(self, entity_extractor: MedicalEntityExtractor):
        self.entity_extractor = entity_extractor

    def run(self, state: AgentState) -> Dict[str, Any]:
        start_time = time.time()
        
        query = state["query_state"]["condensed_query"]
        medical_entities = self.entity_extractor.extract_entities(query)
        
        processing_time = time.time() - start_time
        
        return {
            "query_state": {
                "medical_entities": medical_entities
            },
            "performance_state": {
                "node_timings": {"decompose_query": processing_time}
            }
        }

class RetrieveAndRankNode:
    def __init__(self, hyde_generator: HyDEGenerator, query_expander: MedicalQueryExpander, 
                dense_searcher: DenseSearchEngine, sparse_searcher: SparseSearchEngine, 
                result_merger: ResultMerger, reranker: CrossEncoderReranker, 
                context_assembler: ContextAssembler):
        self.hyde_generator = hyde_generator
        self.query_expander = query_expander
        self.dense_searcher = dense_searcher
        self.sparse_searcher = sparse_searcher
        self.result_merger = result_merger
        self.reranker = reranker
        self.context_assembler = context_assembler

    def run(self, state: AgentState) -> dict:
        start_time = time.time()
        query = state["query_state"]["condensed_query"]

        hypothetical_doc = self.hyde_generator.generate(query)
        expanded_terms = self.query_expander.expand(query)
        dense_results = self.dense_searcher.search(hypothetical_doc, top_k=20)
        sparse_results = self.sparse_searcher.search(expanded_terms, top_k=10)
        merged_results = self.result_merger.merge(dense_results, sparse_results)
        reranked_chunks = self.reranker.rerank(query, merged_results, top_k=8)
        parent_chunks, assembled_context = self.context_assembler.assemble(reranked_chunks)
        
        processing_time = time.time() - start_time

        return {
            "search_state": {"reranked_chunks": reranked_chunks},
            "context_state": {
                "parent_chunks": parent_chunks,
                "assembled_context": assembled_context
            },
            "performance_state": {"node_timings": {"retrieve_and_rank": processing_time}}
        }

class CritiqueContextNode:
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager

    def run(self, state: AgentState) -> Dict[str, Any]:
        start_time = time.time()
        
        query = state["query_state"]["condensed_query"]
        context = state["context_state"]["parent_chunks"]
        
        is_sufficient = self.context_manager.evaluate_sufficiency(query, context)
        
        processing_time = time.time() - start_time
        
        return {
            "context_state": {
                "context_sufficiency": is_sufficient,
            },
            "performance_state": {
                "node_timings": {"critique_context": processing_time}
            }
        }

class GenerateAnswerNode:
    def __init__(self, answer_generator: AnswerGenerator):
        self.answer_generator = answer_generator

    def run(self, state: AgentState) -> dict:
        start_time = time.time()
        query = state["query_state"]["condensed_query"]
        context = state["context_state"]["parent_chunks"]
        is_streaming = state["generation_state"].get("is_streaming", False)

        answer = self.answer_generator.generate(query, context)
        rich_citations = self._create_rich_citations(answer, context)
        
        response_payload = {
            "final_answer": answer,
            "rich_citations": rich_citations,
        }

        if is_streaming:
            response_payload["streaming_response"] = self._create_streaming_generator(answer, rich_citations)

        processing_time = time.time() - start_time
        return {
            "generation_state": response_payload,
            "performance_state": {"node_timings": {"generate_answer": processing_time}}
        }

    def _create_rich_citations(self, answer: str, context: List[Document]) -> List[Dict]:
        citations = []
        citation_pattern = r'\[(\d+)\]'
        matches = re.findall(citation_pattern, answer)

        for match in matches:
            idx = int(match) - 1
            if 0 <= idx < len(context):
                chunk = context[idx]
                citations.append({
                    "document_id": chunk.id,
                    "source_name": chunk.metadata.get("pdf_name", "Unknown"),
                    "page_number": chunk.metadata.get("page_no"),
                    "section_title": chunk.metadata.get("section_title", "N/A"),
                    "medical_entities": chunk.metadata.get("medical_entities", []),
                    "table_type": chunk.metadata.get("table_type"),
                    "primary_topics": chunk.metadata.get("primary_topics", []),
                    "content_preview": chunk.text[:200] + "...",
                    "page_order": chunk.metadata.get("order_idx", 0)
                })
        return citations

    def _create_streaming_generator(self, answer: str, citations: List[Dict]):
        for i in range(0, len(answer), 32):
            yield answer[i:i+32]
            time.sleep(0)

        sources_section = "\n\n## RELEVANT SOURCES:\n\n"
        for i, citation in enumerate(citations, 1):
            sources_section += f"**[{i}] {citation['source_name']}** (Page {citation['page_number']})\n"
            sources_section += f"Section: {citation['section_title']}\n"
            sources_section += f"Medical Entities: {', '.join(citation['medical_entities'][:5])}\n"
            if citation['table_type']:
                sources_section += f"Table Type: {citation['table_type']}\n"
            sources_section += f"Content: {citation['content_preview']}\n\n"
        
        yield sources_section
class HandleErrorNode:
    def run(self, state: AgentState) -> Dict[str, Any]:
        error_message = state["error_state"].get("error_message", "An unknown error occurred.")
        failed_node = state["error_state"].get("failed_node", "Unknown")
        
        logging.exception(f"Exception caught in node '{failed_node}': {error_message}")
        
        user_response = "I apologize, but I encountered an issue while processing your request. Please try again or rephrase your question."
        
        return {
            "generation_state": {
                "final_answer": user_response,
                "rich_citations": [],
                "is_streaming": False,
            },
            "error_state": {
                "error_message": error_message,
                "failed_node": failed_node
            }
        }
