import time
import re
import logging
from typing import Dict, List, Any, Optional
from .state import AgentState
from rag_chatbot.app.models.data_models import Document
from rag_chatbot.app.services.query_engine.transformation import (
    QueryCondenser, 
    MedicalQueryExpander
)
from rag_chatbot.app.services.query_engine.search import (
    InMeiliSearch,
    Reranker
)
from rag_chatbot.app.services.query_engine.generation import AnswerGenerator
from rag_chatbot.app.core.extractor import MedicalEntityExtractor

class CondenseQuestionNode:
    def __init__(self, query_condenser: QueryCondenser):
        self.query_condenser = query_condenser

    def run(self, state: AgentState) -> Dict[str, Any]:
        start_time = time.time()
        
        query = state["query_state"]["original_query"]
        chat_history = state["query_state"]["chat_history"]
        
        if not chat_history:
            condensed_query = query
        else:
            condensed_query = self.query_condenser.condense(query, chat_history)
        
        processing_time = time.time() - start_time
        
        updated_query_state = state["query_state"].copy()
        updated_query_state["condensed_query"] = condensed_query
        
        return {
            "query_state": updated_query_state,
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
        
        updated_query_state = state["query_state"].copy()
        updated_query_state["medical_entities"] = medical_entities
        
        return {
            "query_state": updated_query_state,
            "performance_state": {
                "node_timings": {"decompose_query": processing_time}
            }
        }

class RetrieveAndRankNode:
    def __init__(self, query_expander: MedicalQueryExpander, searcher: InMeiliSearch, reranker: Reranker):
        self.query_expander = query_expander
        self.searcher = searcher
        self.reranker = reranker

    def run(self, state: AgentState) -> dict:
        start_time = time.time()
        query = state["query_state"]["condensed_query"]

        expanded_terms = self.query_expander.expand(query)
        
        # 1. Initial Keyword Search
        initial_results = self.searcher.search(expanded_terms, top_k=30)
        
        # 2. Rerank the results for relevance
        reranked_results = self.reranker.rerank(query, initial_results, top_k=8)
        
        # 3. Assemble the final context
        assembled_context = "\n\n".join([doc.text for doc in reranked_results])
        
        processing_time = time.time() - start_time

        return {
            "context_state": {
                "parent_chunks": reranked_results,
                "assembled_context": assembled_context,
            },
            "performance_state": {"node_timings": {"retrieve_and_rank": processing_time}}
        }

class GenerateAnswerNode:
    def __init__(self, answer_generator: AnswerGenerator):
        self.answer_generator = answer_generator

    def run(self, state: AgentState) -> dict:
        start_time = time.time()
        query = state["query_state"]["condensed_query"]
        context = state["context_state"]["parent_chunks"]
        is_streaming = state["generation_state"].get("is_streaming", False)

        # Safety check for empty context
        if not context:
            return {
                "generation_state": {
                    "final_answer": "I could not find any relevant information in the documents to answer your question.",
                    "rich_citations": [],
                }
            }

        raw_answer, final_answer = self.answer_generator.generate(query, context)
        rich_citations = self._create_rich_citations(raw_answer, context)
        
        response_payload = {
            "final_answer": final_answer,
            "rich_citations": rich_citations,
        }

        if is_streaming:
            response_payload["streaming_response"] = self._create_streaming_generator(final_answer, rich_citations)

        processing_time = time.time() - start_time
        updated_generation_state = state["generation_state"].copy()
        updated_generation_state.update(response_payload)

        return {
            "generation_state": updated_generation_state,
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
