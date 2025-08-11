import time
import re
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional
from .state import AgentState
from rag_chatbot.app.models.data_models import Document
from rag_chatbot.app.services.query_engine.transformation import (
    QueryCondenser, 
    SubQueryGenerator,
    ContradictionDetector
)
from rag_chatbot.app.services.query_engine.search import (
    DenseSearchEngine,
    Reranker
)
from rag_chatbot.app.services.query_engine.generation import AnswerGenerator
from rag_chatbot.app.core.extractor import MedicalEntityExtractor
from rag_chatbot.app.services.query_engine.context import MedicalContextAssembler
from concurrent.futures import ThreadPoolExecutor
from rag_chatbot.app.models.agent_models import SubQueryGeneration, SubQueryResponse

class GenerateSubQueriesNode:
    def __init__(self, sub_query_generator: SubQueryGenerator):
        self.sub_query_generator = sub_query_generator

    def run(self, state: AgentState) -> Dict[str, Any]:
        try:
            start_time = time.time()
            question = state["query_state"]["original_query"]
            subqueries = asyncio.run(self.sub_query_generator.generate(question))
            
            processing_time = time.time() - start_time
            
            return {
                "sub_queries": [subqueries.query1, subqueries.query2],
                "performance_state": {"node_timings": {"generate_sub_queries": processing_time}}
            }
        except Exception as e:
            logging.error(f"Error in GenerateSubQueriesNode: {e}", exc_info=True)
            return {
                "error_state": {
                    "error_message": str(e),
                    "failed_node": "generate_sub_queries"
                }
            }

class ProcessSubQueriesNode:
    def __init__(self, searcher: DenseSearchEngine, reranker: Reranker, context_assembler: MedicalContextAssembler, answer_generator: AnswerGenerator):
        self.searcher = searcher
        self.reranker = reranker
        self.context_assembler = context_assembler
        self.answer_generator = answer_generator

    def process_single_subquery(self, subquery: str) -> SubQueryResponse:
        child_chunks = self.searcher.search(subquery, top_k=10)
        
        reranked_chunks = self.reranker.rerank(subquery, child_chunks, top_k=3)
        
        parent_chunks, assembled_context = self.context_assembler.assemble(reranked_chunks)
        
        response_prompt = (
            f"Based on the following context, please provide a direct and comprehensive answer to the subquery.\n\n"
            f"Context:\n{assembled_context}\n\n"
            f"Subquery: {subquery}\n\n"
            f"Answer:"
        )
        subquery_response = self.answer_generator.generate(response_prompt, "")
        
        summary_prompt = (
            f"Based on the following context, please provide a summary that captures ALL numerical values, percentages, names, dates, and key facts exactly as they appear.\n\n"
            f"Context:\n{assembled_context}\n\n"
            f"Summary:"
        )
        summary = self.answer_generator.generate(summary_prompt, "")
        
        return SubQueryResponse(subquery_response=subquery_response, summary=summary)

    def run(self, state: AgentState) -> Dict[str, Any]:
        try:
            start_time = time.time()
            sub_queries = state["sub_queries"]
            
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(self.process_single_subquery, sub_queries))

            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"Error processing subquery {i}: {result}", exc_info=True)
                    processed_results.append(SubQueryResponse(
                        subquery_response="Unable to process this subquery due to technical issues.",
                        summary="No data available for this query."
                    ))
                else:
                    processed_results.append(result)

            processing_time = time.time() - start_time
            
            return {
                "generation_state": {
                    "sub_query_responses": [res.model_dump() for res in processed_results]
                },
                "performance_state": {"node_timings": {"process_sub_queries": processing_time}}
            }
        except Exception as e:
            logging.error(f"Error in ProcessSubQueriesNode: {e}", exc_info=True)
            return {
                "error_state": {
                    "error_message": str(e),
                    "failed_node": "process_sub_queries"
                }
            }

class SynthesizeFinalAnswerNode:
    def __init__(self, answer_generator: AnswerGenerator):
        self.answer_generator = answer_generator

    def run(self, state: AgentState) -> dict:
        try:
            start_time = time.time()
            original_query = state["query_state"]["original_query"]
            sub_query_responses = state["generation_state"]["sub_query_responses"]
            
            synthesis_context = ""
            for i, res in enumerate(sub_query_responses):
                synthesis_context += f"Subquery {i+1} Response: {res['subquery_response']}\n"
                synthesis_context += f"Subquery {i+1} Summary: {res['summary']}\n\n"

            synthesis_prompt = (
                f"Based on the following subquery responses and summaries, please synthesize a final, coherent, and complete answer to the main user query.\n\n"
                f"Main User Query: {original_query}\n\n"
                f"{synthesis_context}"
                f"FINAL SYNTHESIZED ANSWER:"
            )

            final_answer = self.answer_generator.generate(synthesis_prompt, "")
            
            processing_time = time.time() - start_time
            
            return {
                "generation_state": {
                    "final_answer": final_answer,
                    "sub_query_responses": sub_query_responses
                },
                "performance_state": {"node_timings": {"synthesize_final_answer": processing_time}}
            }
        except Exception as e:
            logging.error(f"Error in SynthesizeFinalAnswerNode: {e}", exc_info=True)
            return {
                "error_state": {
                    "error_message": str(e),
                    "failed_node": "synthesize_final_answer"
                }
            }

class HandleErrorNode:
    def run(self, state: AgentState) -> Dict[str, Any]:
        error_message = state["error_state"].get("error_message", "An unknown error occurred.")
        failed_node = state["error_state"].get("failed_node", "Unknown")
        
        logging.exception(f"Exception caught in node '{failed_node}': {error_message}")
        
        user_response = "I apologize, but I encountered an issue while processing your request. Please try again or rephrase your question."
        
        return {
            "generation_state": {
                "final_answer": user_response,
            }
        }
