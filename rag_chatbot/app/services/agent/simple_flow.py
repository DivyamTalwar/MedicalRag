import asyncio
import logging
from typing import List
from rag_chatbot.app.models.agent_models import SubQueryGeneration, SubQueryResponse
from rag_chatbot.app.services.query_engine.transformation import SubQueryGenerator
from rag_chatbot.app.services.query_engine.search import DenseSearchEngine, Reranker
from rag_chatbot.app.services.query_engine.generation import AnswerGenerator
from rag_chatbot.app.services.query_engine.context import MedicalContextAssembler
from rag_chatbot.app.core.llm import CustomLLM
from rag_chatbot.app.services.query_engine.templates import get_fallback_response

class SimpleRAGFlow:
    def __init__(self):
        self.llm = CustomLLM()
        self.sub_query_generator = SubQueryGenerator(self.llm)
        self.searcher = DenseSearchEngine()
        self.reranker = Reranker()
        self.context_assembler = MedicalContextAssembler()
        self.answer_generator = AnswerGenerator(self.llm)

    async def simple_subquery_process(self, query: str, max_retries: int = 3) -> SubQueryResponse:
        for attempt in range(max_retries):
            try:
                logging.info(f"Processing subquery (attempt {attempt + 1}/{max_retries}): {query}")
                docs = self.searcher.search(query, top_k=10)
                reranked = self.reranker.rerank(query, docs, top_k=3)
                parents, context = self.context_assembler.assemble(reranked)
                
                if "Context assembly failed" in context:
                    raise ValueError("Context assembly failed")

                response_prompt = f"Query: {query}\nContext: {context}"
                response = self.answer_generator.generate(response_prompt, "")
                
                summary_prompt = f"Summarize key facts: {context}"
                summary = self.answer_generator.generate(summary_prompt, "")
                
                return SubQueryResponse(
                    subquery_response=response,
                    summary=summary
                )
            except Exception as e:
                logging.error(f"Error in simple_subquery_process (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt + 1 == max_retries:
                    return SubQueryResponse(
                        subquery_response=get_fallback_response(query, "context_assembly"),
                        summary="No data available for this query."
                    )
                await asyncio.sleep(1)  # Wait before retrying
        return SubQueryResponse(
            subquery_response=get_fallback_response(query, "context_assembly"),
            summary="No data available for this query."
        )

    async def synthesize_final_answer(self, question: str, subquery1: str, response1: SubQueryResponse, subquery2: str, response2: SubQueryResponse, max_retries: int = 3) -> str:
        synthesis_context = (
            f"Subquery 1: {subquery1}\nResponse 1: {response1.subquery_response}\nSummary 1: {response1.summary}\n\n"
            f"Subquery 2: {subquery2}\nResponse 2: {response2.subquery_response}\nSummary 2: {response2.summary}"
        )
        
        synthesis_prompt = (
            f"You are an expert medical information synthesizer. Your task is to generate a comprehensive, detailed, and highly accurate answer to the main user query based on the provided subquery responses and summaries. "
            f"You must synthesize the information into a single, coherent response. Do not repeat information. "
            f"Ensure that all numerical values, percentages, names, dates, and key facts are accurately represented. "
            f"Your response must be complete and not truncated. If the context is insufficient to provide a complete answer, clearly state what information is missing.\n\n"
            f"Main User Query: {question}\n\n"
            f"{synthesis_context}\n\n"
            f"DETAILED AND COMPREHENSIVE FINAL ANSWER:"
        )
        
        for attempt in range(max_retries):
            try:
                response = self.answer_generator.generate(synthesis_prompt, "")
                if len(response) > 20 and not response.endswith("..."):
                    return response
            except Exception as e:
                logging.error(f"Error in synthesize_final_answer (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(1)
        
        return get_fallback_response(question, "synthesis")

    async def run(self, question: str) -> str:
        subqueries = await self.sub_query_generator.generate(question)
        
        tasks = [
            self.simple_subquery_process(subqueries.query1),
            self.simple_subquery_process(subqueries.query2)
        ]
        responses = await asyncio.gather(*tasks)
        
        final_response = await self.synthesize_final_answer(
            question=question,
            subquery1=subqueries.query1,
            response1=responses[0],
            subquery2=subqueries.query2, 
            response2=responses[1]
        )
        
        return final_response
