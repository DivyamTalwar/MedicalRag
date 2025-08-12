import asyncio
import time
import logging
from typing import List, Dict, Any
from llama_index.core.base.llms.types import ChatMessage
from rag_chatbot.app.core.llm import CustomLLM
from rag_chatbot.app.core.embeddings import get_embedding_model
from .transformation import SubQueryGenerator
from .search import DenseSearchEngine, MedicalReranker
from .context import ContextAssembler
from .generation import AnswerGenerator

class QueryEngine:
    def __init__(self):
        self.llm = CustomLLM()
        self.embeddings = get_embedding_model()
        
        self.sub_query_generator = SubQueryGenerator(self.llm)
        self.dense_searcher = DenseSearchEngine()
        self.reranker = MedicalReranker()
        self.context_assembler = ContextAssembler()
        self.answer_generator = AnswerGenerator(self.llm)

    async def process_sub_query(self, sub_query: str) -> List[Dict[str, Any]]:
        logging.info(f"Processing sub-query: {sub_query}")
        
        dense_results = self.dense_searcher.search(sub_query, top_k=10)
        
        reranked_results = self.reranker.rerank(sub_query, dense_results, top_k=5)
        
        return [doc.metadata for doc in reranked_results]

    async def process_query(self, query: str, chat_history: List[ChatMessage]) -> str:
        try:
            start_time = time.time()
            logging.info(f"Processing query: {query[:100]}...")

            sub_queries_generation = await self.sub_query_generator.generate(query, chat_history)
            sub_queries = sub_queries_generation.queries
            logging.info(f"Generated {len(sub_queries)} sub-queries.")

            tasks = [self.process_sub_query(sq) for sq in sub_queries]
            sub_query_results = await asyncio.gather(*tasks)

            final_context = []
            for i, result in enumerate(sub_query_results):
                final_context.append({
                    "sub_query": sub_queries[i],
                    "results": result
                })

            context_str = self.context_assembler.assemble(final_context)
            
            final_answer = self.answer_generator.generate(query, context_str)

            total_time = time.time() - start_time
            logging.info(f"Query processed in {total_time:.2f}s")
            
            return final_answer
        except Exception as e:
            logging.error(f"Query processing failed: {e}", exc_info=True)
            return "I apologize, but I encountered an error processing your request. Please try again."

    def _check_pinecone_health(self) -> bool:
        try:
            self.dense_searcher.index.describe_index_stats()
            return True
        except Exception as e:
            logging.error(f"Pinecone health check failed: {e}")
            return False

    def _check_mongodb_health(self) -> bool:
        try:
            self.sparse_searcher.mongo_client.admin.command('ping')
            return True
        except Exception as e:
            logging.error(f"MongoDB health check failed: {e}")
            return False

    def _check_llm_health(self) -> bool:
        try:
            test_response = self.llm.complete("Health check test")
            return bool(test_response.text.strip())
        except Exception as e:
            logging.error(f"LLM health check failed: {e}")
            return False

    def _check_reranker_health(self) -> bool:
        try:
            self.reranker.model.predict([["test", "test"]])
            return True
        except Exception as e:
            logging.error(f"Reranker health check failed: {e}")
            return False

    def health_check(self) -> Dict[str, bool]:
        return {
            "pinecone": self._check_pinecone_health(),
            "mongodb": self._check_mongodb_health(),
            "llm": self._check_llm_health(),
            "reranker": self._check_reranker_health()
        }

if __name__ == '__main__':
    engine = QueryEngine()
    
    history = [
        ChatMessage(role="user", content="What is the turnaround time for CT scans?"),
        ChatMessage(role="assistant", content="The average turnaround time for CT scans is 45 minutes.")
    ]
    
    new_question = "What about for MRIs?"
    
    answer = engine.process_query(new_question, history)
    
    print("Final Answer")
    print(answer)
