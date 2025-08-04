import time
import logging
from typing import List, Dict, Any
from llama_index.core.base.llms.types import ChatMessage
from app.core.llm import CustomLLM
from app.core.embeddings import get_embedding_model
from .transformation import QueryCondenser, HyDEGenerator, MedicalQueryExpander
from .search import DenseSearchEngine, SparseSearchEngine, ResultMerger, CrossEncoderReranker
from .context import ContextAssembler
from .generation import AnswerGenerator

class QueryEngine:
    def __init__(self):
        self.llm = CustomLLM()
        self.embeddings = get_embedding_model()
        
        self.query_condenser = QueryCondenser(self.llm)
        self.hyde_generator = HyDEGenerator(self.llm)
        self.query_expander = MedicalQueryExpander()
        
        self.dense_searcher = DenseSearchEngine()
        self.sparse_searcher = SparseSearchEngine()
        self.result_merger = ResultMerger()
        self.reranker = CrossEncoderReranker()
        
        self.context_assembler = ContextAssembler()
        
        self.answer_generator = AnswerGenerator(self.llm)

    def process_query(self, query: str, chat_history: List[ChatMessage]) -> str:
        try:
            start_time = time.time()
            logging.info(f"Processing query: {query[:100]}...")

            condensed_query = self.query_condenser.condense(query, chat_history)
            logging.info(f"Condensed query: {condensed_query}")

            hyde_start = time.time()
            hypothetical_doc = self.hyde_generator.generate(condensed_query)
            logging.info(f"HyDE generation took {time.time() - hyde_start:.2f}s")

            expanded_terms = self.query_expander.expand(condensed_query)
            
            search_start = time.time()
            dense_results = self.dense_searcher.search(hypothetical_doc)
            sparse_results = self.sparse_searcher.search(expanded_terms)
            logging.info(f"Dense/Sparse search took {time.time() - search_start:.2f}s")

            merge_start = time.time()
            merged_results = self.result_merger.merge(dense_results, sparse_results)
            reranked_chunks = self.reranker.rerank(condensed_query, merged_results)
            logging.info(f"Merge and rerank took {time.time() - merge_start:.2f}s")

            context_start = time.time()
            context_chunks = self.context_assembler.assemble(reranked_chunks)
            logging.info(f"Context assembly took {time.time() - context_start:.2f}s")
            
            gen_start = time.time()
            final_answer = self.answer_generator.generate(condensed_query, context_chunks)
            logging.info(f"Answer generation took {time.time() - gen_start:.2f}s")

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
