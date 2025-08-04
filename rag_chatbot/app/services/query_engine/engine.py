import time
import logging
import hashlib
from functools import lru_cache
from typing import List, Dict, Any
from llama_index.core.base.llms.types import ChatMessage
from app.core.llm import CustomLLM
from app.core.embeddings import get_embedding_model
from .transformation import QueryCondenser, HyDEGenerator, MedicalQueryExpander
from .search import DenseSearchEngine, SparseSearchEngine, ResultMerger, CrossEncoderReranker
from .context import ContextAssembler
from .generation import AnswerGenerator

class QueryEngine:
    """
    The main query engine that orchestrates the entire RAG pipeline.
    """
    def __init__(self):
        # Initialize all components
        self.llm = CustomLLM()
        self.embeddings = get_embedding_model()
        
        # Step 7: Transformation
        self.query_condenser = QueryCondenser(self.llm)
        self.hyde_generator = HyDEGenerator(self.llm)
        self.query_expander = MedicalQueryExpander()
        
        # Step 8: Search & Re-ranking
        self.dense_searcher = DenseSearchEngine(self.embeddings)
        self.sparse_searcher = SparseSearchEngine()
        self.result_merger = ResultMerger()
        self.reranker = CrossEncoderReranker()
        
        # Step 9: Context Assembly
        self.context_assembler = ContextAssembler()
        
        # Step 10: Generation
        self.answer_generator = AnswerGenerator(self.llm)

    @lru_cache(maxsize=1000)
    def _cached_search(self, query_hash: str, hypothetical_doc: str, expanded_terms_tuple: tuple):
        expanded_terms = list(expanded_terms_tuple)
        dense_results = self.dense_searcher.search(hypothetical_doc)
        sparse_results = self.sparse_searcher.search(expanded_terms)
        return dense_results, sparse_results

    def process_query(self, query: str, chat_history: List[ChatMessage]) -> str:
        """
        Processes a user query through the full RAG pipeline.

        Args:
            query: The user's current question.
            chat_history: The history of the conversation.

        Returns:
            The final, generated answer.
        """
        try:
            start_time = time.time()
            logging.info(f"Processing query: {query[:100]}...")

            # 1. Condense query
            condensed_query = self.query_condenser.condense(query, chat_history)
            logging.info(f"Condensed query: {condensed_query}")

            # 2. Generate hypothetical document
            hyde_start = time.time()
            hypothetical_doc = self.hyde_generator.generate(condensed_query)
            logging.info(f"HyDE generation took {time.time() - hyde_start:.2f}s")

            # 3. Expand query for sparse search
            expanded_terms = self.query_expander.expand(condensed_query)
            
            # 4. Perform searches
            search_start = time.time()
            query_hash = hashlib.sha256(condensed_query.encode()).hexdigest()[:16]
            dense_results, sparse_results = self._cached_search(query_hash, hypothetical_doc, tuple(expanded_terms))
            logging.info(f"Dense/Sparse search took {time.time() - search_start:.2f}s")

            # 5. Merge and re-rank
            merge_start = time.time()
            merged_results = self.result_merger.merge(dense_results, sparse_results)
            reranked_chunks = self.reranker.rerank(condensed_query, merged_results)
            logging.info(f"Merge and rerank took {time.time() - merge_start:.2f}s")

            # 6. Assemble context
            context_start = time.time()
            context_chunks = self.context_assembler.assemble(reranked_chunks)
            logging.info(f"Context assembly took {time.time() - context_start:.2f}s")
            
            # 7. Generate final answer
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
        """Returns the health status of all components."""
        return {
            "pinecone": self._check_pinecone_health(),
            "mongodb": self._check_mongodb_health(),
            "llm": self._check_llm_health(),
            "reranker": self._check_reranker_health()
        }

# Example usage (for testing)
if __name__ == '__main__':
    engine = QueryEngine()
    
    # Mock chat history
    history = [
        ChatMessage(role="user", content="What is the turnaround time for CT scans?"),
        ChatMessage(role="assistant", content="The average turnaround time for CT scans is 45 minutes.")
    ]
    
    # New query
    new_question = "What about for MRIs?"
    
    # Process
    answer = engine.process_query(new_question, history)
    
    print("--- Final Answer ---")
    print(answer)
