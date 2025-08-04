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
        llm = CustomLLM()
        embeddings = get_embedding_model()
        
        # Step 7: Transformation
        self.query_condenser = QueryCondenser(llm)
        self.hyde_generator = HyDEGenerator(llm)
        self.query_expander = MedicalQueryExpander()
        
        # Step 8: Search & Re-ranking
        self.dense_searcher = DenseSearchEngine(embeddings)
        self.sparse_searcher = SparseSearchEngine()
        self.result_merger = ResultMerger()
        self.reranker = CrossEncoderReranker()
        
        # Step 9: Context Assembly
        self.context_assembler = ContextAssembler()
        
        # Step 10: Generation
        self.answer_generator = AnswerGenerator(llm)

    def process_query(self, query: str, chat_history: List[ChatMessage]) -> str:
        """
        Processes a user query through the full RAG pipeline.

        Args:
            query: The user's current question.
            chat_history: The history of the conversation.

        Returns:
            The final, generated answer.
        """
        # 1. Condense query
        condensed_query = self.query_condenser.condense(query, chat_history)
        
        # 2. Generate hypothetical document
        hypothetical_doc = self.hyde_generator.generate(condensed_query)
        
        # 3. Expand query for sparse search
        expanded_terms = self.query_expander.expand(condensed_query)
        
        # 4. Perform searches
        dense_results = self.dense_searcher.search(hypothetical_doc)
        sparse_results = self.sparse_searcher.search(expanded_terms)
        
        # 5. Merge and re-rank
        merged_results = self.result_merger.merge(dense_results, sparse_results)
        reranked_chunks = self.reranker.rerank(condensed_query, merged_results)
        
        # 6. Assemble context
        final_context = self.context_assembler.assemble(reranked_chunks)
        
        # 7. Generate final answer
        final_answer = self.answer_generator.generate(condensed_query, self.context_assembler._fetch_parent_chunks(self.context_assembler._extract_parent_ids(reranked_chunks)))

        return final_answer

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
