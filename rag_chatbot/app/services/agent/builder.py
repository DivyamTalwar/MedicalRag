from .graph import MedicalRAGAgent
from .nodes import (
    CondenseQuestionNode,
    DecomposeQueryNode,
    RetrieveAndRankNode,
    CritiqueContextNode,
    GenerateAnswerNode,
    HandleErrorNode,
)
from rag_chatbot.app.services.query_engine.transformation import QueryCondenser, HyDEGenerator, MedicalQueryExpander
from rag_chatbot.app.services.query_engine.search import DenseSearchEngine, SparseSearchEngine, ResultMerger, CrossEncoderReranker
from rag_chatbot.app.services.query_engine.generation import AnswerGenerator
from rag_chatbot.app.services.query_engine.context import ContextAssembler, ContextManager
from rag_chatbot.app.core.llm import CustomLLM
from rag_chatbot.app.core.embeddings import get_embedding_model
from rag_chatbot.app.core.extractor import MedicalEntityExtractor

def build_medical_rag_agent() -> MedicalRAGAgent:
    llm = CustomLLM()
    embeddings = get_embedding_model()
    
    # Instantiate components directly
    query_condenser = QueryCondenser(llm)
    entity_extractor = MedicalEntityExtractor()
    query_expander = MedicalQueryExpander()
    hyde_generator = HyDEGenerator(llm)
    dense_searcher = DenseSearchEngine()
    sparse_searcher = SparseSearchEngine()
    result_merger = ResultMerger()
    reranker = CrossEncoderReranker()
    context_assembler = ContextAssembler()
    answer_generator = AnswerGenerator(llm)
    context_manager = ContextManager(llm)

    # Build nodes with correct dependencies
    condense_question_node = CondenseQuestionNode(query_condenser)
    decompose_query_node = DecomposeQueryNode(entity_extractor)
    retrieve_and_rank_node = RetrieveAndRankNode(
        hyde_generator, query_expander, dense_searcher, 
        sparse_searcher, result_merger, reranker, context_assembler
    )
    critique_context_node = CritiqueContextNode(context_manager)
    generate_answer_node = GenerateAnswerNode(answer_generator)
    handle_error_node = HandleErrorNode()

    agent = MedicalRAGAgent(
        condense_question_node=condense_question_node,
        decompose_query_node=decompose_query_node,
        retrieve_and_rank_node=retrieve_and_rank_node,
        critique_context_node=critique_context_node,
        generate_answer_node=generate_answer_node,
        handle_error_node=handle_error_node,
    )
    
    return agent
