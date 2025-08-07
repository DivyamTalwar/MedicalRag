from .graph import MedicalRAGAgent
from .nodes import (
    CondenseQuestionNode,
    DecomposeQueryNode,
    RetrieveAndRankNode,
    GenerateAnswerNode,
    HandleErrorNode,
)
from rag_chatbot.app.services.query_engine.transformation import QueryCondenser, MedicalQueryExpander
from rag_chatbot.app.services.query_engine.search import InMeiliSearch, Reranker
from rag_chatbot.app.services.query_engine.generation import AnswerGenerator
from rag_chatbot.app.core.llm import CustomLLM
from rag_chatbot.app.core.extractor import MedicalEntityExtractor

def build_medical_rag_agent() -> MedicalRAGAgent:
    llm = CustomLLM()
    
    query_condenser = QueryCondenser(llm=llm)
    entity_extractor = MedicalEntityExtractor()
    query_expander = MedicalQueryExpander()
    searcher = InMeiliSearch()
    reranker = Reranker()
    answer_generator = AnswerGenerator(llm=llm)

    condense_question_node = CondenseQuestionNode(query_condenser)
    decompose_query_node = DecomposeQueryNode(entity_extractor)
    retrieve_and_rank_node = RetrieveAndRankNode(query_expander, searcher, reranker)
    generate_answer_node = GenerateAnswerNode(answer_generator)
    handle_error_node = HandleErrorNode()

    agent = MedicalRAGAgent(
        condense_question_node=condense_question_node,
        decompose_query_node=decompose_query_node,
        retrieve_and_rank_node=retrieve_and_rank_node,
        generate_answer_node=generate_answer_node,
        handle_error_node=handle_error_node,
    )
    
    return agent
