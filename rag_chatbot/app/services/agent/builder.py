from .graph import MedicalRAGAgent
from .nodes import (
    GenerateSubQueriesNode,
    ProcessSubQueriesNode,
    SynthesizeFinalAnswerNode,
    HandleErrorNode,
)
from rag_chatbot.app.services.query_engine.transformation import SubQueryGenerator
from rag_chatbot.app.services.query_engine.search import DenseSearchEngine, Reranker
from rag_chatbot.app.services.query_engine.generation import AnswerGenerator
from rag_chatbot.app.core.llm import CustomLLM
from rag_chatbot.app.services.query_engine.context import MedicalContextAssembler

def build_medical_rag_agent() -> MedicalRAGAgent:
    llm = CustomLLM()
    
    sub_query_generator = SubQueryGenerator(llm=llm)
    searcher = DenseSearchEngine()
    reranker = Reranker()
    context_assembler = MedicalContextAssembler()
    answer_generator = AnswerGenerator(llm=llm)

    generate_sub_queries_node = GenerateSubQueriesNode(sub_query_generator)
    process_sub_queries_node = ProcessSubQueriesNode(searcher, reranker, context_assembler, answer_generator)
    synthesize_final_answer_node = SynthesizeFinalAnswerNode(answer_generator)
    handle_error_node = HandleErrorNode()

    agent = MedicalRAGAgent(
        generate_sub_queries_node=generate_sub_queries_node,
        process_sub_queries_node=process_sub_queries_node,
        synthesize_final_answer_node=synthesize_final_answer_node,
        handle_error_node=handle_error_node,
    )
    
    return agent
