import time
import logging
import asyncio
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    GenerateSubQueriesNode,
    ProcessSubQueriesNode,
    SynthesizeFinalAnswerNode,
    HandleErrorNode,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalRAGAgent:
    def __init__(self, 
                generate_sub_queries_node: GenerateSubQueriesNode, 
                process_sub_queries_node: ProcessSubQueriesNode, 
                synthesize_final_answer_node: SynthesizeFinalAnswerNode,
                handle_error_node: HandleErrorNode):
        
        self.workflow = StateGraph(AgentState)
        self.nodes = {
            "generate_sub_queries": generate_sub_queries_node,
            "process_sub_queries": process_sub_queries_node,
            "synthesize_final_answer": synthesize_final_answer_node,
            "handle_error": handle_error_node
        }
        
        self._add_nodes()
        self._add_edges()
        self.graph = self.workflow.compile()

    def _add_nodes(self):
        self.workflow.add_node("generate_sub_queries", self.nodes["generate_sub_queries"].run)
        self.workflow.add_node("process_sub_queries", self.nodes["process_sub_queries"].run)
        self.workflow.add_node("synthesize_final_answer", self.nodes["synthesize_final_answer"].run)
        self.workflow.add_node("handle_error", self.nodes["handle_error"].run)

    def _add_edges(self):
        self.workflow.set_entry_point("generate_sub_queries")
        self.workflow.add_edge("generate_sub_queries", "process_sub_queries")
        self.workflow.add_edge("process_sub_queries", "synthesize_final_answer")
        self.workflow.add_edge("synthesize_final_answer", END)
        self.workflow.add_conditional_edges(
            "generate_sub_queries",
            lambda x: "handle_error" if x.get("error_state") else "process_sub_queries",
            {
                "handle_error": "handle_error",
                "process_sub_queries": "process_sub_queries"
            }
        )
        self.workflow.add_conditional_edges(
            "process_sub_queries",
            lambda x: "handle_error" if x.get("error_state") else "synthesize_final_answer",
            {
                "handle_error": "handle_error",
                "synthesize_final_answer": "synthesize_final_answer"
            }
        )
        self.workflow.add_edge("handle_error", END)

    def run(self, inputs: dict) -> dict:
        start_time = time.time()
        
        try:
            logger.info("Starting Medical RAG Agent execution")
            result = self.graph.invoke(inputs)
            end_time = time.time()
            total_execution_time = end_time - start_time
            
            result.setdefault("performance_state", {})
            result["performance_state"]["total_execution_time"] = total_execution_time
            
            logger.info(f"Medical RAG Agent execution completed in {total_execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Critical failure in Medical RAG Agent: {e}", exc_info=True)
            return {
                "error_state": {
                    "error_message": f"Critical system failure: {str(e)}",
                    "failed_node": "graph_execution"
                },
                "generation_state": {
                    "final_answer": "I apologize, but I encountered a critical system error. Please try again later."
                }
            }
