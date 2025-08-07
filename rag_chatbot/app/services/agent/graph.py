import time
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import (
    CondenseQuestionNode,
    DecomposeQueryNode,
    RetrieveAndRankNode,
    CritiqueContextNode,
    GenerateAnswerNode,
    HandleErrorNode,
)

class MedicalRAGAgent:
    def __init__(self, 
                condense_question_node: CondenseQuestionNode, 
                decompose_query_node: DecomposeQueryNode, 
                retrieve_and_rank_node: RetrieveAndRankNode, 
                critique_context_node: CritiqueContextNode, 
                generate_answer_node: GenerateAnswerNode,
                handle_error_node: HandleErrorNode):
        
        self.workflow = StateGraph(AgentState)
        self._add_nodes(
            condense_question_node, decompose_query_node, 
            retrieve_and_rank_node, critique_context_node, 
            generate_answer_node, handle_error_node
        )
        self._add_edges()
        self.graph = self.workflow.compile()

    def _add_nodes(self, condense_question_node, decompose_query_node, 
                retrieve_and_rank_node, critique_context_node, 
                generate_answer_node, handle_error_node):
        self.workflow.add_node("condense_question", condense_question_node.run)
        self.workflow.add_node("decompose_query", decompose_query_node.run)
        self.workflow.add_node("retrieve_and_rank", retrieve_and_rank_node.run)
        self.workflow.add_node("critique_context", critique_context_node.run)
        self.workflow.add_node("generate_answer", generate_answer_node.run)
        self.workflow.add_node("handle_error", handle_error_node.run)

    def _add_edges(self):
        self.workflow.set_entry_point("condense_question")
        self.workflow.add_edge("condense_question", "decompose_query")
        self.workflow.add_edge("decompose_query", "retrieve_and_rank")
        self.workflow.add_edge("retrieve_and_rank", "generate_answer")
        self.workflow.add_edge("generate_answer", END)
        self.workflow.add_edge("handle_error", END)

    def _route_context_decision(self, state: AgentState) -> str:
        if state["context_state"]["context_sufficiency"]:
            return "sufficient"
        else:
            return "insufficient"

    def run(self, inputs: dict):
        start_time = time.time()
        
        result = self.graph.invoke(inputs, {"recursion_limit": 15})
        
        end_time = time.time()
        
        total_duration = sum(result.get("performance_state", {}).get("node_timings", {}).values())
        result["performance_state"]["total_duration"] = total_duration
        
        return result
