import os
from typing import List, Dict, Any
from llama_index.core.base.llms.types import ChatMessage
from app.core.llm import CustomLLM
from app.core.extractor import MedicalEntityExtractor

class QueryCondenser:
    def __init__(self, llm: CustomLLM):
        self.llm = llm
        self._template = self._create_template()

    def _create_template(self):
        return (
            "Given the following conversation and a follow-up question, rephrase the "
            "follow-up question to be a standalone question that is precise and "
            "medically relevant. Preserve all medical terminology, entities, and values.\n\n"
            "Chat History:\n"
            "{chat_history}\n\n"
            "Follow-up Input: {question}\n"
            "Standalone Medical Question:"
        )

    def condense(self, question: str, chat_history: List[ChatMessage]) -> str:
        if not chat_history:
            return question

        history_str = "\n".join(
            [f"{msg.role.capitalize()}: {msg.content}" for msg in chat_history]
        )
        
        prompt = self._template.format(chat_history=history_str, question=question)
        
        response = self.llm.complete(prompt)
        
        return response.text.strip()

class HyDEGenerator:
    def __init__(self, llm: CustomLLM):
        self.llm = llm
        self.templates = self._create_templates()

    def _create_templates(self) -> Dict[str, str]:
        return {
            "clinical_findings": (
                "Please write a sample radiology report finding and impression section for a patient "
                "based on the following query. The report should be detailed, use appropriate "
                "medical terminology, and directly address the user's question. This is a "
                "hypothetical document for search purposes.\n\n"
                "Query: {query}\n\n"
                "Hypothetical Radiology Report:"
            ),
            "performance_metrics": (
                "Please generate a hypothetical summary with a markdown table that answers the "
                "following query about performance metrics. The summary should be plausible and "
                "contain realistic data points. This is for search purposes.\n\n"
                "Query: {query}\n\n"
                "Hypothetical Performance Summary:"
            ),
            "procedure_protocol": (
                "Please write a hypothetical medical protocol or procedural document that would "
                "answer the following query. The document should be structured, clear, and use "
                "standard medical formatting. This is for search purposes.\n\n"
                "Query: {query}\n\n"
                "Hypothetical Protocol Document:"
            ),
            "default": (
                "Please write a hypothetical document that contains the answer to the following "
                "question. The document should be detailed, well-structured, and contain relevant "
                "medical or technical information. This is for search purposes.\n\n"
                "Query: {query}\n\n"
                "Hypothetical Document:"
            )
        }

    def _classify_query_type(self, query: str) -> str:
        query_lower = query.lower()
        if any(term in query_lower for term in ["finding", "impression", "report", "scan result"]):
            return "clinical_findings"
        if any(term in query_lower for term in ["performance", "metric", "tat", "turnaround", "volume"]):
            return "performance_metrics"
        if any(term in query_lower for term in ["protocol", "procedure", "how to"]):
            return "procedure_protocol"
        return "default"

    def generate(self, query: str) -> str:
        query_type = self._classify_query_type(query)
        template = self.templates[query_type]
        prompt = template.format(query=query)
        
        response = self.llm.complete(prompt)
        
        return response.text.strip()

class MedicalQueryExpander:
    def __init__(self):
        self.entity_extractor = MedicalEntityExtractor()
        self.synonym_map = {
            "ct": "Computed Tomography",
            "mri": "Magnetic Resonance Imaging",
            "us": "Ultrasound",
            "dx": "Diagnosis",
            "fx": "Fracture",
        }

    def expand(self, query: str) -> List[str]:
        entities = self.entity_extractor.extract_entities(query)
        flat_entities = self.entity_extractor.flatten_entities(entities)
        
        expanded_terms = set(flat_entities)
        
        expanded_terms.update(query.lower().split())

        for term in query.lower().split():
            term = term.strip(".,?!")
            if term in self.synonym_map:
                expanded_terms.add(self.synonym_map[term].lower())

        return list(expanded_terms)
