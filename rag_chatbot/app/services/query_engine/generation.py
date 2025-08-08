import re
import time
import logging
from typing import List, Dict, Any
from rag_chatbot.app.core.llm import CustomLLM
from rag_chatbot.app.models.data_models import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self, llm: CustomLLM):
        self.llm = llm
        self._prompt_template = self._create_prompt_template()

    def _create_prompt_template(self) -> str:
        return (
            "You are a highly knowledgeable medical AI assistant. Your task is to provide a "
            "comprehensive and evidence-based answer to the user's question based *only* on the "
            "provided context. You must adhere to the following rules:\n"
            "1. **Synthesize, Don't Summarize**: If the user asks for a summary, you must provide a detailed, multi-point synthesis of all relevant information from the context.\n"
            "2. **Be Concise:** Synthesize the information from the context into a clear and "
            "easy-to-understand answer. Do not copy the context verbatim.\n"
            "3. **No Outside Knowledge:** Do not use any information not present in the provided context.\n"
            "4. **Disclaimer:** Conclude your response with the mandatory disclaimer: "
            "'This information is for informational purposes only and does not constitute medical advice.'\n\n"
            "CONTEXT\n"
            "{assembled_context}\n\n"
            "QUESTION\n"
            "{question}\n\n"
            "ANSWER\n"
        )

    def _get_query_type(self, query: str) -> str:
        if re.search(r'\b(lab|value|result|test)\b', query, re.IGNORECASE):
            return "lab_result"
        elif re.search(r'\b(procedure|scan|exam|radiology)\b', query, re.IGNORECASE):
            return "procedure"
        return "general"

    def _validate_answer_quality(self, answer: str, context: str, query_type: str) -> bool:
        # 1. Adaptive Answer Length Validation
        min_length = 20 if query_type == "lab_result" else 50
        if len(answer.split()) < min_length:
            logger.warning(f"Validation failed: Answer too short for query type '{query_type}'.")
            return False

        # 2. Expanded Medical Terminology Preservation
        term_pattern = r'\b([A-Z][a-z]*\d*|[A-Z]{2,}|mg/dL|mmHg|mEq/L|Hemoglobin|HbA1c|TSH|HDL|pH|PACS|RIS|P2 Peak|P3 Peak)\b'
        context_terms = set(re.findall(term_pattern, context))
        answer_terms = set(re.findall(term_pattern, answer))
        if context_terms:
            preserved_terms = context_terms.intersection(answer_terms)
            preservation_ratio = len(preserved_terms) / len(context_terms)
            if preservation_ratio < 0.7:
                logger.warning(f"Validation failed: Low medical terminology preservation ({preservation_ratio:.2f}).")
                return False

        # 3. Expanded Medical Value and Reference Range Preservation
        value_pattern = r'([<>]\s*\d+\.\d+|\d+\.\d+\s*-\s*\d+\.\d+|\d+\s*-\s*\d+|\d+\.\d+\s*(?:mg/dL|mmHg|mEq/L|%)?|\d+\.\d+)'
        context_values = set(re.findall(value_pattern, context))
        answer_values = set(re.findall(value_pattern, answer))
        if context_values:
            preserved_values = context_values.intersection(answer_values)
            preservation_ratio = len(preserved_values) / len(context_values)
            if preservation_ratio < 0.8:
                logger.warning(f"Validation failed: Low medical value preservation ({preservation_ratio:.2f}).")
                return False

        return True

    def generate(self, query: str, assembled_context: str, max_retries=3, initial_backoff=1) -> str:
        base_prompt = self._prompt_template.format(
            assembled_context=assembled_context,
            question=query
        )
        query_type = self._get_query_type(query)

        retries = 0
        backoff_time = initial_backoff
        while retries < max_retries:
            prompt = base_prompt
            if retries > 0:
                prompt += "\n\nLet's try again. Please ensure the answer is comprehensive, accurate, and directly uses the provided context."

            try:
                response = self.llm.invoke(prompt).content
                response = str(response).encode('utf-8', 'ignore').decode('utf-8')

                if self._validate_answer_quality(response, assembled_context, query_type):
                    return response
                else:
                    logger.warning(f"Answer quality validation failed. Retrying ({retries + 1}/{max_retries})...")
                    retries += 1

            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                retries += 1
                if retries >= max_retries:
                    logger.critical("LLM failed after multiple retries. Returning a fallback response.")
                    return "I am sorry, but I was unable to generate a response. Please try again later."
                
                logger.info(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
                backoff_time *= 2  # Exponential backoff

        logger.error("Failed to generate a valid answer after multiple retries.")
        return "I am sorry, but I could not generate a satisfactory answer. Please rephrase your question or try again."
