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
            "2. **Be Detailed and Comprehensive:** Your answer must be complete and not truncated. Provide as much detail as possible based on the context.\n"
            "3. **No Outside Knowledge:** Do not use any information not present in the provided context.\n"
            "4. **Disclaimer:** Conclude your response with the mandatory disclaimer: "
            "'This information is for informational purposes only and does not constitute medical advice.'\n\n"
            "CONTEXT\n"
            "{assembled_context}\n\n"
            "QUESTION\n"
            "{question}\n\n"
            "ANSWER\n"
        )

    def generate(self, query: str, assembled_context: str) -> str:
        prompt = self._prompt_template.format(
            assembled_context=assembled_context,
            question=query
        )
        try:
            response = self.llm.invoke(prompt).content
            return str(response).encode('utf-8', 'ignore').decode('utf-8')
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return "I am sorry, but I was unable to generate a response. Please try again later."
