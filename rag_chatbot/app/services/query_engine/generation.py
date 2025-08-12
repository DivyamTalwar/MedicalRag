import re
import time
import logging
from typing import List, Dict, Any
from rag_chatbot.app.core.llm import CustomLLM
from rag_chatbot.app.models.data_models import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self, llm: CustomLLM):
        self.llm = llm
        self._prompt_template = self._create_prompt_template()

    def _create_prompt_template(self) -> str:
        return """You are an expert medical AI assistant. Your sole purpose is to answer questions based *strictly* on the provided context and chat history. Do not use any external knowledge.

**Rules:**
1.  **Context is King:** Your answer must be derived exclusively from the `CONTEXT` section. Do not invent or infer information.
2.  **Synthesize and Detail:** Provide a comprehensive, detailed synthesis of all relevant information. Do not simply summarize.
3.  **Acknowledge Limitations:** If the context does not contain the answer, state that clearly. For example: 'The provided context does not contain information about [topic].'
4.  **Mandatory Disclaimer:** Always conclude your entire response with the following disclaimer on a new line: `This information is for informational purposes only and does not constitute medical advice.`

---

**CHAT HISTORY:**
{chat_history}

**CONTEXT:**
{assembled_context}

**QUESTION:**
{question}

---

**Example Response:**
Based on the provided context, the patient's Arterial Blood Gas (ABG) analysis shows a pH of 7.35, which is at the lower end of the normal range (7.35-7.45). The PCO2 is 45 mmHg, and the HCO3 is 24 mEq/L, both of which are within their respective normal ranges.

This information is for informational purposes only and does not constitute medical advice.

**YOUR ANSWER:**
"""

    def generate(self, query: str, assembled_context: str, chat_history: List = []) -> str:
        history_str = "\n".join(
            [f"{msg.role.capitalize()}: {msg.content}" for msg in chat_history]
        )
        prompt = self._prompt_template.format(
            assembled_context=assembled_context,
            question=query,
            chat_history=history_str
        )
        try:
            response = self.llm.invoke(prompt)['content']
            return str(response).encode('utf-8', 'ignore').decode('utf-8')
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return "I am sorry, but I was unable to generate a response. Please try again later."
