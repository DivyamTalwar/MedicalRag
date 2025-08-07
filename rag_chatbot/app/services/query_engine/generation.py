import re
from typing import List, Dict, Any
from rag_chatbot.app.core.llm import CustomLLM

class CitationManager:
    def __init__(self, context_chunks: List[Dict]):
        self.context_chunks = context_chunks
        self.citation_map = {
            i + 1: chunk.metadata for i, chunk in enumerate(context_chunks)
        }

    def get_formatted_sources(self) -> str:
        formatted_sources = []
        for i, chunk in enumerate(self.context_chunks):
            metadata = chunk.metadata
            source_str = (
                f"Source [{i+1}]:\n"
                f"Document: {metadata.get('pdf_name', 'N/A')}\n"
                f"Page: {metadata.get('page_no', 'N/A')}\n"
                f"Section: {metadata.get('section_title', 'N/A')}\n"
                f"Content: {chunk.text}\n"
            )
            formatted_sources.append(source_str)
        return "\n---\n".join(formatted_sources)

    def format_citations(self, response_text: str) -> str:
        def replace_func(match):
            try:
                source_num = int(match.group(1))
                if source_num in self.citation_map:
                    metadata = self.citation_map[source_num]
                    return (
                        f" [Source: {metadata.get('pdf_name', 'N/A')}, "
                        f"Page: {metadata.get('page_no', 'N/A')}]"
                    )
            except (ValueError, IndexError):
                pass
            return match.group(0)

        return re.sub(r'\[(\d+)\]', replace_func, response_text)

class AnswerGenerator:
    def __init__(self, llm: CustomLLM):
        self.llm = llm
        self._prompt_template = self._create_prompt_template()

    def _create_prompt_template(self) -> str:
        return (
            "You are a highly knowledgeable medical AI assistant. Your task is to provide a "
            "precise, evidence-based answer to the user's question based *only* on the "
            "provided sources. You must adhere to the following rules:\n"
            "1. **Cite Everything:** For every piece of information you use, you must cite the "
            "corresponding source number, like this: [1], [2], etc.\n"
            "2. **Be Concise:** Synthesize the information from the sources into a clear and "
            "easy-to-understand answer. Do not copy the sources verbatim.\n"
            "3. **No Outside Knowledge:** Do not use any information not present in the provided sources.\n"
            "4. **Disclaimer:** Conclude your response with the mandatory disclaimer: "
            "'This information is for informational purposes only and does not constitute medical advice.'\n\n"
            "SOURCES\n"
            "{formatted_sources}\n\n"
            "QUESTION\n"
            "{question}\n\n"
            "ANSWER\n"
        )

    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        citation_manager = CitationManager(context_chunks)
        formatted_sources = citation_manager.get_formatted_sources()

        prompt = self._prompt_template.format(
            formatted_sources=formatted_sources,
            question=query
        )

        raw_response = self.llm.complete(prompt).text

        final_response = citation_manager.format_citations(raw_response)

        return final_response
