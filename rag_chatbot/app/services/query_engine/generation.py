import re
from typing import List, Dict, Any
from app.core.llm import CustomLLM

class CitationManager:
    """
    Manages the generation and tracking of citations for the LLM response.
    """
    def __init__(self, context_chunks: List[Dict]):
        self.context_chunks = context_chunks
        self.citation_map = {
            i + 1: chunk['metadata'] for i, chunk in enumerate(context_chunks)
        }

    def get_formatted_sources(self) -> str:
        """Formats the source chunks for inclusion in the prompt."""
        formatted_sources = []
        for i, chunk in enumerate(self.context_chunks):
            metadata = chunk.get('metadata', {})
            source_str = (
                f"Source [{i+1}]:\n"
                f"Document: {metadata.get('pdf_name', 'N/A')}\n"
                f"Page: {metadata.get('page_no', 'N/A')}\n"
                f"Section: {metadata.get('section_title', 'N/A')}\n"
                f"Content: {chunk.get('text', '')}\n"
            )
            formatted_sources.append(source_str)
        return "\n---\n".join(formatted_sources)

    def format_citations(self, response_text: str) -> str:
        """
        Finds citation markers like [1], [2], etc., in the text and replaces
        them with properly formatted, detailed citations.
        """
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
            return match.group(0) # Return original if not a valid citation

        return re.sub(r'\[(\d+)\]', replace_func, response_text)

class MedicalResponseValidator:
    """
    Ensures that the generated response meets medical accuracy and compliance standards.
    """
    def validate(self, response: str, citations_present: bool) -> bool:
        """
        Performs validation checks on the LLM's response.

        Args:
            response: The generated text from the LLM.
            citations_present: A boolean indicating if citations were found.

        Returns:
            True if the response is valid, False otherwise.
        """
        # 1. Check for mandatory disclaimer
        disclaimer = "this information is for informational purposes only"
        if disclaimer not in response.lower():
            return False
        
        # 2. Ensure claims are cited
        if not citations_present and len(response.split()) > 20:
             # Allow short, non-informational responses without citations
            return False

        # 3. Check for diagnostic language
        diagnostic_phrases = ["you have", "you are suffering from", "the diagnosis is"]
        if any(phrase in response.lower() for phrase in diagnostic_phrases):
            return False
            
        return True

class AnswerGenerator:
    """
    Generates the final, cited, and validated answer.
    """
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
            "--- SOURCES ---\n"
            "{formatted_sources}\n\n"
            "--- QUESTION ---\n"
            "{question}\n\n"
            "--- ANSWER ---\n"
        )

    def generate(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Generates the final answer.

        Args:
            query: The user's standalone question.
            context_chunks: The final list of context chunks.

        Returns:
            A formatted, cited, and validated final answer.
        """
        citation_manager = CitationManager(context_chunks)
        formatted_sources = citation_manager.get_formatted_sources()

        prompt = self._prompt_template.format(
            formatted_sources=formatted_sources,
            question=query
        )

        raw_response = self.llm.complete(prompt).text

        # Validation
        validator = MedicalResponseValidator()
        citations_found = bool(re.search(r'\[\d+\]', raw_response))
        
        if not validator.validate(raw_response, citations_found):
            return (
                "I could not generate a valid response based on the provided documents. "
                "Please try rephrasing your question."
            )

        # Add detailed citations
        final_response = citation_manager.format_citations(raw_response)

        return final_response
