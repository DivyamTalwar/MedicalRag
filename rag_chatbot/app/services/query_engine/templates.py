def get_fallback_response(question: str, error_type: str) -> str:
    if error_type == "context_assembly":
        return (
            f"I encountered a technical issue retrieving specific details about {question}. "
            f"However, based on general CIVIE documentation patterns, I can provide "
            f"guidance on where this information would typically be found..."
        )
    else:
        return "I am sorry, but I could not generate a satisfactory answer. Please rephrase your question or try again."
