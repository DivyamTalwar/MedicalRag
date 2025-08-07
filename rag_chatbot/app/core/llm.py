import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def get_llm():
    """
    Initializes and returns a ChatOpenAI instance configured for the custom LLM.
    """
    api_key = os.getenv("MODELS_API_KEY")
    # The base URL should not include the /chat/completions part
    llm_endpoint = "https://api.us.inc/omega/civie/v1"

    if not api_key or not llm_endpoint:
        raise ValueError("API key or endpoint not found. Make sure to set MODELS_API_KEY and LLM_ENDPOINT in your .env file")

    return ChatOpenAI(
        model="omega",
        api_key=api_key,
        base_url=llm_endpoint,
        temperature=0.7,
        max_tokens=4096,
        max_retries=5,  # Add robust retry logic
    )

# For compatibility with the existing code, we can assign the function to a variable
# that looks like a class constructor.
CustomLLM = get_llm
