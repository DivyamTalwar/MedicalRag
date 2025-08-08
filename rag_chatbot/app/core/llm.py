import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_llm():
    api_key = os.getenv("MODELS_API_KEY")
    llm_endpoint = "https://api.us.inc/omega/civie/v1"

    if not api_key or not llm_endpoint:
        raise ValueError("API key or endpoint not found. Make sure to set MODELS_API_KEY and LLM_ENDPOINT in your .env file")

    return ChatOpenAI(
        model="omega",
        api_key=api_key,
        base_url=llm_endpoint,
        temperature=0.2,
        max_tokens=8192,
        max_retries=5,
        request_timeout=600,
    )

CustomLLM = get_llm
