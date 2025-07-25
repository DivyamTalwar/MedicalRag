import os
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding

# Load .env file from the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

def get_embedding_model():
    return OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
