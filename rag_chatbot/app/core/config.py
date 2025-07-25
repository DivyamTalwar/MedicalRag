import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
MODELS_API_KEY = os.getenv("MODELS_API_KEY")
STREAM_MODE = os.getenv("STREAM_MODE", "True").lower() in ("true", "1", "t")
