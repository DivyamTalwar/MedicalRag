import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME", "legendary-medical-rag")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

OMEGA_API_KEY = os.getenv("OMEGA_API_KEY")
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")
OMEGA_MODEL = os.getenv("OMEGA_MODEL_NAME", "omega-medical-v1")

EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
EMBEDDING_API_KEY = os.getenv("MODELS_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "medical-embeddings-v1")

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.75"))

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env")
if not OMEGA_API_KEY:
    raise ValueError("OMEGA_API_KEY not found in .env")
if not EMBEDDING_API_KEY:
    raise ValueError("EMBEDDING_API_KEY not found in .env")

print("[OK] Config loaded successfully")
print(f"[OK] Pinecone: {PINECONE_INDEX}")
print(f"[OK] LLM: {OMEGA_MODEL}")  
print(f"[OK] Embeddings: {EMBEDDING_MODEL}")