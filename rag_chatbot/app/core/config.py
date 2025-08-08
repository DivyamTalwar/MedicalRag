import os

PARENT_INDEX_NAME = os.getenv("PARENT_INDEX_NAME", "parent")
CHILD_INDEX_NAME = os.getenv("CHILD_INDEX_NAME", "children")
DIMENSION = int(os.getenv("DIMENSION", 1024))

MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "AdvanceRag")

LLM_ENDPOINT = os.getenv("LLM_ENDPOINT")
MODELS_API_KEY = os.getenv("MODELS_API_KEY")
