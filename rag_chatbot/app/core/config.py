import os

# Pinecone Configuration
PARENT_INDEX_NAME = os.getenv("PARENT_INDEX_NAME", "parent")
CHILD_INDEX_NAME = os.getenv("CHILD_INDEX_NAME", "children")
DIMENSION = int(os.getenv("DIMENSION", 1024))

# MongoDB Configuration
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "AdvanceRag")
