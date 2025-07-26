import os
import requests
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from typing import List

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

class CustomEmbedding(Embeddings):
    def __init__(self, api_url: str, api_key: str, model: str = "usf1-embed"):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

        valid_texts = []
        valid_indices = []
        max_text_length = 8000

        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text[:max_text_length]) 
                valid_indices.append(i)

        if not valid_texts:
            return [[0.0] * 1024 for _ in texts]

        all_valid_embeddings = []
        batch_size = 32
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            payload = {
                "model": self.model,
                "input": batch
            }
            try:
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                if not (isinstance(data, dict) and 'result' in data and 'data' in data['result']):
                    raise RuntimeError(f"Unexpected embedding API response format: {data}")
                
                new_embeddings = [[float(v) for v in item['embedding']] for item in data['result']['data']]
                all_valid_embeddings.extend(new_embeddings)
            except requests.exceptions.RequestException as e:
                print(f"Embedding API request failed. Error: {e}")
                raise

        final_embeddings = [[0.0] * 1024 for _ in texts]
        for i, original_index in enumerate(valid_indices):
            final_embeddings[original_index] = all_valid_embeddings[i]
            
        return final_embeddings

    def embed_query(self, text: str) -> List[float]:
        if not text or not text.strip():
            return [0.0] * 1024 
        
        embeddings = self.embed_documents([text])
        if not embeddings:
            return [0.0] * 1024
        return embeddings[0]

def get_embedding_model() -> CustomEmbedding:
    api_url = os.getenv("EMBEDDING_API_URL")
    api_key = os.getenv("MODELS_API_KEY")
    
    if not api_url or not api_key:
        raise ValueError("EMBEDDING_API_URL and MODELS_API_KEY must be set in the .env file")
        
    return CustomEmbedding(
        api_url=api_url,
        api_key=api_key
    )
