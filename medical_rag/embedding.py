import requests
import numpy as np
from typing import List, Union
from config import EMBEDDING_API_URL, EMBEDDING_API_KEY, EMBEDDING_MODEL

class EmbeddingService:
    def __init__(self):
        self.api_url = EMBEDDING_API_URL
        self.api_key = EMBEDDING_API_KEY
        self.model = EMBEDDING_MODEL
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        print(f"[OK] Embedding service initialized: {self.model}")
    
    def get_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        payload = {
            "model": self.model,
            "input": text.strip(),
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"[ERROR] API returned {response.status_code}: {response.text}")
                raise Exception(f"Embedding API failed: {response.status_code}")
            
            result = response.json()
            
            if "result" in result:
                result = result["result"]
            
            if "data" not in result or not result["data"]:
                raise Exception("No embedding data in response")
            
            embedding = result["data"][0]["embedding"]
            print(f"[OK] Generated embedding: {len(embedding)} dimensions")
            return embedding
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed: {e}")
            raise Exception(f"Embedding request failed: {e}")
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        clean_texts = [text.strip() for text in texts if text and text.strip()]
        if not clean_texts:
            raise ValueError("No valid texts provided")
        
        payload = {
            "model": self.model,
            "input": clean_texts,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"[ERROR] API returned {response.status_code}: {response.text}")
                raise Exception(f"Embedding API failed: {response.status_code}")
            
            result = response.json()
            
            if "result" in result:
                result = result["result"]
            
            if "data" not in result:
                raise Exception("No data in response")
            
            embeddings = [item["embedding"] for item in result["data"]]
            print(f"[OK] Generated {len(embeddings)} embeddings")
            return embeddings
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Batch request failed: {e}")
            raise Exception(f"Batch embedding request failed: {e}")