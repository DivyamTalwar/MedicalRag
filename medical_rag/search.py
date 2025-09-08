from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
from embedding import EmbeddingService
from config import PINECONE_API_KEY, PINECONE_INDEX, PINECONE_ENV

class SearchService:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX
        self.embedding_service = EmbeddingService()
        self.index = self._get_or_create_index()
        print(f"[OK] Search service connected to Pinecone index: {self.index_name}")
    
    def _get_or_create_index(self):
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            print(f"[OK] Connected to existing index: {stats['total_vector_count']} vectors")
            return index
        except Exception as e:
            print(f"[INFO] Index not found, creating new index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            index = self.pc.Index(self.index_name)
            print(f"[OK] Created new index: {self.index_name}")
            return index
    
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None) -> bool:
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        print(f"[INFO] Adding {len(texts)} documents to index...")
        embeddings = self.embedding_service.get_embeddings_batch(texts)
        
        vectors = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            metadata['text'] = text
            vectors.append({
                "id": f"doc_{i}_{hash(text[:50])}",
                "values": embedding,
                "metadata": metadata
            })
        
        try:
            self.index.upsert(vectors=vectors)
            print(f"[OK] Added {len(vectors)} documents to vector database")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to add documents: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5, filter_dict: Dict = None) -> List[Dict]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        print(f"[INFO] Searching for: '{query}' (top_k={top_k})")
        query_embedding = self.embedding_service.get_embedding(query.strip())
        
        try:
            search_results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            results = []
            for match in search_results["matches"]:
                result = {
                    "id": match["id"],
                    "score": float(match["score"]),
                    "text": match.get("metadata", {}).get("text", ""),
                    "metadata": match.get("metadata", {})
                }
                results.append(result)
            
            print(f"[OK] Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            print(f"[ERROR] Search failed: {e}")
            return []
    
    def get_stats(self) -> Dict:
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.get("total_vector_count", 0),
                "index_name": self.index_name,
                "dimension": stats.get("dimension", 1024),
                "status": "connected"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }