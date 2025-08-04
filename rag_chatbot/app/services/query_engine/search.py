import os
import time
import logging
from typing import List, Dict, Any
from pinecone import Pinecone
from pymongo import MongoClient
from bson.codec_options import CodecOptions, UuidRepresentation
from app.core.embeddings import CustomEmbedding

class DenseSearchEngine:
    def __init__(self, index_name: str = "children"):
        self.embeddings = CustomEmbedding()
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pinecone_client.Index(index_name)
        self.index_name = index_name

    def search(self, query_text: str, top_k: int = 20, max_retries: int = 3) -> List[Dict[str, Any]]:
        for attempt in range(max_retries):
            try:
                query_vector = self.embeddings.embed_query(query_text)
                
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True
                )
                
                return [
                    {
                        "id": match["id"],
                        "score": match["score"],
                        "metadata": match["metadata"],
                    }
                    for match in results["matches"]
                ]
            except Exception as e:
                logging.warning(f"Pinecone search attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logging.error("All Pinecone search attempts failed.")
                    return []
                time.sleep(2 ** attempt)
        return []

class SparseSearchEngine:
    def __init__(self, db_name: str = "AdvanceRag"):
        self.mongo_client = MongoClient(
            os.getenv("MONGO_URI"),
            maxPoolSize=50,
            minPoolSize=10,
            maxIdleTimeMS=30000,
            serverSelectionTimeoutMS=5000
        )
        codec_options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
        self.db = self.mongo_client.get_database(db_name, codec_options=codec_options)
        self.collection = self.db.get_collection("medical_chunks")
        self._ensure_text_index()

    def _ensure_text_index(self):
        index_info = self.collection.index_information()
        index_name = "text_search_index"
        
        if index_name not in index_info:
            logging.info("Creating text index in MongoDB...")
            self.collection.create_index(
                [
                    ("text", "text"),
                    ("metadata.section_title", "text"),
                    ("metadata.searchable_terms", "text"),
                ],
                name=index_name,
                weights={
                    "text": 1,
                    "metadata.section_title": 5,
                    "metadata.searchable_terms": 10,
                },
                default_language="english"
            )
            logging.info("Text index created.")

    def search(self, search_terms: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        query = " ".join(search_terms)
        try:
            results = self.collection.find(
                {
                    "$text": {"$search": query},
                    "metadata.chunk_type": "parent_standard" 
                },
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(top_k)
            
            return list(results)
        except Exception as e:
            logging.error(f"Error during sparse search in MongoDB: {e}")
            return []

class ResultMerger:
    def merge(self, dense_results: List[Dict], sparse_results: List[Dict], dense_weight: float = 0.6) -> List[Dict]:
        if not dense_results and not sparse_results:
            return []

        if not dense_results:
            dense_weight = 0.0
        elif not sparse_results:
            dense_weight = 1.0
            
        max_dense_score = max([r['score'] for r in dense_results], default=1.0)
        max_sparse_score = max([r['score'] for r in sparse_results], default=1.0)

        for r in dense_results:
            r['normalized_score'] = (r['score'] / max_dense_score) * dense_weight if max_dense_score > 0 else 0
        
        for r in sparse_results:
            r['normalized_score'] = (r['score'] / max_sparse_score) * (1 - dense_weight) if max_sparse_score > 0 else 0

        combined_results = {}
        for r in dense_results:
            parent_id = r['metadata'].get('parent_id')
            if parent_id and parent_id not in combined_results:
                combined_results[parent_id] = r
            elif parent_id and r['normalized_score'] > combined_results[parent_id]['normalized_score']:
                combined_results[parent_id] = r


        for r in sparse_results:
            parent_id = str(r['_id'])
            if parent_id not in combined_results:
                combined_results[parent_id] = r
            else:
                combined_results[parent_id]['normalized_score'] += r['normalized_score']

        sorted_results = sorted(combined_results.values(), key=lambda x: x['normalized_score'], reverse=True)
        
        return sorted_results[:30]

class CrossEncoderReranker:
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict], top_k: int = 8) -> List[Dict]:
        if not documents:
            return []

        pairs = [[query, doc.get('text', doc.get('page_content', ''))] for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc['rerank_score'] = score

        reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked_docs[:top_k]
