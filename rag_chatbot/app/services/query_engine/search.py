import os
import time
import logging
from typing import List, Dict, Any
from pinecone import Pinecone
from pymongo import MongoClient
from bson.codec_options import CodecOptions, UuidRepresentation
from app.core.embeddings import CustomEmbedding

class DenseSearchEngine:
    """
    Finds semantically similar content using vector embeddings from Pinecone.
    """
    def __init__(self, embeddings: CustomEmbedding, index_name: str = "children"):
        self.embeddings = embeddings
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pinecone_client.Index(index_name)
        self.index_name = index_name

    def search(self, query_text: str, top_k: int = 20, max_retries: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the Pinecone index with retry logic.

        Args:
            query_text: The text to search for (hypothetical document).
            top_k: The number of results to return.
            max_retries: The maximum number of retry attempts.

        Returns:
            A list of search results with metadata.
        """
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
                time.sleep(2 ** attempt) # Exponential backoff
        return []

class SparseSearchEngine:
    """
    Finds keyword-relevant content using MongoDB's text search capabilities.
    """
    def __init__(self, db_name: str = "AdvanceRag"):
        self.mongo_client = MongoClient(os.getenv("MONGO_URI"))
        codec_options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
        self.db = self.mongo_client.get_database(db_name, codec_options=codec_options)
        self.collection = self.db.get_collection(db_name)
        self._ensure_text_index()

    def _ensure_text_index(self):
        """Ensures a text index exists on the relevant fields."""
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
        """
        Searches the MongoDB collection for parent chunks matching the search terms.

        Args:
            search_terms: A list of expanded query terms.
            top_k: The number of results to return.

        Returns:
            A list of parent chunk documents with their BM25 scores.
        """
        query = " ".join(search_terms)
        try:
            results = self.collection.find(
                {
                    "$text": {"$search": query},
                    "metadata.chunk_type": "parent_standard" # Search only parent chunks
                },
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(top_k)
            
            return list(results)
        except Exception as e:
            logging.error(f"Error during sparse search in MongoDB: {e}")
            return []

class ResultMerger:
    """
    Combines and deduplicates dense and sparse search results.
    """
    def merge(self, dense_results: List[Dict], sparse_results: List[Dict], dense_weight: float = 0.6) -> List[Dict]:
        """
        Merges, normalizes, and deduplicates search results.

        Args:
            dense_results: Results from the dense search (Pinecone).
            sparse_results: Results from the sparse search (MongoDB).
            dense_weight: The weight to apply to the dense search scores.

        Returns:
            A combined and deduplicated list of results.
        """
        # Normalize scores
        max_dense_score = max([r['score'] for r in dense_results], default=1.0)
        max_sparse_score = max([r['score'] for r in sparse_results], default=1.0)

        for r in dense_results:
            r['normalized_score'] = (r['score'] / max_dense_score) * dense_weight if max_dense_score > 0 else 0
        
        for r in sparse_results:
            r['normalized_score'] = (r['score'] / max_sparse_score) * (1 - dense_weight) if max_sparse_score > 0 else 0

        # Combine and deduplicate
        combined_results = {}
        for r in dense_results:
            # Use parent_id for deduplication key
            parent_id = r['metadata'].get('parent_id')
            if parent_id and parent_id not in combined_results:
                 combined_results[parent_id] = r
            elif parent_id and r['normalized_score'] > combined_results[parent_id]['normalized_score']:
                 combined_results[parent_id] = r


        for r in sparse_results:
            parent_id = str(r['_id']) # In sparse search, the ID is the parent ID
            if parent_id not in combined_results:
                combined_results[parent_id] = r
            else:
                # Add scores if both searches found the same parent
                combined_results[parent_id]['normalized_score'] += r['normalized_score']

        # Sort by final score
        sorted_results = sorted(combined_results.values(), key=lambda x: x['normalized_score'], reverse=True)
        
        return sorted_results[:30] # Return top 30 candidates for re-ranking

class CrossEncoderReranker:
    """
    Uses a powerful cross-encoder model to re-rank the most relevant chunks.
    """
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Dict], top_k: int = 8) -> List[Dict]:
        """
        Re-ranks documents based on their relevance to the query.

        Args:
            query: The original user query.
            documents: The list of candidate documents from the merge step.
            top_k: The final number of documents to return.

        Returns:
            The top_k most relevant documents.
        """
        if not documents:
            return []

        pairs = [[query, doc.get('text', doc.get('page_content', ''))] for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc['rerank_score'] = score

        # Sort by the new rerank score
        reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked_docs[:top_k]
