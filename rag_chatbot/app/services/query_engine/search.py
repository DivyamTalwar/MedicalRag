import os
import json
import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from rag_chatbot.app.models.data_models import Document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InMeiliSearch:
    """
    A simplified, in-memory search engine that acts as the first-pass retriever.
    It loads the data from the JSON file and performs a basic keyword search.
    """
    def __init__(self, file_path: str = "civie_ris_metadata.json"):
        self.documents = self._load_documents(file_path)

    def _load_documents(self, file_path: str) -> List[Document]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            parent_chunks = data.get("parent_chunks", [])
            
            return [
                Document(
                    id=chunk.get("chunk_id"),
                    text=chunk.get("metadata", {}).get("text", ""),
                    metadata=chunk.get("metadata", {})
                )
                for chunk in parent_chunks
            ]
        except Exception as e:
            logging.error(f"FATAL: Failed to load or parse metadata from {file_path}: {e}")
            return []

    def search(self, search_terms: List[str], top_k: int = 30) -> List[Document]:
        if not self.documents:
            return []
            
        results = []
        search_terms_lower = {term.lower() for term in search_terms}
        
        for doc in self.documents:
            text_lower = doc.text.lower()
            score = sum(1 for term in search_terms_lower if term in text_lower)
            
            if score > 0:
                # Add score to a temporary metadata field for sorting
                doc.metadata['search_score'] = score
                results.append(doc)
        
        # Sort by score and return the top_k results
        return sorted(results, key=lambda x: x.metadata['search_score'], reverse=True)[:top_k]

class Reranker:
    """
    A powerful cross-encoder reranker to refine the results from the initial search.
    """
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        try:
            self.model = CrossEncoder(model_name)
            logging.info(f"Cross-encoder model '{model_name}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load cross-encoder model: {e}")
            self.model = None

    def rerank(self, query: str, documents: List[Document], top_k: int = 8) -> List[Document]:
        if not self.model or not documents:
            return []

        # The cross-encoder expects pairs of [query, document_text]
        pairs = [[query, doc.text] for doc in documents]
        
        try:
            # This returns a list of scores, one for each pair
            scores = self.model.predict(pairs, show_progress_bar=False)

            # Add the score to each document's metadata
            for doc, score in zip(documents, scores):
                doc.metadata['rerank_score'] = float(score)

            # Sort the documents by the new rerank_score in descending order
            return sorted(documents, key=lambda x: x.metadata['rerank_score'], reverse=True)[:top_k]
        except Exception as e:
            logging.error(f"Error during reranking: {e}")
            return []
