import os
import logging
from typing import List, Dict, Any
from pymongo import MongoClient
from bson.codec_options import CodecOptions, UuidRepresentation

class ContextAssembler:
    """
    Assembles the final context for the LLM by retrieving and sorting parent chunks.
    """
    def __init__(self, db_name: str = "AdvanceRag"):
        self.mongo_client = MongoClient(os.getenv("MONGO_URI"))
        codec_options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
        self.db = self.mongo_client.get_database(db_name, codec_options=codec_options)
        self.collection = self.db.get_collection("medical_chunks")

    def assemble(self, reranked_chunks: List[Dict]) -> List[Dict]:
        """
        Assembles the final context by fetching and sorting parent chunks.

        Args:
            reranked_chunks: The top re-ranked child chunks.

        Returns:
            A sorted list of parent chunk documents.
        """
        parent_ids = self._extract_parent_ids(reranked_chunks)
        parent_chunks = self._fetch_parent_chunks(parent_ids)
        sorted_chunks = self._intelligent_sort(parent_chunks)
        
        return sorted_chunks

    def _extract_parent_ids(self, reranked_chunks: List[Dict]) -> List[str]:
        """Extracts unique parent IDs from the re-ranked chunks."""
        parent_ids = []
        for chunk in reranked_chunks:
            # Handle both dense (metadata dict) and sparse (full doc) results
            metadata = chunk.get('metadata', {})
            parent_id = metadata.get('parent_id')
            if parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)
        return parent_ids

    def _fetch_parent_chunks(self, parent_ids: List[str]) -> List[Dict]:
        """Fetches the full parent chunk documents from MongoDB."""
        if not parent_ids:
            return []
        
        try:
            # Query by the 'metadata.chunk_id' which is the parent's unique ID
            cursor = self.collection.find({
                "metadata.chunk_id": {"$in": parent_ids}
            })
            return list(cursor)
        except Exception as e:
            logging.error(f"Error fetching parent chunks from MongoDB: {e}")
            return []

    def _intelligent_sort(self, chunks: List[Dict]) -> List[Dict]:
        """Sorts chunks by page number, then by in-page order."""
        
        def sort_key(chunk):
            metadata = chunk.get('metadata', {})
            page_no = metadata.get('page_no', 999)
            order_idx = metadata.get('order_idx', 999)
            return (page_no, order_idx)
            
        return sorted(chunks, key=sort_key)
