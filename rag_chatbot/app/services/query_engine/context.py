import os
import logging
from typing import List, Dict, Any
from pymongo import MongoClient
from bson.codec_options import CodecOptions, UuidRepresentation
import tiktoken

class ContextAssembler:
    """
    Assembles the final context for the LLM by retrieving parent chunks,
    sorting them intelligently, and managing the token budget.
    """
    def __init__(self, db_name: str = "AdvanceRag", token_budget: int = 6000):
        self.mongo_client = MongoClient(os.getenv("MONGO_URI"))
        codec_options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
        self.db = self.mongo_client.get_database(db_name, codec_options=codec_options)
        self.collection = self.db.get_collection(db_name)
        self.token_budget = token_budget
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except:
            self.tokenizer = tiktoken.get_encoding("gpt2")


    def assemble(self, reranked_chunks: List[Dict]) -> str:
        """
        Assembles the final context string from the re-ranked child chunks.

        Args:
            reranked_chunks: The top 8 re-ranked chunks.

        Returns:
            A single string containing the formatted context for the LLM.
        """
        parent_ids = self._extract_parent_ids(reranked_chunks)
        parent_chunks = self._fetch_parent_chunks(parent_ids)
        sorted_chunks = self._intelligent_sort(parent_chunks)
        
        return self._build_context_string(sorted_chunks)

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

    def _build_context_string(self, sorted_chunks: List[Dict]) -> str:
        """Builds the final context string, managing the token budget."""
        context_str = ""
        current_tokens = 0

        for chunk in sorted_chunks:
            metadata = chunk.get('metadata', {})
            text = chunk.get('text', '')
            
            # Format the chunk with its metadata
            formatted_chunk = (
                f"--- Source Document: {metadata.get('pdf_name', 'N/A')}, "
                f"Page: {metadata.get('page_no', 'N/A')} ---\n"
                f"Section: {metadata.get('section_title', 'N/A')}\n\n"
                f"{text}\n\n"
            )
            
            chunk_token_count = len(self.tokenizer.encode(formatted_chunk))
            
            if current_tokens + chunk_token_count > self.token_budget:
                # Truncate the last chunk if it exceeds the budget
                remaining_tokens = self.token_budget - current_tokens
                if remaining_tokens > 100: # Only add if there's meaningful space
                    encoded_text = self.tokenizer.encode(formatted_chunk)
                    truncated_encoded_text = encoded_text[:remaining_tokens]
                    truncated_text = self.tokenizer.decode(truncated_encoded_text)
                    context_str += truncated_text
                break 

            context_str += formatted_chunk
            current_tokens += chunk_token_count
            
        return context_str.strip()
