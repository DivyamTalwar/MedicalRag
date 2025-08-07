import os
import logging
from typing import List, Dict, Any, Tuple
from pymongo import MongoClient
from bson.codec_options import CodecOptions, UuidRepresentation
from rag_chatbot.app.models.data_models import Document
from langchain_core.language_models.llms import BaseLLM

class ContextAssembler:
    def __init__(self, db_name: str = "AdvanceRag"):
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise ValueError("MONGO_URI environment variable not set.")
        
        self.mongo_client = MongoClient(mongo_uri)
        codec_options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
        self.db = self.mongo_client.get_database(db_name, codec_options=codec_options)
        self.collection = self.db.get_collection("medical_chunks")

    def assemble(self, reranked_chunks: List[Document]) -> Tuple[List[Document], str]:
        if not reranked_chunks:
            return [], ""
        
        parent_ids = self._extract_parent_ids(reranked_chunks)
        parent_chunks_data = self._fetch_parent_chunks(parent_ids)
        sorted_chunks_data = self._intelligent_sort(parent_chunks_data)
        
        parent_chunks_docs = [Document(
            id=str(chunk.get('_id', chunk.get('id', ''))),
            text=chunk.get('text', ''),
            metadata=chunk.get('metadata', {})
        ) for chunk in sorted_chunks_data]
        
        assembled_context = "\n\n".join([doc.text for doc in parent_chunks_docs])
        
        return parent_chunks_docs, assembled_context

    def _extract_parent_ids(self, reranked_chunks: List[Document]) -> List[str]:
        parent_ids = []
        for chunk in reranked_chunks:
            parent_id = chunk.metadata.get('parent_id')
            if parent_id and str(parent_id) not in parent_ids:
                parent_ids.append(str(parent_id))
        return parent_ids

    def _fetch_parent_chunks(self, parent_ids: List[str]) -> List[Dict]:
        if not parent_ids:
            return []
        
        try:
            cursor = self.collection.find({
                "metadata.chunk_id": {"$in": parent_ids}
            })
            return list(cursor)
        except Exception as e:
            logging.error(f"Error fetching parent chunks from MongoDB: {e}")
            return []

    def _intelligent_sort(self, chunks: List[Dict]) -> List[Dict]:
        def sort_key(chunk):
            metadata = chunk.get('metadata', {})
            page_no = metadata.get('page_no', 999)
            order_idx = metadata.get('order_idx', 999)
            return (page_no, order_idx)
            
        return sorted(chunks, key=sort_key)

class ContextManager:
    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def evaluate_sufficiency(self, query: str, context: List[Document]) -> bool:
        if not context:
            return False
        
        context_text = "\n\n".join([doc.text for doc in context])
        
        prompt = f"""
        Given the following query and context, evaluate if the context contains sufficient information to provide a comprehensive and accurate answer.
        
        Query: {query}
        
        Context:
        ---
        {context_text}
        ---
        
        Is the context sufficient? Please answer with only "yes" or "no".
        """
        
        try:
            response = self.llm.invoke(prompt)
            decision = response.strip().lower()
            logging.info(f"Context sufficiency evaluation: LLM responded with '{decision}'")
            return "yes" in decision
        except Exception as e:
            logging.error(f"Error during LLM-based context sufficiency evaluation: {e}")
            return False
