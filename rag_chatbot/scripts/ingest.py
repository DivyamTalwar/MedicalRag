import os
import sys
import asyncio
import logging
import math
import re
from uuid import uuid4, UUID
from collections import defaultdict
from typing import List, Dict, Any

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_pinecone import PineconeVectorStore
from app.core.embeddings import get_embedding_model
from langchain_core.documents import Document as LangchainDocument
from llama_index.core import SimpleDirectoryReader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException
from pymongo import MongoClient
from bson.codec_options import CodecOptions, UuidRepresentation
from app.models.data_models import DocumentChunk, ChunkMetadata

load_dotenv()
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
PARENT_INDEX_NAME = "parents-medical"
CHILD_INDEX_NAME = "children-medical"
MONGO_DB_NAME = "AdvanceRag"
DIMENSION = 1024

def create_pinecone_indexes(pinecone_client: Pinecone, vector_size: int):
    for index_name in [PARENT_INDEX_NAME, CHILD_INDEX_NAME]:
        try:
            if index_name not in pinecone_client.list_indexes().names():
                logging.info(f"Creating Pinecone index: '{index_name}'")
                pinecone_client.create_index(
                    name=index_name,
                    dimension=vector_size,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            else:
                logging.info(f"Pinecone index '{index_name}' already exists.")
        except Exception as e:
            logging.error(f"Failed to create/check Pinecone index '{index_name}': {e}")
            raise

def clear_pinecone_index(pinecone_client: Pinecone, index_name: str):
    try:
        if index_name in pinecone_client.list_indexes().names():
            index = pinecone_client.Index(index_name)
            index.delete(delete_all=True)
            logging.info(f"Cleared all vectors from index '{index_name}'.")
    except PineconeException:
        logging.warning(f"Index '{index_name}' was empty or namespace not found. Nothing to clear.")
    except Exception as e:
        logging.error(f"Failed to clear Pinecone index '{index_name}': {e}")


def clean_formatting_artifacts(text: str) -> str:
    text = re.sub(r'\s+', ' ', text) 
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text) 
    
    char_replacements = {
    '&#xA;': '\n',
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&quot;': '"',
    '&#39;': "'",
    '&nbsp;': ' ',
}
    
    for artifact, replacement in char_replacements.items():
        text = text.replace(artifact, replacement)
    
    text = re.sub(r'\|\s*\|\s*\|', '| |', text) 
    text = re.sub(r'\|\s*$', '|', text, flags=re.MULTILINE) 
    
    text = re.sub(r'(\d+)\s*%', r'\1%', text) 
    text = re.sub(r'\$\s*(\d)', r'$\1', text) 
    
    bullet_rx = r'^\s*([•\u2022\*-])\s+'
    text = re.sub(bullet_rx, r'• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(\d+)\.\s+', r'\1. ', text, flags=re.MULTILINE)
    
    return text.strip()

def enhance_table_structure(text: str) -> str:    
    lines = text.split('\n')
    enhanced_lines = []
    in_table = False
    
    for i, line in enumerate(lines):
        if '|' in line and line.count('|') >= 2:
            if not in_table:
                in_table = True
                next_line = lines[i + 1] if i + 1 < len(lines) else ''
                is_already_separator = re.fullmatch(r'\s*\|(\s*[-:]+\s*\|)+\s*', next_line)
                if '|' in next_line and not is_already_separator:
                    enhanced_lines.append(line)
                    cells = line.split('|')
                    separator = '|' + '|'.join(['---' for _ in cells[1:-1]]) + '|'
                    enhanced_lines.append(separator)
                    continue
            
            cells = [cell.strip() for cell in line.split('|')]
            cleaned_line = '| ' + ' | '.join(cells[1:-1]) + ' |'
            enhanced_lines.append(cleaned_line)
        else:
            if in_table:
                in_table = False
            enhanced_lines.append(line)
    
    return '\n'.join(enhanced_lines)

def validate_chunk_quality(text: str) -> bool:
    if len(text.strip()) < 10:
        return False
    
    if len(text) > 0:
        special_char_ratio = len(re.findall(r'[^\w\s\-.,!?()|\n]', text)) / len(text)
        if special_char_ratio > 0.1:
            return False
    
    if '|' in text and not re.search(r'\|[^|]+\|', text):
        return False
    
    return True

async def main():
    logging.info("Starting ingestion with LlamaParse")

    try:
        mongo_client = MongoClient(os.getenv("MONGO_URI"))
        codec_options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
        db = mongo_client.get_database(MONGO_DB_NAME, codec_options=codec_options)
        collection = db.get_collection(MONGO_DB_NAME)
        logging.info(f"Successfully connected to MongoDB.")
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        return

    try:
        pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        logging.info("Successfully connected to Pinecone.")
    except Exception as e:
        logging.error(f"Failed to connect to Pinecone: {e}")
        return

    embeddings_model = get_embedding_model()
    create_pinecone_indexes(pinecone_client, DIMENSION)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        logging.warning("No PDF files found. Exiting.")
        return

    parser = LlamaParse(
        api_key=os.getenv("llama_index_api_key"),
        result_type="markdown",
        complemental_formatting_instruction="""
    Extract ALL content with maximum formatting preservation:
    - Maintain exact table structures with proper column alignment
    - Preserve all special characters, symbols, and formatting
    - Keep original spacing and indentation where meaningful
    - Convert complex tables to markdown format with | separators
    - Maintain header hierarchies and bullet point structures
    - Preserve numerical formatting (percentages, currency, decimals)
    - Keep footnotes and citations in their original context
    """,
        premium_mode=True,
        table_structure_recognition=True,
        image_ocr=True,
        parsing_instruction="Focus on maintaining original document structure and formatting",
        auto_mode_trigger_on_table_failure=True,
    )
    file_extractor = {".pdf": parser}
    
    try:
        raw_nodes = SimpleDirectoryReader(input_files=pdf_files, file_extractor=file_extractor).load_data()
        logging.info(f"Successfully parsed {len(raw_nodes)} nodes using LlamaParse.")
    except Exception as e:
        logging.error(f"Failed during LlamaParse extraction: {e}")
        return

    all_mongo_chunks = []
    parent_docs_for_pinecone = []
    child_docs_for_pinecone = []

    headers_to_split_on = [
        ("#", "Header 1"), 
        ("##", "Header 2"), 
        ("###", "Header 3"),
        ("####", "Header 4")
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, 
        strip_headers=False
    )

    nodes_by_doc = defaultdict(list)
    for node in raw_nodes:
        nodes_by_doc[node.metadata.get("file_name")].append(node)

    for pdf_name, nodes in nodes_by_doc.items():
        doc_id = uuid4()
        logging.info(f"Processing document: {pdf_name} (ID: {doc_id})")
        
        cleaned_nodes = [clean_formatting_artifacts(node.get_content()) for node in nodes]
        full_text = "\n\n".join(cleaned_nodes)
        full_text = enhance_table_structure(full_text) 
        
        num_parent_chunks = 3
        text_len = len(full_text)
        if text_len == 0:
            logging.warning(f"Document {pdf_name} is empty after cleaning. Skipping.")
            continue
        chunk_size = math.ceil(text_len / num_parent_chunks)
        parent_text_chunks = [full_text[i:i+chunk_size] for i in range(0, text_len, chunk_size)]

        for i, parent_text in enumerate(parent_text_chunks):
            if not parent_text.strip():
                continue
            parent_id = uuid4()
            parent_metadata = ChunkMetadata(
                doc_id=doc_id,
                parent_id=None,
                pdf_name=pdf_name,
                page_no=i + 1,
                order_idx=i,
                chunk_type="parent_document_part"
            )
            parent_chunk = DocumentChunk(chunk_id=parent_id, text=parent_text, metadata=parent_metadata)
            all_mongo_chunks.append(parent_chunk.model_dump())
            
            pinecone_parent_metadata = {k: str(v) for k, v in parent_chunk.metadata.model_dump().items() if v is not None}
            pinecone_parent_metadata["chunk_id"] = str(parent_chunk.chunk_id)
            parent_docs_for_pinecone.append(LangchainDocument(page_content=parent_chunk.text, metadata=pinecone_parent_metadata))

            child_docs = markdown_splitter.split_text(parent_text)
            for j, child_doc in enumerate(child_docs):
                if validate_chunk_quality(child_doc.page_content):
                    child_metadata = parent_metadata.model_copy(update={
                        "parent_id": parent_id, 
                        "order_idx": j + 1, 
                        "chunk_type": "child_chunk",
                        "section_title": child_doc.metadata.get("Header 1") or child_doc.metadata.get("Header 2") or child_doc.metadata.get("Header 3")
                    })
                    child_chunk = DocumentChunk(text=child_doc.page_content, metadata=child_metadata)
                    all_mongo_chunks.append(child_chunk.model_dump())
                    
                    pinecone_child_metadata = {k: str(v) for k, v in child_chunk.metadata.model_dump().items() if v is not None}
                    pinecone_child_metadata["chunk_id"] = str(child_chunk.chunk_id)
                    child_docs_for_pinecone.append(LangchainDocument(page_content=child_chunk.text, metadata=pinecone_child_metadata))
                else:
                    logging.warning(f"Skipped low-quality chunk in {pdf_name}")

    logging.info(f"Created {len(parent_docs_for_pinecone)} parent chunks and {len(child_docs_for_pinecone)} child chunks.")

    if all_mongo_chunks:
        try:
            collection.delete_many({})
            collection.insert_many(all_mongo_chunks)
            logging.info(f"Inserted {len(all_mongo_chunks)} chunks into MongoDB.")
        except Exception as e:
            logging.error(f"Failed to insert data into MongoDB: {e}")
            return

    logging.info("Upserting vectors to Pinecone in batches")
    try:
        clear_pinecone_index(pinecone_client, PARENT_INDEX_NAME)
        clear_pinecone_index(pinecone_client, CHILD_INDEX_NAME)
        
        async def upsert_parents():
            if parent_docs_for_pinecone:
                logging.info(f"Upserting {len(parent_docs_for_pinecone)} parent vectors to '{PARENT_INDEX_NAME}'...")
                await asyncio.to_thread(PineconeVectorStore.from_documents, documents=parent_docs_for_pinecone, embedding=embeddings_model, index_name=PARENT_INDEX_NAME, batch_size=100)
                logging.info(f"Finished upserting to '{PARENT_INDEX_NAME}'.")

        async def upsert_children():
            if child_docs_for_pinecone:
                logging.info(f"Upserting {len(child_docs_for_pinecone)} child vectors to '{CHILD_INDEX_NAME}'...")
                await asyncio.to_thread(PineconeVectorStore.from_documents, documents=child_docs_for_pinecone, embedding=embeddings_model, index_name=CHILD_INDEX_NAME, batch_size=100)
                logging.info(f"Finished upserting to '{CHILD_INDEX_NAME}'.")

        results = await asyncio.gather(upsert_parents(), upsert_children(), return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"An exception occurred during Pinecone upsert: {result}")

    except Exception as e:
        logging.error(f"Failed to upsert vectors to Pinecone: {e}")
        return

    logging.info("🎉 Ingestion process finished successfully.")

if __name__ == "__main__":
    asyncio.run(main())
