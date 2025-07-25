import os
import sys
import asyncio
import logging
from uuid import uuid4, UUID
from collections import defaultdict
import nest_asyncio

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document as LangchainDocument
from llama_index.core import SimpleDirectoryReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException
from pymongo import MongoClient
from bson.codec_options import CodecOptions, UuidRepresentation

from app.models.data_models import DocumentChunk, ChunkMetadata

load_dotenv()
nest_asyncio.apply()

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
PARENT_INDEX_NAME = "parents-medical"
CHILD_INDEX_NAME = "children-medical"
MONGO_DB_NAME = "AdvanceRag"
DIMENSION = 1536

# --- Helper Functions ---
def create_pinecone_indexes(pinecone_client: Pinecone, vector_size: int):
    for index_name in [PARENT_INDEX_NAME, CHILD_INDEX_NAME]:
        try:
            if index_name not in pinecone_client.list_indexes().names():
                logging.info(f"Creating Pinecone index: '{index_name}' with dimension {vector_size}")
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
            logging.info(f"✅ Cleared all vectors from index '{index_name}'.")
    except PineconeException:
        logging.warning(f"⚠️ Index '{index_name}' was empty or namespace not found. Nothing to clear.")
    except Exception as e:
        logging.error(f"❌ Failed to clear Pinecone index '{index_name}': {e}")

def process_page_chunks(page_text, doc_id, pdf_name, page_num, text_splitter, all_mongo_chunks, parent_docs, child_docs):
    """Processes a single page's text into parent and child chunks."""
    if not page_text.strip():
        return

    parent_id = uuid4()
    parent_metadata = ChunkMetadata(
        doc_id=doc_id,
        parent_id=None,
        pdf_name=pdf_name,
        page_no=page_num,
        order_idx=page_num,
        chunk_type="page",
    )
    parent_chunk = DocumentChunk(chunk_id=parent_id, text=page_text, metadata=parent_metadata)
    all_mongo_chunks.append(parent_chunk.model_dump())
    
    pinecone_parent_metadata = {k: str(v) for k, v in parent_chunk.metadata.model_dump().items() if v is not None}
    pinecone_parent_metadata["chunk_id"] = str(parent_chunk.chunk_id)
    parent_docs.append(LangchainDocument(page_content=parent_chunk.text, metadata=pinecone_parent_metadata))

    child_texts = text_splitter.split_text(page_text)
    for i, child_text in enumerate(child_texts):
        child_metadata = parent_metadata.model_copy(update={"parent_id": parent_id, "order_idx": i + 1, "chunk_type": "paragraph_child"})
        child_chunk = DocumentChunk(text=child_text, metadata=child_metadata)
        all_mongo_chunks.append(child_chunk.model_dump())
        
        pinecone_child_metadata = {k: str(v) for k, v in child_chunk.metadata.model_dump().items() if v is not None}
        pinecone_child_metadata["chunk_id"] = str(child_chunk.chunk_id)
        child_docs.append(LangchainDocument(page_content=child_chunk.text, metadata=pinecone_child_metadata))

# --- Main Ingestion Logic ---
async def main():
    logging.info("🚀 Starting ingestion process with LlamaParse based on Roadmap.txt")

    # 1. Connect to Services
    try:
        mongo_client = MongoClient(os.getenv("MONGO_URI"))
        codec_options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
        db = mongo_client.get_database(MONGO_DB_NAME, codec_options=codec_options)
        collection = db.get_collection(MONGO_DB_NAME)
        logging.info(f"✅ Successfully connected to MongoDB. Using collection: {MONGO_DB_NAME}")
    except Exception as e:
        logging.error(f"❌ Failed to connect to MongoDB: {e}")
        return

    try:
        pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        logging.info("✅ Successfully connected to Pinecone.")
    except Exception as e:
        logging.error(f"❌ Failed to connect to Pinecone: {e}")
        return

    # 2. Setup Embeddings and Indexes
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
    create_pinecone_indexes(pinecone_client, DIMENSION)

    # 3. Load and Parse Documents using LlamaParse
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        logging.warning("⚠️ No PDF files found in /data directory. Exiting.")
        return

    parser = LlamaParse(api_key=os.getenv("llama_index_api_key"), result_type="markdown")
    file_extractor = {".pdf": parser}
    
    try:
        raw_nodes = SimpleDirectoryReader(input_files=pdf_files, file_extractor=file_extractor).load_data()
        logging.info(f"✅ Successfully parsed {len(raw_nodes)} nodes using LlamaParse.")
    except Exception as e:
        logging.error(f"❌ Failed during LlamaParse extraction: {e}")
        return

    # 4. Process Nodes into Page-Level Parent Chunks and Child Chunks
    all_mongo_chunks = []
    parent_docs_for_pinecone = []
    child_docs_for_pinecone = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=40)

    # State for tracking document and page breaks
    current_doc_id = None
    current_pdf_name = None
    current_page_text = ""
    current_page_num = None

    for node in raw_nodes:
        node_pdf_name = node.metadata.get("file_name")
        node_page_num = int(node.metadata.get("page_label") or node.metadata.get("page_number", 0))

        if node_pdf_name != current_pdf_name:
            # Process the last page of the previous document
            if current_page_text:
                process_page_chunks(current_page_text, current_doc_id, current_pdf_name, current_page_num, text_splitter, all_mongo_chunks, parent_docs_for_pinecone, child_docs_for_pinecone)
            
            # Start a new document
            current_pdf_name = node_pdf_name
            current_doc_id = uuid4()
            logging.info(f"Processing document: {current_pdf_name} (ID: {current_doc_id})")
            current_page_text = node.get_content()
            current_page_num = node_page_num
        
        elif node_page_num != current_page_num:
            # Process the completed page
            process_page_chunks(current_page_text, current_doc_id, current_pdf_name, current_page_num, text_splitter, all_mongo_chunks, parent_docs_for_pinecone, child_docs_for_pinecone)
            
            # Start a new page
            current_page_text = node.get_content()
            current_page_num = node_page_num
        
        else:
            # Continue accumulating text for the current page
            current_page_text += "\n\n" + node.get_content()

    # Process the very last page of the last document
    if current_page_text:
        process_page_chunks(current_page_text, current_doc_id, current_pdf_name, current_page_num, text_splitter, all_mongo_chunks, parent_docs_for_pinecone, child_docs_for_pinecone)

    logging.info(f"✅ Created {len(parent_docs_for_pinecone)} parent chunks and {len(child_docs_for_pinecone)} child chunks.")

    # 5. Store in MongoDB
    if all_mongo_chunks:
        try:
            collection.delete_many({})
            collection.insert_many(all_mongo_chunks)
            logging.info(f"✅ Inserted {len(all_mongo_chunks)} chunks into MongoDB.")
        except Exception as e:
            logging.error(f"❌ Failed to insert data into MongoDB: {e}")
            return

    # 6. Upsert to Pinecone
    logging.info("⏳ Upserting vectors to Pinecone in batches...")
    try:
        clear_pinecone_index(pinecone_client, PARENT_INDEX_NAME)
        clear_pinecone_index(pinecone_client, CHILD_INDEX_NAME)
        
        batch_size = 100
        if parent_docs_for_pinecone:
            logging.info(f"Upserting {len(parent_docs_for_pinecone)} parent vectors to '{PARENT_INDEX_NAME}'...")
            for i in range(0, len(parent_docs_for_pinecone), batch_size):
                batch = parent_docs_for_pinecone[i:i + batch_size]
                PineconeVectorStore.from_documents(documents=batch, embedding=embeddings_model, index_name=PARENT_INDEX_NAME)
                logging.info(f"  - Upserted batch {i//batch_size + 1}")
            logging.info(f"✅ Finished upserting to '{PARENT_INDEX_NAME}'.")

        if child_docs_for_pinecone:
            logging.info(f"Upserting {len(child_docs_for_pinecone)} child vectors to '{CHILD_INDEX_NAME}'...")
            for i in range(0, len(child_docs_for_pinecone), batch_size):
                batch = child_docs_for_pinecone[i:i + batch_size]
                PineconeVectorStore.from_documents(documents=batch, embedding=embeddings_model, index_name=CHILD_INDEX_NAME)
                logging.info(f"  - Upserted batch {i//batch_size + 1}")
            logging.info(f"✅ Finished upserting to '{CHILD_INDEX_NAME}'.")

    except Exception as e:
        logging.error(f"❌ Failed to upsert vectors to Pinecone: {e}")
        return

    logging.info("🎉 Ingestion process finished successfully.")

if __name__ == "__main__":
    asyncio.run(main())
