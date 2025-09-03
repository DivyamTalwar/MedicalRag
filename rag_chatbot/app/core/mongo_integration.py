"""
LEGENDARY MongoDB Integration
Document storage and metadata management
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.errors import ConnectionFailure, OperationFailure
import gridfs
from bson import ObjectId

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegendaryMongoStore:
    """
    MongoDB integration for medical document management
    Handles document storage, metadata, and versioning
    """
    
    def __init__(self):
        """Initialize MongoDB connection"""
        # Get configuration from environment
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        self.database_name = os.getenv("MONGODB_DATABASE", "legendary_medical_rag")
        self.collection_name = os.getenv("MONGODB_COLLECTION", "medical_documents")
        
        try:
            # Connect to MongoDB
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            
            # Get database and collections
            self.db = self.client[self.database_name]
            self.documents = self.db[self.collection_name]
            self.chunks = self.db["document_chunks"]
            self.queries = self.db["query_history"]
            self.feedback = self.db["user_feedback"]
            
            # Initialize GridFS for large files
            self.fs = gridfs.GridFS(self.db)
            
            # Create indexes
            self._create_indexes()
            
            logger.info(f"MongoDB initialized: {self.database_name}/{self.collection_name}")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"MongoDB initialization error: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for performance"""
        try:
            # Document indexes
            self.documents.create_index([("document_id", ASCENDING)], unique=True)
            self.documents.create_index([("created_at", DESCENDING)])
            self.documents.create_index([("metadata.type", ASCENDING)])
            self.documents.create_index([("metadata.specialty", ASCENDING)])
            self.documents.create_index([("$**", TEXT)])  # Text search index
            
            # Chunk indexes
            self.chunks.create_index([("document_id", ASCENDING)])
            self.chunks.create_index([("chunk_index", ASCENDING)])
            self.chunks.create_index([("embedding_id", ASCENDING)])
            
            # Query history indexes
            self.queries.create_index([("timestamp", DESCENDING)])
            self.queries.create_index([("user_id", ASCENDING)])
            
            logger.info("Database indexes created successfully")
            
        except OperationFailure as e:
            logger.warning(f"Index creation warning: {e}")
    
    def store_document(self, 
                       file_path: str,
                       content: str,
                       metadata: Optional[Dict] = None) -> str:
        """Store a document with metadata"""
        try:
            # Generate document ID
            doc_hash = hashlib.sha256(content.encode()).hexdigest()
            document_id = f"doc_{doc_hash[:16]}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            
            # Prepare document
            document = {
                "document_id": document_id,
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "content_hash": doc_hash,
                "content_length": len(content),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "version": 1,
                "metadata": metadata or {},
                "processing_status": "pending",
                "statistics": {
                    "word_count": len(content.split()),
                    "char_count": len(content),
                    "line_count": len(content.splitlines())
                }
            }
            
            # Store content in GridFS if large
            if len(content) > 1024 * 1024:  # 1MB threshold
                file_id = self.fs.put(
                    content.encode('utf-8'),
                    filename=Path(file_path).name,
                    document_id=document_id,
                    content_type="text/plain"
                )
                document["content_gridfs_id"] = file_id
                document["content_stored_in"] = "gridfs"
            else:
                document["content"] = content
                document["content_stored_in"] = "document"
            
            # Insert document
            result = self.documents.insert_one(document)
            
            logger.info(f"Document stored: {document_id} ({Path(file_path).name})")
            return document_id
            
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            raise
    
    def store_chunks(self, 
                    document_id: str,
                    chunks: List[Dict[str, Any]]) -> int:
        """Store document chunks with embeddings reference"""
        try:
            chunk_documents = []
            
            for i, chunk in enumerate(chunks):
                chunk_doc = {
                    "document_id": document_id,
                    "chunk_id": f"{document_id}_chunk_{i}",
                    "chunk_index": i,
                    "text": chunk["text"],
                    "start_char": chunk.get("start_char", 0),
                    "end_char": chunk.get("end_char", len(chunk["text"])),
                    "embedding_id": chunk.get("embedding_id"),
                    "metadata": chunk.get("metadata", {}),
                    "created_at": datetime.utcnow()
                }
                chunk_documents.append(chunk_doc)
            
            # Bulk insert chunks
            if chunk_documents:
                result = self.chunks.insert_many(chunk_documents)
                
                # Update document status
                self.documents.update_one(
                    {"document_id": document_id},
                    {
                        "$set": {
                            "processing_status": "chunked",
                            "chunk_count": len(chunk_documents),
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                
                logger.info(f"Stored {len(chunk_documents)} chunks for document {document_id}")
                return len(result.inserted_ids)
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            raise
    
    def get_document(self, document_id: str) -> Optional[Dict]:
        """Retrieve a document by ID"""
        try:
            document = self.documents.find_one({"document_id": document_id})
            
            if document:
                # Retrieve content from GridFS if needed
                if document.get("content_stored_in") == "gridfs":
                    grid_file = self.fs.get(document["content_gridfs_id"])
                    document["content"] = grid_file.read().decode('utf-8')
                
                # Remove MongoDB's _id for cleaner output
                document.pop("_id", None)
                
            return document
            
        except Exception as e:
            logger.error(f"Failed to retrieve document: {e}")
            return None
    
    def get_document_chunks(self, document_id: str) -> List[Dict]:
        """Get all chunks for a document"""
        try:
            chunks = list(self.chunks.find(
                {"document_id": document_id},
                sort=[("chunk_index", ASCENDING)]
            ))
            
            # Clean up MongoDB IDs
            for chunk in chunks:
                chunk.pop("_id", None)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}")
            return []
    
    def search_documents(self, 
                        query: str,
                        filters: Optional[Dict] = None,
                        limit: int = 10) -> List[Dict]:
        """Search documents using text search and filters"""
        try:
            # Build search query
            search_query = {"$text": {"$search": query}}
            
            # Add filters if provided
            if filters:
                search_query.update(filters)
            
            # Execute search
            results = list(self.documents.find(
                search_query,
                {"score": {"$meta": "textScore"}}
            ).sort(
                [("score", {"$meta": "textScore"})]
            ).limit(limit))
            
            # Clean up results
            for doc in results:
                doc.pop("_id", None)
                if doc.get("content_stored_in") == "gridfs":
                    doc.pop("content_gridfs_id", None)  # Don't return GridFS ID
            
            logger.info(f"Search found {len(results)} documents for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def store_query_history(self,
                           query: str,
                           response: str,
                           metadata: Optional[Dict] = None) -> str:
        """Store query and response for analysis"""
        try:
            query_doc = {
                "query_id": f"query_{datetime.now().strftime('%Y%m%d%H%M%S')}_{hashlib.md5(query.encode()).hexdigest()[:8]}",
                "query": query,
                "response": response,
                "timestamp": datetime.utcnow(),
                "metadata": metadata or {},
                "feedback": None,
                "rating": None
            }
            
            result = self.queries.insert_one(query_doc)
            return query_doc["query_id"]
            
        except Exception as e:
            logger.error(f"Failed to store query history: {e}")
            return ""
    
    def add_feedback(self,
                    query_id: str,
                    rating: int,
                    comment: Optional[str] = None) -> bool:
        """Add user feedback for a query"""
        try:
            result = self.queries.update_one(
                {"query_id": query_id},
                {
                    "$set": {
                        "feedback": comment,
                        "rating": rating,
                        "feedback_timestamp": datetime.utcnow()
                    }
                }
            )
            
            # Also store in feedback collection for analysis
            self.feedback.insert_one({
                "query_id": query_id,
                "rating": rating,
                "comment": comment,
                "timestamp": datetime.utcnow()
            })
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {
                "total_documents": self.documents.count_documents({}),
                "total_chunks": self.chunks.count_documents({}),
                "total_queries": self.queries.count_documents({}),
                "total_feedback": self.feedback.count_documents({}),
                "documents_by_status": {},
                "storage_size": self.db.command("dbstats")["dataSize"],
                "recent_documents": [],
                "popular_queries": []
            }
            
            # Documents by status
            pipeline = [
                {"$group": {"_id": "$processing_status", "count": {"$sum": 1}}}
            ]
            for item in self.documents.aggregate(pipeline):
                stats["documents_by_status"][item["_id"]] = item["count"]
            
            # Recent documents
            recent = list(self.documents.find(
                {}, 
                {"document_id": 1, "file_name": 1, "created_at": 1}
            ).sort("created_at", DESCENDING).limit(5))
            
            for doc in recent:
                doc.pop("_id", None)
                stats["recent_documents"].append(doc)
            
            # Average ratings
            avg_rating = self.feedback.aggregate([
                {"$group": {"_id": None, "avg_rating": {"$avg": "$rating"}}}
            ])
            ratings = list(avg_rating)
            if ratings:
                stats["average_rating"] = ratings[0]["avg_rating"]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def cleanup_old_documents(self, days: int = 30) -> int:
        """Clean up old documents"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Find old documents
            old_docs = list(self.documents.find(
                {"created_at": {"$lt": cutoff_date}},
                {"document_id": 1, "content_gridfs_id": 1}
            ))
            
            if not old_docs:
                return 0
            
            # Delete GridFS files
            for doc in old_docs:
                if "content_gridfs_id" in doc:
                    try:
                        self.fs.delete(doc["content_gridfs_id"])
                    except:
                        pass
            
            # Delete documents and chunks
            doc_ids = [doc["document_id"] for doc in old_docs]
            
            self.chunks.delete_many({"document_id": {"$in": doc_ids}})
            result = self.documents.delete_many({"document_id": {"$in": doc_ids}})
            
            logger.info(f"Cleaned up {result.deleted_count} old documents")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0
    
    def export_collection(self, collection_name: str, output_file: str):
        """Export a collection to JSON"""
        try:
            collection = self.db[collection_name]
            documents = list(collection.find())
            
            # Convert ObjectId to string
            for doc in documents:
                if "_id" in doc:
                    doc["_id"] = str(doc["_id"])
            
            with open(output_file, 'w') as f:
                json.dump(documents, f, indent=2, default=str)
            
            logger.info(f"Exported {len(documents)} documents to {output_file}")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return 0
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


# Example usage
if __name__ == "__main__":
    from datetime import timedelta
    
    # Initialize MongoDB
    mongo = LegendaryMongoStore()
    
    # Test document storage
    test_content = """
    Patient presents with symptoms of Type 2 Diabetes including increased thirst,
    frequent urination, and unexplained weight loss. Blood glucose levels elevated.
    Recommended treatment includes metformin and lifestyle modifications.
    """
    
    doc_id = mongo.store_document(
        file_path="test_medical_record.txt",
        content=test_content,
        metadata={
            "type": "medical_record",
            "specialty": "endocrinology",
            "patient_id": "TEST123"
        }
    )
    
    print(f"Document stored with ID: {doc_id}")
    
    # Store chunks
    test_chunks = [
        {"text": "Patient presents with symptoms of Type 2 Diabetes", "embedding_id": "emb_1"},
        {"text": "including increased thirst, frequent urination", "embedding_id": "emb_2"},
        {"text": "Blood glucose levels elevated", "embedding_id": "emb_3"}
    ]
    
    chunks_stored = mongo.store_chunks(doc_id, test_chunks)
    print(f"Stored {chunks_stored} chunks")
    
    # Retrieve document
    doc = mongo.get_document(doc_id)
    if doc:
        print(f"Retrieved document: {doc['file_name']}")
    
    # Get statistics
    stats = mongo.get_statistics()
    print("\nDatabase Statistics:")
    for key, value in stats.items():
        if not isinstance(value, (dict, list)):
            print(f"  {key}: {value}")
    
    # Close connection
    mongo.close()