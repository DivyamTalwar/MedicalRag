"""
LEGENDARY OMEGA LLM & CUSTOM EMBEDDINGS ADAPTER
The ultimate integration for your custom API endpoints
"""

import os
import json
import time
import hashlib
import pickle
import logging
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from functools import lru_cache
from datetime import datetime

import requests
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OmegaLLM:
    """
    LEGENDARY Omega LLM Implementation
    Integrates with your custom LLM endpoint
    """
    
    def __init__(self):
        """Initialize Omega LLM with environment configuration"""
        self.base_url = os.getenv("LLM_ENDPOINT", "https://api.us.inc/omega/civie/v1/chat/completions")
        self.api_key = os.getenv("MODELS_API_KEY")
        self.model_name = os.getenv("OMEGA_MODEL_NAME", "omega-medical-v1")
        self.max_tokens = int(os.getenv("OMEGA_MAX_TOKENS", 4096))
        self.temperature = float(os.getenv("OMEGA_TEMPERATURE", 0.1))
        self.top_p = float(os.getenv("OMEGA_TOP_P", 0.95))
        self.timeout = int(os.getenv("OMEGA_TIMEOUT", 120))
        
        # Request session with connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        logger.info(f"Initialized Omega LLM: {self.model_name}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _make_request(self, messages: List[Dict], **kwargs) -> Dict:
        """Make API request with retry logic"""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
            "stream": False
        }
        
        # Add any additional parameters
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        
        try:
            response = self.session.post(
                self.base_url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Omega LLM request failed: {str(e)}")
            raise
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate response from Omega LLM"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            result = self._make_request(messages, **kwargs)
            
            # Extract response based on API format
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice:
                    return choice["message"].get("content", "")
                elif "text" in choice:
                    return choice["text"]
            elif "content" in result:
                return result["content"]
            elif "response" in result:
                return result["response"]
            else:
                logger.warning(f"Unknown response format: {result.keys()}")
                return str(result)
                
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat with conversation history"""
        try:
            result = self._make_request(messages, **kwargs)
            
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                if "message" in choice:
                    return choice["message"].get("content", "")
                elif "text" in choice:
                    return choice["text"]
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Chat failed: {str(e)}")
            raise
    
    async def agenerate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Async generation for better performance"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.generate, 
            prompt, 
            system_prompt, 
            kwargs
        )


class CustomEmbeddings:
    """
    LEGENDARY Custom Embeddings Implementation
    With caching, batching, and performance optimization
    """
    
    def __init__(self, use_cache: bool = True):
        """Initialize custom embeddings"""
        self.base_url = os.getenv("EMBEDDING_API_URL", "https://api.us.inc/usf/v1/embed/embeddings")
        self.api_key = os.getenv("MODELS_API_KEY")
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME", "medical-embeddings-v1")
        self.dimension = int(os.getenv("EMBEDDING_DIMENSION", 768))
        self.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", 100))
        self.timeout = int(os.getenv("EMBEDDING_TIMEOUT", 60))
        
        # Caching configuration
        self.use_cache = use_cache
        self.cache_dir = Path("./embedding_cache")
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Request session
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Performance tracking
        self.total_embeddings = 0
        self.cache_hits = 0
        
        logger.info(f"Initialized Custom Embeddings: dim={self.dimension}, batch={self.batch_size}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        content = f"{self.model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """Retrieve embedding from cache"""
        if not self.use_cache:
            return None
        
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.cache_hits += 1
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
                return None
        return None
    
    def _save_to_cache(self, text: str, embedding: List[float]):
        """Save embedding to cache"""
        if not self.use_cache:
            return
        
        cache_key = self._get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _make_request(self, texts: List[str]) -> List[List[float]]:
        """Make embedding API request"""
        embeddings = []
        
        # Process each text individually since API expects single input
        for text in texts:
            payload = {
                "input": text  # Your API expects single string in "input" field
            }
            
            try:
                response = self.session.post(
                    self.base_url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Parse response - your API returns nested structure
                if "result" in result and "data" in result["result"]:
                    # Extract embedding from result.data[0].embedding
                    embedding = result["result"]["data"][0]["embedding"]
                    embeddings.append(embedding)
                else:
                    logger.warning(f"Unknown embedding response format: {result}")
                    raise ValueError(f"Unknown embedding response format")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Embedding request failed for text: {str(e)}")
                raise
        
        return embeddings
    
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text"""
        # Check cache first
        cached = self._get_from_cache(text)
        if cached is not None:
            return cached
        
        # Get embedding from API
        embeddings = self._make_request([text])
        embedding = embeddings[0]
        
        # Save to cache
        self._save_to_cache(text, embedding)
        self.total_embeddings += 1
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts with batching and caching"""
        embeddings = []
        texts_to_embed = []
        text_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                embeddings.append((i, cached))
            else:
                texts_to_embed.append(text)
                text_indices.append(i)
        
        # Process uncached texts in batches
        if texts_to_embed:
            for batch_start in range(0, len(texts_to_embed), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(texts_to_embed))
                batch_texts = texts_to_embed[batch_start:batch_end]
                batch_indices = text_indices[batch_start:batch_end]
                
                # Get embeddings from API
                batch_embeddings = self._make_request(batch_texts)
                
                # Save to cache and collect results
                for text, embedding, idx in zip(batch_texts, batch_embeddings, batch_indices):
                    self._save_to_cache(text, embedding)
                    embeddings.append((idx, embedding))
                    self.total_embeddings += 1
        
        # Sort by original index
        embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in embeddings]
    
    async def aembed_batch(self, texts: List[str]) -> List[List[float]]:
        """Async batch embedding"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_batch, texts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics"""
        cache_ratio = (self.cache_hits / max(1, self.total_embeddings + self.cache_hits)) * 100
        return {
            "total_embeddings": self.total_embeddings,
            "cache_hits": self.cache_hits,
            "cache_hit_ratio": f"{cache_ratio:.2f}%",
            "dimension": self.dimension,
            "batch_size": self.batch_size
        }
    
    def clear_cache(self):
        """Clear embedding cache"""
        if self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Embedding cache cleared")


class PineconeVectorStore:
    """
    LEGENDARY Pinecone Integration
    Optimized for medical document retrieval
    """
    
    def __init__(self, embeddings: CustomEmbeddings):
        """Initialize Pinecone vector store"""
        from pinecone import Pinecone, ServerlessSpec
        
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "legendary-medical-rag")
        self.dimension = int(os.getenv("PINECONE_DIMENSION", 768))
        
        self.embeddings = embeddings
        
        # Initialize Pinecone with new API
        self.pc = Pinecone(
            api_key=self.api_key
        )
        
        # Check if index exists
        existing_indexes = self.pc.list_indexes().names()
        
        # Create index if it doesn't exist
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
            time.sleep(10)  # Wait for index to be ready
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
        
        # Get index stats
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            logger.info(f"Index stats: {stats}")
        except Exception as e:
            logger.warning(f"Could not get index stats: {e}")
    
    def upsert_documents(self, documents: List[Dict], batch_size: int = 100):
        """Upload documents to Pinecone"""
        total_uploaded = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Prepare texts and metadata
            texts = [doc["text"] for doc in batch]
            metadatas = [doc.get("metadata", {}) for doc in batch]
            
            # Get embeddings
            embeddings = self.embeddings.embed_batch(texts)
            
            # Prepare vectors for Pinecone
            vectors = []
            for j, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
                vector_id = f"doc_{i+j}_{hashlib.md5(text.encode()).hexdigest()[:8]}"
                
                # Add text to metadata
                metadata["text"] = text[:1000]  # Store first 1000 chars in metadata
                metadata["full_text_hash"] = hashlib.md5(text.encode()).hexdigest()
                
                vectors.append({
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                })
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors)
            total_uploaded += len(vectors)
            
            logger.info(f"Uploaded {total_uploaded}/{len(documents)} documents to Pinecone")
        
        return total_uploaded
    
    def search(self, query: str, top_k: int = 10, filter: Optional[Dict] = None) -> List[Dict]:
        """Search for similar documents"""
        # Get query embedding
        query_embedding = self.embeddings.embed_text(query)
        
        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata
            })
        
        return formatted_results
    
    def delete_all(self):
        """Delete all vectors from index"""
        self.index.delete(delete_all=True)
        logger.info(f"Deleted all vectors from {self.index_name}")
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        return self.index.describe_index_stats()


class RerankerModel:
    """
    LEGENDARY Reranker Model
    Improves retrieval accuracy by reranking results
    """
    
    def __init__(self):
        """Initialize reranker with API configuration"""
        self.base_url = "https://api.us.inc/usf-shiprocket/v1/embed/reranker"
        self.api_key = os.getenv("MODELS_API_KEY")
        self.model_name = "shunya-rerank"
        self.timeout = 30
        
        # Request session
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        logger.info(f"Initialized Reranker: {self.model_name}")
    
    def rerank(self, query: str, texts: List[str], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Rerank texts based on relevance to query
        
        Args:
            query: The search query
            texts: List of texts to rerank
            top_k: Return only top K results (None = return all)
        
        Returns:
            List of dicts with 'text', 'score', and 'index'
        """
        if not texts:
            return []
        
        payload = {
            "model": self.model_name,
            "query": query,
            "texts": texts
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
        
        try:
            response = self.session.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Parse response
            if "result" in result and "data" in result["result"]:
                reranked = result["result"]["data"]
                # Sort by score descending
                reranked.sort(key=lambda x: x["score"], reverse=True)
                
                # Limit to top_k if specified
                if top_k:
                    reranked = reranked[:top_k]
                
                logger.info(f"Reranked {len(texts)} texts, returning top {len(reranked)}")
                return reranked
            else:
                logger.warning(f"Unknown reranker response format: {result}")
                # Return original order if reranking fails
                return [{"text": text, "score": 1.0 / (i + 1), "index": i} 
                       for i, text in enumerate(texts)]
                
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
            # Return original order if reranking fails
            return [{"text": text, "score": 1.0 / (i + 1), "index": i} 
                   for i, text in enumerate(texts)]


class LegendaryRAGPipeline:
    """
    The Ultimate RAG Pipeline
    Combining Omega LLM, Custom Embeddings, and Pinecone
    """
    
    def __init__(self):
        """Initialize the legendary pipeline"""
        logger.info("Initializing LEGENDARY RAG Pipeline...")
        
        # Initialize components
        self.llm = OmegaLLM()
        self.embeddings = CustomEmbeddings(use_cache=True)
        self.vector_store = PineconeVectorStore(self.embeddings)
        self.reranker = RerankerModel()  # NEW: Reranker for better accuracy
        
        # Configuration
        self.retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", 20))
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", 5))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.75))
        self.use_reranker = os.getenv("USE_RERANKER", "true").lower() == "true"
        
        logger.info("LEGENDARY RAG Pipeline initialized successfully with Reranker!")
    
    def ingest_documents(self, file_paths: List[str]):
        """Ingest documents into the system"""
        documents = []
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create document chunks (simplified for demonstration)
                chunks = self._chunk_text(content)
                
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "metadata": {
                            "source": file_path,
                            "chunk_index": i,
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                
                logger.info(f"Processed {file_path}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
        
        # Upload to Pinecone
        if documents:
            count = self.vector_store.upsert_documents(documents)
            logger.info(f"Successfully uploaded {count} document chunks to Pinecone")
        
        return len(documents)
    
    def _chunk_text(self, text: str, max_chunk_size: int = 512) -> List[str]:
        """Simple text chunking"""
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1
            if current_size + word_size > max_chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def query(self, question: str, use_context: bool = True) -> Dict[str, Any]:
        """Query the RAG system with reranking for better accuracy"""
        start_time = time.time()
        
        # Search for relevant documents
        search_start = time.time()
        search_results = self.vector_store.search(
            query=question,
            top_k=self.retrieval_top_k
        )
        search_time = time.time() - search_start
        
        # Filter by similarity threshold
        relevant_docs = [
            doc for doc in search_results 
            if doc["score"] >= self.similarity_threshold
        ]
        
        # Apply reranking if enabled and we have documents
        rerank_time = 0
        if self.use_reranker and relevant_docs:
            rerank_start = time.time()
            
            # Extract texts for reranking
            texts_to_rerank = [doc["text"] for doc in relevant_docs]
            
            # Rerank the documents
            reranked = self.reranker.rerank(
                query=question,
                texts=texts_to_rerank,
                top_k=self.rerank_top_k
            )
            
            # Reorganize documents based on reranking scores
            if reranked:
                reranked_docs = []
                for item in reranked:
                    original_idx = item["index"]
                    if original_idx < len(relevant_docs):
                        doc = relevant_docs[original_idx].copy()
                        doc["rerank_score"] = item["score"]
                        reranked_docs.append(doc)
                
                relevant_docs = reranked_docs
                logger.info(f"Reranked {len(texts_to_rerank)} docs to top {len(reranked_docs)}")
            
            rerank_time = time.time() - rerank_start
        
        # Prepare context
        if use_context and relevant_docs:
            # Use reranked order for context
            context_docs = relevant_docs[:self.rerank_top_k]
            context = "\n\n".join([
                f"Document {i+1}:\n{doc['text']}"
                for i, doc in enumerate(context_docs)
            ])
            
            prompt = f"""Based on the following context, please answer the question accurately and comprehensively.

Context:
{context}

Question: {question}

Answer:"""
        else:
            prompt = question
            context_docs = []
        
        # Generate response
        gen_start = time.time()
        response = self.llm.generate(
            prompt=prompt,
            system_prompt="You are a medical expert assistant. Provide accurate, detailed, and helpful responses based on the given context."
        )
        gen_time = time.time() - gen_start
        
        # Prepare result with timing breakdown
        result = {
            "question": question,
            "answer": response,
            "sources": [
                {
                    "text": doc["text"][:200] + "...",
                    "score": doc.get("score", 0),
                    "rerank_score": doc.get("rerank_score", None),
                    "metadata": doc.get("metadata", {})
                }
                for doc in context_docs
            ],
            "processing_time": time.time() - start_time,
            "timing_breakdown": {
                "search_time": search_time,
                "rerank_time": rerank_time,
                "generation_time": gen_time
            },
            "total_documents_found": len(search_results),
            "relevant_documents": len(relevant_docs),
            "reranking_applied": self.use_reranker and len(relevant_docs) > 0
        }
        
        return result
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries efficiently"""
        results = []
        
        for question in questions:
            try:
                result = self.query(question)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process question: {question}, Error: {e}")
                results.append({
                    "question": question,
                    "answer": f"Error: {str(e)}",
                    "sources": [],
                    "error": True
                })
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "llm_model": self.llm.model_name,
            "embedding_stats": self.embeddings.get_stats(),
            "vector_store_stats": self.vector_store.get_stats(),
            "configuration": {
                "retrieval_top_k": self.retrieval_top_k,
                "rerank_top_k": self.rerank_top_k,
                "similarity_threshold": self.similarity_threshold
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Test the pipeline
    pipeline = LegendaryRAGPipeline()
    
    # Test query
    test_question = "What are the symptoms of diabetes?"
    result = pipeline.query(test_question)
    
    print("=" * 80)
    print("LEGENDARY RAG PIPELINE TEST")
    print("=" * 80)
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer'][:500]}...")
    print(f"Processing time: {result['processing_time']:.2f}s")
    print(f"Documents found: {result['total_documents_found']}")
    print(f"Relevant documents: {result['relevant_documents']}")
    print("=" * 80)
    
    # Get stats
    stats = pipeline.get_pipeline_stats()
    print("\nPipeline Statistics:")
    print(json.dumps(stats, indent=2))