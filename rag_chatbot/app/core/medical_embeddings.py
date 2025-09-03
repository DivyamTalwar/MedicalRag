"""
LEGENDARY Medical Embeddings & Vector Store Configuration
Biomedical-optimized embeddings with production-ready vector database
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from enum import Enum
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import faiss
import qdrant_client
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, Range, MatchValue,
    SearchParams, HnswConfigDiff, OptimizersConfigDiff,
    CollectionInfo, UpdateStatus
)
import redis
from sklearn.preprocessing import normalize
import pickle
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel(Enum):
    PUBMEDBERT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    BIOBERT = "dmis-lab/biobert-v1.1"
    SCIBERT = "allenai/scibert_scivocab_uncased"
    BIOLINKBERT = "michiyasunaga/BioLinkBERT-large"
    BGE_M3 = "BAAI/bge-m3"
    MEDCPT = "ncbi/MedCPT-Query-Encoder"
    CLINICAL_BERT = "emilyalsentzer/Bio_ClinicalBERT"
    SAPBERT = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

@dataclass
class EmbeddingConfig:
    model_name: str
    dimension: int
    max_sequence_length: int
    batch_size: int
    normalize: bool
    pooling_strategy: str
    device: str
    cache_enabled: bool
    quantization_enabled: bool

@dataclass
class VectorSearchResult:
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None

class MedicalEmbeddingEngine:
    def __init__(self, 
                 model_type: EmbeddingModel = EmbeddingModel.PUBMEDBERT,
                 device: Optional[str] = None,
                 cache_enabled: bool = True,
                 quantization_enabled: bool = False):
        
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_enabled = cache_enabled
        self.quantization_enabled = quantization_enabled
        
        # Initialize models
        self._initialize_models()
        
        # Initialize cache
        if cache_enabled:
            self.cache = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=0,
                decode_responses=False
            )
        else:
            self.cache = None
        
        # Configuration
        self.config = self._get_model_config()
        
        logger.info(f"Initialized {model_type.name} on {self.device}")
    
    def _initialize_models(self):
        if self.model_type in [EmbeddingModel.BGE_M3]:
            self.model = SentenceTransformer(self.model_type.value, device=self.device)
            self.tokenizer = None
            self.transformer = None
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type.value)
            self.transformer = AutoModel.from_pretrained(self.model_type.value).to(self.device)
            self.model = None
        
        if self.quantization_enabled and self.device == 'cuda':
            self._apply_quantization()
    
    def _apply_quantization(self):
        if self.transformer:
            self.transformer = torch.quantization.quantize_dynamic(
                self.transformer,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            logger.info("Applied INT8 quantization to transformer model")
    
    def _get_model_config(self) -> EmbeddingConfig:
        configs = {
            EmbeddingModel.PUBMEDBERT: EmbeddingConfig(
                model_name=EmbeddingModel.PUBMEDBERT.value,
                dimension=768,
                max_sequence_length=512,
                batch_size=32,
                normalize=True,
                pooling_strategy='mean',
                device=self.device,
                cache_enabled=self.cache_enabled,
                quantization_enabled=self.quantization_enabled
            ),
            EmbeddingModel.BIOLINKBERT: EmbeddingConfig(
                model_name=EmbeddingModel.BIOLINKBERT.value,
                dimension=1024,
                max_sequence_length=512,
                batch_size=16,
                normalize=True,
                pooling_strategy='mean',
                device=self.device,
                cache_enabled=self.cache_enabled,
                quantization_enabled=self.quantization_enabled
            ),
            EmbeddingModel.BGE_M3: EmbeddingConfig(
                model_name=EmbeddingModel.BGE_M3.value,
                dimension=1024,
                max_sequence_length=8192,
                batch_size=32,
                normalize=True,
                pooling_strategy='cls',
                device=self.device,
                cache_enabled=self.cache_enabled,
                quantization_enabled=self.quantization_enabled
            ),
            EmbeddingModel.MEDCPT: EmbeddingConfig(
                model_name=EmbeddingModel.MEDCPT.value,
                dimension=768,
                max_sequence_length=512,
                batch_size=32,
                normalize=True,
                pooling_strategy='mean',
                device=self.device,
                cache_enabled=self.cache_enabled,
                quantization_enabled=self.quantization_enabled
            )
        }
        
        return configs.get(self.model_type, configs[EmbeddingModel.PUBMEDBERT])
    
    def _get_cache_key(self, text: str) -> str:
        return f"emb:{self.model_type.name}:{hashlib.md5(text.encode()).hexdigest()}"
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _cls_pooling(self, model_output):
        return model_output[0][:, 0]
    
    def encode_text(self, text: Union[str, List[str]], show_progress: bool = False) -> np.ndarray:
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        
        embeddings = []
        
        # Check cache first
        if self.cache_enabled and self.cache:
            cached_embeddings = []
            texts_to_encode = []
            
            for t in texts:
                cache_key = self._get_cache_key(t)
                try:
                    cached = self.cache.get(cache_key)
                    if cached:
                        cached_embeddings.append(pickle.loads(cached))
                    else:
                        texts_to_encode.append(t)
                        cached_embeddings.append(None)
                except:
                    texts_to_encode.append(t)
                    cached_embeddings.append(None)
        else:
            texts_to_encode = texts
            cached_embeddings = [None] * len(texts)
        
        # Encode non-cached texts
        if texts_to_encode:
            if self.model:  # SentenceTransformer
                new_embeddings = self.model.encode(
                    texts_to_encode,
                    batch_size=self.config.batch_size,
                    show_progress_bar=show_progress,
                    normalize_embeddings=self.config.normalize,
                    device=self.device
                )
            else:  # Custom transformer
                new_embeddings = self._encode_with_transformer(texts_to_encode)
            
            # Combine cached and new embeddings
            new_emb_idx = 0
            for i, cached_emb in enumerate(cached_embeddings):
                if cached_emb is None:
                    embedding = new_embeddings[new_emb_idx]
                    new_emb_idx += 1
                    
                    # Cache the new embedding
                    if self.cache_enabled and self.cache:
                        cache_key = self._get_cache_key(texts[i])
                        try:
                            self.cache.set(cache_key, pickle.dumps(embedding), ex=86400)  # 24h expiry
                        except:
                            pass
                    
                    embeddings.append(embedding)
                else:
                    embeddings.append(cached_emb)
        else:
            embeddings = cached_embeddings
        
        return np.array(embeddings) if len(embeddings) > 1 else embeddings[0]
    
    def _encode_with_transformer(self, texts: List[str]) -> np.ndarray:
        all_embeddings = []
        
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_sequence_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                model_output = self.transformer(**encoded_input)
            
            # Apply pooling
            if self.config.pooling_strategy == 'mean':
                embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            elif self.config.pooling_strategy == 'cls':
                embeddings = self._cls_pooling(model_output)
            else:
                embeddings = model_output.last_hidden_state.mean(dim=1)
            
            # Normalize if needed
            if self.config.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_query(self, query: str, add_medical_context: bool = True) -> np.ndarray:
        if add_medical_context:
            # Add medical query prefix for better retrieval
            prefixes = {
                EmbeddingModel.MEDCPT: "Query: ",
                EmbeddingModel.PUBMEDBERT: "Medical question: ",
                EmbeddingModel.BGE_M3: "Represent this medical query for retrieval: "
            }
            prefix = prefixes.get(self.model_type, "")
            query = prefix + query
        
        return self.encode_text(query)
    
    def encode_document(self, document: str, add_context: bool = True) -> np.ndarray:
        if add_context:
            # Add document prefix for better representation
            prefixes = {
                EmbeddingModel.MEDCPT: "Document: ",
                EmbeddingModel.PUBMEDBERT: "Medical document: ",
                EmbeddingModel.BGE_M3: "Represent this medical document for retrieval: "
            }
            prefix = prefixes.get(self.model_type, "")
            document = prefix + document
        
        return self.encode_text(document)

class QdrantVectorStore:
    def __init__(self,
                 collection_name: str,
                 embedding_engine: MedicalEmbeddingEngine,
                 host: str = "localhost",
                 port: int = 6333,
                 https: bool = False,
                 api_key: Optional[str] = None):
        
        self.collection_name = collection_name
        self.embedding_engine = embedding_engine
        
        # Initialize Qdrant client
        self.client = qdrant_client.QdrantClient(
            host=host,
            port=port,
            https=https,
            api_key=api_key
        )
        
        # Initialize collection
        self._initialize_collection()
    
    def _initialize_collection(self):
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} exists with {collection_info.points_count} points")
        except:
            # Create collection with optimized settings
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_engine.config.dimension,
                    distance=Distance.COSINE
                ),
                hnsw_config=HnswConfigDiff(
                    m=48,
                    ef_construct=256,
                    full_scan_threshold=10000
                ),
                optimizers_config=OptimizersConfigDiff(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1000,
                    default_segment_number=8,
                    memmap_threshold=50000,
                    indexing_threshold=20000
                ),
                on_disk_payload=False  # Keep payload in memory for speed
            )
            logger.info(f"Created collection {self.collection_name}")
    
    def upsert_documents(self, 
                        documents: List[Dict[str, Any]],
                        batch_size: int = 100,
                        parallel: bool = True) -> bool:
        
        total = len(documents)
        logger.info(f"Upserting {total} documents to {self.collection_name}")
        
        for i in range(0, total, batch_size):
            batch = documents[i:i + batch_size]
            
            # Prepare points
            points = []
            texts_to_encode = []
            
            for doc in batch:
                doc_id = doc.get('id', hashlib.md5(doc['content'].encode()).hexdigest())
                texts_to_encode.append(doc['content'])
            
            # Batch encode
            embeddings = self.embedding_engine.encode_document(texts_to_encode)
            if len(texts_to_encode) == 1:
                embeddings = [embeddings]
            
            for j, doc in enumerate(batch):
                doc_id = doc.get('id', hashlib.md5(doc['content'].encode()).hexdigest())
                
                point = PointStruct(
                    id=doc_id,
                    vector=embeddings[j].tolist(),
                    payload={
                        'content': doc['content'],
                        'metadata': doc.get('metadata', {}),
                        'timestamp': time.time()
                    }
                )
                points.append(point)
            
            # Upsert batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            
            logger.info(f"Upserted batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
        
        return True
    
    def search(self,
              query: str,
              top_k: int = 10,
              filters: Optional[Dict[str, Any]] = None,
              score_threshold: Optional[float] = None) -> List[VectorSearchResult]:
        
        # Encode query
        query_vector = self.embedding_engine.encode_query(query)
        
        # Build filter
        search_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, dict):
                    if 'gte' in value and 'lte' in value:
                        conditions.append(
                            FieldCondition(
                                key=f"metadata.{key}",
                                range=Range(gte=value['gte'], lte=value['lte'])
                            )
                        )
                else:
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
            
            if conditions:
                search_filter = Filter(must=conditions)
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
            with_vectors=False,
            search_params=SearchParams(
                hnsw_ef=256,
                exact=False
            ),
            score_threshold=score_threshold
        )
        
        # Convert to VectorSearchResult
        search_results = []
        for result in results:
            search_results.append(VectorSearchResult(
                id=str(result.id),
                score=result.score,
                content=result.payload.get('content', ''),
                metadata=result.payload.get('metadata', {})
            ))
        
        return search_results
    
    def hybrid_search(self,
                     query: str,
                     sparse_query: Optional[Dict[str, float]] = None,
                     top_k: int = 10,
                     alpha: float = 0.7) -> List[VectorSearchResult]:
        
        # Dense search
        dense_results = self.search(query, top_k=top_k * 2)
        
        if not sparse_query:
            return dense_results[:top_k]
        
        # Combine scores (alpha * dense + (1-alpha) * sparse)
        combined_scores = {}
        
        for result in dense_results:
            combined_scores[result.id] = alpha * result.score
        
        # Add sparse scores
        for doc_id, sparse_score in sparse_query.items():
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - alpha) * sparse_score
            else:
                combined_scores[doc_id] = (1 - alpha) * sparse_score
        
        # Sort by combined score
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Fetch documents for top results
        final_results = []
        for doc_id, score in sorted_ids[:top_k]:
            # Find in dense results or fetch from store
            for result in dense_results:
                if result.id == doc_id:
                    result.score = score
                    final_results.append(result)
                    break
        
        return final_results

class FAISSVectorStore:
    def __init__(self,
                 embedding_engine: MedicalEmbeddingEngine,
                 index_type: str = "IVF_HNSW",
                 nlist: int = 1000,
                 nprobe: int = 50):
        
        self.embedding_engine = embedding_engine
        self.dimension = embedding_engine.config.dimension
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        
        # Initialize index
        self._initialize_index()
        
        # Document storage
        self.documents = {}
        self.id_to_idx = {}
        self.idx_to_id = {}
    
    def _initialize_index(self):
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IVF_Flat":
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == "IVF_HNSW":
            quantizer = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Enable GPU if available
        if faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
            logger.info("FAISS index moved to GPU")
    
    def train_index(self, training_vectors: np.ndarray):
        if hasattr(self.index, 'train'):
            if self.embedding_engine.config.normalize:
                training_vectors = normalize(training_vectors, norm='l2')
            self.index.train(training_vectors)
            logger.info(f"Trained index with {len(training_vectors)} vectors")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        embeddings = []
        
        for doc in documents:
            doc_id = doc.get('id', hashlib.md5(doc['content'].encode()).hexdigest())
            
            # Check if document already exists
            if doc_id in self.id_to_idx:
                continue
            
            # Encode document
            embedding = self.embedding_engine.encode_document(doc['content'])
            
            if self.embedding_engine.config.normalize:
                embedding = normalize(embedding.reshape(1, -1), norm='l2')[0]
            
            embeddings.append(embedding)
            
            # Store document
            idx = len(self.documents)
            self.documents[doc_id] = doc
            self.id_to_idx[doc_id] = idx
            self.idx_to_id[idx] = doc_id
        
        if embeddings:
            embeddings = np.array(embeddings).astype('float32')
            self.index.add(embeddings)
            logger.info(f"Added {len(embeddings)} documents to index")
    
    def search(self, query: str, top_k: int = 10) -> List[VectorSearchResult]:
        # Encode query
        query_vector = self.embedding_engine.encode_query(query)
        
        if self.embedding_engine.config.normalize:
            query_vector = normalize(query_vector.reshape(1, -1), norm='l2')
        else:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = query_vector.astype('float32')
        
        # Set search parameters
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        # Search
        scores, indices = self.index.search(query_vector, top_k)
        
        # Convert to results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx in self.idx_to_id:
                doc_id = self.idx_to_id[idx]
                doc = self.documents[doc_id]
                results.append(VectorSearchResult(
                    id=doc_id,
                    score=float(score),
                    content=doc['content'],
                    metadata=doc.get('metadata', {})
                ))
        
        return results
    
    def save_index(self, path: str):
        faiss.write_index(self.index, f"{path}.index")
        
        with open(f"{path}.docs", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id
            }, f)
        
        logger.info(f"Saved index to {path}")
    
    def load_index(self, path: str):
        self.index = faiss.read_index(f"{path}.index")
        
        with open(f"{path}.docs", 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.id_to_idx = data['id_to_idx']
            self.idx_to_id = data['idx_to_id']
        
        logger.info(f"Loaded index from {path}")

class HybridVectorStore:
    def __init__(self,
                 dense_store: Union[QdrantVectorStore, FAISSVectorStore],
                 enable_cache: bool = True):
        
        self.dense_store = dense_store
        self.enable_cache = enable_cache
        
        if enable_cache:
            self.cache = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=1,
                decode_responses=False
            )
        else:
            self.cache = None
    
    def search(self,
              query: str,
              top_k: int = 10,
              use_reranking: bool = True,
              filters: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        
        # Check cache
        cache_key = f"search:{hashlib.md5(f'{query}:{top_k}:{filters}'.encode()).hexdigest()}"
        
        if self.enable_cache and self.cache:
            try:
                cached = self.cache.get(cache_key)
                if cached:
                    return pickle.loads(cached)
            except:
                pass
        
        # Perform search
        if hasattr(self.dense_store, 'search'):
            if isinstance(self.dense_store, QdrantVectorStore):
                results = self.dense_store.search(query, top_k, filters)
            else:
                results = self.dense_store.search(query, top_k)
        else:
            results = []
        
        # Cache results
        if self.enable_cache and self.cache and results:
            try:
                self.cache.set(cache_key, pickle.dumps(results), ex=3600)  # 1 hour expiry
            except:
                pass
        
        return results

if __name__ == "__main__":
    # Example usage
    
    # Initialize embedding engine
    embedding_engine = MedicalEmbeddingEngine(
        model_type=EmbeddingModel.PUBMEDBERT,
        cache_enabled=True,
        quantization_enabled=True
    )
    
    # Initialize Qdrant store
    qdrant_store = QdrantVectorStore(
        collection_name="medical_documents",
        embedding_engine=embedding_engine
    )
    
    # Sample documents
    sample_docs = [
        {
            'id': 'doc1',
            'content': 'Patient presents with chest pain and shortness of breath. ECG shows ST elevation.',
            'metadata': {'type': 'clinical_note', 'department': 'emergency'}
        },
        {
            'id': 'doc2',
            'content': 'Lab results: Troponin I: 2.5 ng/mL (elevated), CK-MB: 45 U/L',
            'metadata': {'type': 'lab_report', 'critical': True}
        }
    ]
    
    # Upsert documents
    qdrant_store.upsert_documents(sample_docs)
    
    # Search
    results = qdrant_store.search(
        query="What are the cardiac biomarkers?",
        top_k=5
    )
    
    for result in results:
        print(f"Score: {result.score:.3f} | Content: {result.content[:100]}...")
    
    # Initialize FAISS store
    faiss_store = FAISSVectorStore(
        embedding_engine=embedding_engine,
        index_type="HNSW"
    )
    
    # Add documents to FAISS
    faiss_store.add_documents(sample_docs)
    
    # Search with FAISS
    faiss_results = faiss_store.search("cardiac enzymes", top_k=5)
    
    print("\nFAISS Results:")
    for result in faiss_results:
        print(f"Score: {result.score:.3f} | Content: {result.content[:100]}...")