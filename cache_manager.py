#!/usr/bin/env python3

import time
import hashlib
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
from functools import lru_cache
import redis
import threading
from datetime import datetime, timedelta

@dataclass
class CacheEntry:
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = None
    ttl: int = 3600

class LRUCache:
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self.cache = OrderedDict()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                entry = self.cache[key]
                
                # Update metadata
                entry.access_count += 1
                entry.last_accessed = time.time()
                
                self.stats['hits'] += 1
                
                # Check TTL
                if time.time() - entry.timestamp > entry.ttl:
                    del self.cache[key]
                    self.stats['misses'] += 1
                    return None
                
                return entry.value
            
            self.stats['misses'] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.maxsize:
                # Remove least recently used
                self.cache.popitem(last=False)
                self.stats['evictions'] += 1
            
            # Add new entry
            self.cache[key] = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                last_accessed=time.time(),
                ttl=ttl
            )
    
    def clear(self):
        
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'maxsize': self.maxsize,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'hit_rate': hit_rate
            }

class SemanticCache:

    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
        self.cache = {}
        self.embeddings = {}
        self.stats = {
            'exact_hits': 0,
            'semantic_hits': 0,
            'misses': 0
        }
        self.lock = threading.RLock()
    
    def _compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get(self, query: str, query_embedding: List[float]) -> Optional[Any]:
        
        with self.lock:
            # Check exact match first
            query_hash = hashlib.md5(query.encode()).hexdigest()
            if query_hash in self.cache:
                entry = self.cache[query_hash]
                
                # Check TTL
                if time.time() - entry.timestamp > entry.ttl:
                    del self.cache[query_hash]
                    del self.embeddings[query_hash]
                    self.stats['misses'] += 1
                    return None
                
                self.stats['exact_hits'] += 1
                entry.access_count += 1
                entry.last_accessed = time.time()
                return entry.value
            
            # Check semantic similarity
            best_match = None
            best_similarity = 0.0
            
            for cached_hash, cached_embedding in self.embeddings.items():
                similarity = self._compute_similarity(query_embedding, cached_embedding)
                
                if similarity > self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cached_hash
            
            if best_match:
                entry = self.cache[best_match]
                
                # Check TTL
                if time.time() - entry.timestamp > entry.ttl:
                    del self.cache[best_match]
                    del self.embeddings[best_match]
                    self.stats['misses'] += 1
                    return None
                
                self.stats['semantic_hits'] += 1
                entry.access_count += 1
                entry.last_accessed = time.time()
                return entry.value
            
            self.stats['misses'] += 1
            return None
    
    def set(self, query: str, query_embedding: List[float], value: Any, ttl: int = 3600):
        
        with self.lock:
            query_hash = hashlib.md5(query.encode()).hexdigest()
            
            self.cache[query_hash] = CacheEntry(
                key=query,
                value=value,
                timestamp=time.time(),
                last_accessed=time.time(),
                ttl=ttl
            )
            
            self.embeddings[query_hash] = query_embedding
    
    def get_stats(self) -> Dict[str, Any]:
        
        with self.lock:
            total_hits = self.stats['exact_hits'] + self.stats['semantic_hits']
            total_requests = total_hits + self.stats['misses']
            hit_rate = total_hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'exact_hits': self.stats['exact_hits'],
                'semantic_hits': self.stats['semantic_hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'semantic_hit_ratio': self.stats['semantic_hits'] / total_hits if total_hits > 0 else 0
            }

class PrecomputedContextCache:

    def __init__(self):
        self.cache = {}
        self.patterns = self._initialize_patterns()
        self.stats = {
            'hits': 0,
            'misses': 0,
            'patterns_matched': {}
        }
        self.lock = threading.RLock()
    
    def _initialize_patterns(self) -> Dict[str, str]:
        
        return {
            'diagnosis': r'(?i)what.*(diagnosis|condition|disease)',
            'treatment': r'(?i)what.*(treatment|medication|therapy|drug)',
            'symptoms': r'(?i)what.*(symptom|sign|present|complain)',
            'test_results': r'(?i)what.*(test|result|lab|finding)',
            'procedure': r'(?i)what.*(procedure|surgery|operation)',
            'side_effects': r'(?i)what.*(side effect|adverse|reaction)',
            'dosage': r'(?i)what.*(dose|dosage|how much)',
            'contraindication': r'(?i)what.*(contraindication|avoid|not take)',
            'prognosis': r'(?i)what.*(prognosis|outcome|survival)',
            'prevention': r'(?i)how.*(prevent|avoid|reduce risk)'
        }
    
    def _detect_pattern(self, query: str) -> Optional[str]:
        
        import re
        
        for pattern_name, pattern_regex in self.patterns.items():
            if re.match(pattern_regex, query):
                return pattern_name
        
        return None
    
    def get(self, query: str) -> Optional[Dict[str, Any]]:
        
        with self.lock:
            # Detect pattern
            pattern = self._detect_pattern(query)
            
            if pattern and pattern in self.cache:
                entry = self.cache[pattern]
                
                # Check TTL
                if time.time() - entry.timestamp > entry.ttl:
                    del self.cache[pattern]
                    self.stats['misses'] += 1
                    return None
                
                self.stats['hits'] += 1
                
                # Track pattern usage
                if pattern not in self.stats['patterns_matched']:
                    self.stats['patterns_matched'][pattern] = 0
                self.stats['patterns_matched'][pattern] += 1
                
                entry.access_count += 1
                entry.last_accessed = time.time()
                
                return entry.value
            
            self.stats['misses'] += 1
            return None
    
    def set(self, pattern: str, context: Dict[str, Any], ttl: int = 7200):
        
        with self.lock:
            self.cache[pattern] = CacheEntry(
                key=pattern,
                value=context,
                timestamp=time.time(),
                last_accessed=time.time(),
                ttl=ttl
            )
    
    def precompute_common_contexts(self, retriever, generator):
        
        common_queries = {
            'diagnosis': "What is the diagnosis?",
            'treatment': "What treatment is recommended?",
            'symptoms': "What are the symptoms?",
            'test_results': "What do the test results show?",
            'procedure': "What procedure was performed?"
        }
        
        for pattern, query in common_queries.items():
            # Retrieve and generate context
            results = retriever.retrieve(query)
            context = generator.build_context(results)
            
            # Cache the context
            self.set(pattern, {
                'query': query,
                'context': context,
                'chunks': results[:5]
            })
    
    def get_stats(self) -> Dict[str, Any]:
        
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'patterns_matched': self.stats['patterns_matched'],
                'patterns_available': list(self.patterns.keys())
            }

class CacheManager:

    def __init__(self):
        # Initialize all cache layers
        self.query_cache = LRUCache(maxsize=10000)           # Layer 1
        self.result_cache = SemanticCache(similarity_threshold=0.95)  # Layer 2
        self.context_cache = PrecomputedContextCache()       # Layer 3
        
        # Redis for distributed caching (optional)
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=1
            )
            self.redis_client.ping()
            self.redis_available = True
        except:
            self.redis_available = False
        
        # Global stats
        self.global_stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0,
            'response_times': []
        }
    
    def get_cached_embedding(self, query: str) -> Optional[List[float]]:
        
        return self.query_cache.get(query)
    
    def cache_embedding(self, query: str, embedding: List[float]):
        
        self.query_cache.set(query, embedding, ttl=3600)
    
    def get_cached_results(self, query: str, query_embedding: List[float]) -> Optional[List[Any]]:
        
        return self.result_cache.get(query, query_embedding)
    
    def cache_results(self, query: str, query_embedding: List[float], results: List[Any]):
        
        self.result_cache.set(query, query_embedding, results, ttl=3600)
    
    def get_cached_context(self, query: str) -> Optional[Dict[str, Any]]:
        
        return self.context_cache.get(query)
    
    def cache_context(self, pattern: str, context: Dict[str, Any]):
        
        self.context_cache.set(pattern, context, ttl=7200)
    
    def process_with_cache(self, query: str, retriever, embedder, generator) -> Tuple[Any, float]:
        
        start_time = time.time()
        self.global_stats['total_queries'] += 1
        
        # Layer 3: Check pre-computed context cache
        cached_context = self.get_cached_context(query)
        if cached_context:
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            self.global_stats['cache_hits'] += 1
            self._update_response_time(response_time)
            return cached_context, response_time
        
        # Layer 1: Check query embedding cache
        query_embedding = self.get_cached_embedding(query)
        if not query_embedding:
            query_embedding = embedder.create_embedding(query)
            self.cache_embedding(query, query_embedding)
        
        # Layer 2: Check result cache
        cached_results = self.get_cached_results(query, query_embedding)
        if cached_results:
            # Generate context from cached results
            context = generator.build_context(cached_results)
            response_time = (time.time() - start_time) * 1000
            self.global_stats['cache_hits'] += 1
            self._update_response_time(response_time)
            return {'context': context, 'results': cached_results}, response_time
        
        # Cache miss - perform full retrieval
        self.global_stats['cache_misses'] += 1
        
        # Retrieve and cache
        results = retriever.retrieve(query)
        self.cache_results(query, query_embedding, results)
        
        # Generate context
        context = generator.build_context(results)
        
        response_time = (time.time() - start_time) * 1000
        self._update_response_time(response_time)
        
        return {'context': context, 'results': results}, response_time
    
    def _update_response_time(self, response_time: float):
        
        self.global_stats['response_times'].append(response_time)
        
        # Keep only last 1000 response times
        if len(self.global_stats['response_times']) > 1000:
            self.global_stats['response_times'] = self.global_stats['response_times'][-1000:]
        
        # Update average
        self.global_stats['avg_response_time'] = np.mean(self.global_stats['response_times'])
    
    def get_all_stats(self) -> Dict[str, Any]:
        
        cache_hit_rate = (self.global_stats['cache_hits'] / 
                         self.global_stats['total_queries'] 
                         if self.global_stats['total_queries'] > 0 else 0)
        
        return {
            'global': {
                'total_queries': self.global_stats['total_queries'],
                'cache_hits': self.global_stats['cache_hits'],
                'cache_misses': self.global_stats['cache_misses'],
                'hit_rate': cache_hit_rate,
                'avg_response_time_ms': self.global_stats['avg_response_time'],
                'p95_response_time_ms': np.percentile(self.global_stats['response_times'], 95) 
                                       if self.global_stats['response_times'] else 0
            },
            'layer1_query_cache': self.query_cache.get_stats(),
            'layer2_result_cache': self.result_cache.get_stats(),
            'layer3_context_cache': self.context_cache.get_stats(),
            'redis_available': self.redis_available
        }
    
    def warm_cache(self, common_queries: List[str], retriever, embedder, generator):
        
        print("Warming up cache with common queries...")
        
        for query in common_queries:
            self.process_with_cache(query, retriever, embedder, generator)
        
        print(f"Cache warmed with {len(common_queries)} queries")
    
    def clear_all_caches(self):
        
        self.query_cache.clear()
        self.result_cache = SemanticCache(similarity_threshold=0.95)
        self.context_cache = PrecomputedContextCache()
        
        if self.redis_available:
            self.redis_client.flushdb()