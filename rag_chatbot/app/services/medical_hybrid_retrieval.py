"""
LEGENDARY Hybrid Medical Retrieval System
Advanced multi-modal retrieval with medical context awareness
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import logging
from datetime import datetime, timedelta
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import spacy
from sentence_transformers import CrossEncoder
import torch
from collections import defaultdict
import hashlib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    LAB_INTERPRETATION = "lab_interpretation"
    MEDICATION_INFO = "medication_info"
    SYMPTOM_ANALYSIS = "symptom_analysis"
    PROCEDURE_DETAILS = "procedure_details"
    PATIENT_HISTORY = "patient_history"
    GUIDELINE_SEARCH = "guideline_search"
    GENERAL = "general"

@dataclass
class RetrievalContext:
    query_type: QueryType
    medical_entities: List[Dict[str, Any]]
    temporal_context: Optional[Dict[str, Any]]
    patient_context: Optional[Dict[str, Any]]
    urgency_level: str
    required_expertise: List[str]
    confidence_threshold: float

@dataclass
class RetrievalResult:
    doc_id: str
    content: str
    score: float
    retrieval_method: str
    metadata: Dict[str, Any]
    relevance_explanation: str
    medical_entities: List[Dict[str, Any]]
    confidence: float
    citations: List[str] = field(default_factory=list)

class MedicalQueryAnalyzer:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_sci_md")
        except:
            self.nlp = spacy.load("en_core_web_sm")
        
        self.query_patterns = self._compile_query_patterns()
        self.urgency_keywords = {
            'critical': ['emergency', 'urgent', 'critical', 'severe', 'acute', 'immediately'],
            'high': ['abnormal', 'elevated', 'concerning', 'significant'],
            'moderate': ['mild', 'moderate', 'stable'],
            'low': ['routine', 'follow-up', 'check']
        }
    
    def _compile_query_patterns(self) -> Dict[QueryType, List[re.Pattern]]:
        return {
            QueryType.DIAGNOSIS: [
                re.compile(r'\b(?:diagnos[ei]|differential|rule out|suspect|assess)\b', re.I),
                re.compile(r'\b(?:what is|could it be|possible causes?)\b', re.I)
            ],
            QueryType.TREATMENT: [
                re.compile(r'\b(?:treat|therap|management|intervention|protocol)\b', re.I),
                re.compile(r'\b(?:how to manage|best practice|guideline)\b', re.I)
            ],
            QueryType.LAB_INTERPRETATION: [
                re.compile(r'\b(?:lab|test|result|value|range|normal|abnormal)\b', re.I),
                re.compile(r'\b(?:\d+\.?\d*)\s*(?:mg/dL|mmol/L|IU/mL|ng/mL)\b', re.I)
            ],
            QueryType.MEDICATION_INFO: [
                re.compile(r'\b(?:medicat|drug|dose|dosage|prescription|side effect)\b', re.I),
                re.compile(r'\b(?:mg|mcg|ml|tablet|capsule|injection)\b', re.I)
            ],
            QueryType.SYMPTOM_ANALYSIS: [
                re.compile(r'\b(?:symptom|pain|discomfort|feeling|experience)\b', re.I),
                re.compile(r'\b(?:chest|head|abdom|fever|cough|fatigue)\b', re.I)
            ],
            QueryType.PROCEDURE_DETAILS: [
                re.compile(r'\b(?:procedure|surgery|operation|intervention)\b', re.I),
                re.compile(r'\b(?:CT|MRI|ultrasound|biopsy|endoscopy)\b', re.I)
            ]
        }
    
    def analyze_query(self, query: str, chat_history: Optional[List[str]] = None) -> RetrievalContext:
        # Determine query type
        query_type = self._classify_query_type(query)
        
        # Extract medical entities
        medical_entities = self._extract_medical_entities(query)
        
        # Determine urgency
        urgency = self._assess_urgency(query)
        
        # Extract temporal context
        temporal_context = self._extract_temporal_context(query)
        
        # Determine required expertise
        expertise = self._determine_expertise(query_type, medical_entities)
        
        # Set confidence threshold based on query type and urgency
        confidence_threshold = self._calculate_confidence_threshold(query_type, urgency)
        
        return RetrievalContext(
            query_type=query_type,
            medical_entities=medical_entities,
            temporal_context=temporal_context,
            patient_context=self._extract_patient_context(query, chat_history),
            urgency_level=urgency,
            required_expertise=expertise,
            confidence_threshold=confidence_threshold
        )
    
    def _classify_query_type(self, query: str) -> QueryType:
        scores = {}
        
        for qtype, patterns in self.query_patterns.items():
            score = sum(1 for pattern in patterns if pattern.search(query))
            if score > 0:
                scores[qtype] = score
        
        if scores:
            return max(scores, key=scores.get)
        return QueryType.GENERAL
    
    def _extract_medical_entities(self, query: str) -> List[Dict[str, Any]]:
        entities = []
        doc = self.nlp(query)
        
        # NER entities
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        # Lab values
        lab_pattern = re.compile(r'(\w+)\s*[:=]\s*(\d+\.?\d*)\s*(mg/dL|mmol/L|IU/mL|ng/mL|%)')
        for match in lab_pattern.finditer(query):
            entities.append({
                'text': match.group(0),
                'type': 'LAB_VALUE',
                'test': match.group(1),
                'value': float(match.group(2)),
                'unit': match.group(3)
            })
        
        # Medications
        med_pattern = re.compile(r'(\w+)\s+(\d+)\s*(mg|mcg|ml|units?)', re.I)
        for match in med_pattern.finditer(query):
            entities.append({
                'text': match.group(0),
                'type': 'MEDICATION',
                'drug': match.group(1),
                'dose': match.group(2),
                'unit': match.group(3)
            })
        
        return entities
    
    def _assess_urgency(self, query: str) -> str:
        query_lower = query.lower()
        
        for level, keywords in self.urgency_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return level
        
        return 'moderate'
    
    def _extract_temporal_context(self, query: str) -> Optional[Dict[str, Any]]:
        temporal_patterns = {
            'recent': r'\b(?:recent|recently|new|newly)\b',
            'chronic': r'\b(?:chronic|long-term|persistent|ongoing)\b',
            'acute': r'\b(?:acute|sudden|abrupt)\b',
            'duration': r'\b(?:for|since|lasted?)\s+(\d+)\s*(days?|weeks?|months?|years?)\b'
        }
        
        context = {}
        for key, pattern in temporal_patterns.items():
            match = re.search(pattern, query, re.I)
            if match:
                context[key] = match.group(0) if not match.groups() else match.groups()
        
        return context if context else None
    
    def _extract_patient_context(self, query: str, chat_history: Optional[List[str]]) -> Optional[Dict[str, Any]]:
        context = {}
        
        # Age pattern
        age_match = re.search(r'\b(\d{1,3})\s*(?:year|yr|y/?o)\b', query, re.I)
        if age_match:
            context['age'] = int(age_match.group(1))
        
        # Gender pattern
        gender_match = re.search(r'\b(male|female|man|woman)\b', query, re.I)
        if gender_match:
            context['gender'] = gender_match.group(1).lower()
        
        # Extract from chat history
        if chat_history:
            for msg in chat_history[-5:]:  # Last 5 messages
                if 'patient' in msg.lower():
                    context['has_history'] = True
                    break
        
        return context if context else None
    
    def _determine_expertise(self, query_type: QueryType, entities: List[Dict[str, Any]]) -> List[str]:
        expertise = []
        
        # Query type based expertise
        type_expertise = {
            QueryType.DIAGNOSIS: ['internal_medicine', 'diagnostics'],
            QueryType.TREATMENT: ['therapeutics', 'clinical_guidelines'],
            QueryType.LAB_INTERPRETATION: ['laboratory_medicine', 'pathology'],
            QueryType.MEDICATION_INFO: ['pharmacology', 'clinical_pharmacy'],
            QueryType.PROCEDURE_DETAILS: ['surgery', 'interventional']
        }
        
        if query_type in type_expertise:
            expertise.extend(type_expertise[query_type])
        
        # Entity based expertise
        for entity in entities:
            if entity.get('type') == 'CARDIOLOGY':
                expertise.append('cardiology')
            elif entity.get('type') == 'NEUROLOGY':
                expertise.append('neurology')
        
        return list(set(expertise))
    
    def _calculate_confidence_threshold(self, query_type: QueryType, urgency: str) -> float:
        base_threshold = {
            QueryType.DIAGNOSIS: 0.8,
            QueryType.TREATMENT: 0.85,
            QueryType.LAB_INTERPRETATION: 0.75,
            QueryType.MEDICATION_INFO: 0.9,
            QueryType.PROCEDURE_DETAILS: 0.7,
            QueryType.GENERAL: 0.6
        }
        
        urgency_modifier = {
            'critical': 0.1,
            'high': 0.05,
            'moderate': 0,
            'low': -0.05
        }
        
        threshold = base_threshold.get(query_type, 0.7)
        threshold += urgency_modifier.get(urgency, 0)
        
        return min(0.95, max(0.5, threshold))

class HybridMedicalRetriever:
    def __init__(self,
                 dense_retriever,
                 sparse_retriever=None,
                 knowledge_graph=None,
                 reranker_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever or self._initialize_sparse_retriever()
        self.knowledge_graph = knowledge_graph or self._initialize_knowledge_graph()
        self.query_analyzer = MedicalQueryAnalyzer()
        
        # Initialize reranker
        self.reranker = CrossEncoder(reranker_model)
        
        # Cache for performance
        self.retrieval_cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    def _initialize_sparse_retriever(self):
        return BM25Retriever()
    
    def _initialize_knowledge_graph(self):
        return MedicalKnowledgeGraph()
    
    async def retrieve(self,
                       query: str,
                       top_k: int = 20,
                       context: Optional[RetrievalContext] = None,
                       filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        
        # Analyze query if context not provided
        if not context:
            context = self.query_analyzer.analyze_query(query)
        
        # Check cache
        cache_key = self._generate_cache_key(query, top_k, filters)
        if cache_key in self.retrieval_cache:
            cached_time, cached_results = self.retrieval_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                logger.info(f"Returning cached results for query: {query[:50]}...")
                return cached_results
        
        # Parallel retrieval from multiple sources
        retrieval_tasks = []
        
        # Dense retrieval
        retrieval_tasks.append(self._dense_retrieve(query, top_k * 2, context, filters))
        
        # Sparse retrieval
        if self.sparse_retriever:
            retrieval_tasks.append(self._sparse_retrieve(query, top_k * 2, context, filters))
        
        # Knowledge graph retrieval
        if self.knowledge_graph and context.medical_entities:
            retrieval_tasks.append(self._graph_retrieve(context.medical_entities, top_k))
        
        # Execute parallel retrieval
        all_results = await asyncio.gather(*retrieval_tasks)
        
        # Combine results
        combined_results = self._combine_results(all_results, context)
        
        # Rerank results
        reranked_results = self._rerank_results(query, combined_results, context)
        
        # Apply medical-specific filtering
        filtered_results = self._apply_medical_filters(reranked_results, context)
        
        # Final selection
        final_results = filtered_results[:top_k]
        
        # Cache results
        self.retrieval_cache[cache_key] = (datetime.now(), final_results)
        
        logger.info(f"Retrieved {len(final_results)} documents for query: {query[:50]}...")
        return final_results
    
    async def _dense_retrieve(self,
                             query: str,
                             top_k: int,
                             context: RetrievalContext,
                             filters: Optional[Dict[str, Any]]) -> List[RetrievalResult]:
        
        # Enhance query based on context
        enhanced_query = self._enhance_query(query, context)
        
        # Retrieve from dense index
        dense_results = await asyncio.to_thread(
            self.dense_retriever.search,
            enhanced_query,
            top_k,
            filters
        )
        
        # Convert to RetrievalResult
        results = []
        for result in dense_results:
            retrieval_result = RetrievalResult(
                doc_id=result.id,
                content=result.content,
                score=result.score,
                retrieval_method='dense',
                metadata=result.metadata,
                relevance_explanation=f"Semantic similarity: {result.score:.3f}",
                medical_entities=[],
                confidence=result.score
            )
            results.append(retrieval_result)
        
        return results
    
    async def _sparse_retrieve(self,
                              query: str,
                              top_k: int,
                              context: RetrievalContext,
                              filters: Optional[Dict[str, Any]]) -> List[RetrievalResult]:
        
        # Extract keywords for BM25
        keywords = self._extract_keywords(query, context)
        
        # Retrieve using BM25
        sparse_results = await asyncio.to_thread(
            self.sparse_retriever.search,
            keywords,
            top_k
        )
        
        # Convert to RetrievalResult
        results = []
        for doc_id, score, content in sparse_results:
            retrieval_result = RetrievalResult(
                doc_id=doc_id,
                content=content,
                score=score,
                retrieval_method='sparse',
                metadata={},
                relevance_explanation=f"Keyword match score: {score:.3f}",
                medical_entities=[],
                confidence=score / 100  # Normalize BM25 score
            )
            results.append(retrieval_result)
        
        return results
    
    async def _graph_retrieve(self,
                             entities: List[Dict[str, Any]],
                             top_k: int) -> List[RetrievalResult]:
        
        # Query knowledge graph
        graph_results = await asyncio.to_thread(
            self.knowledge_graph.query_entities,
            entities,
            top_k
        )
        
        # Convert to RetrievalResult
        results = []
        for node_id, data, score in graph_results:
            retrieval_result = RetrievalResult(
                doc_id=node_id,
                content=data.get('content', ''),
                score=score,
                retrieval_method='graph',
                metadata=data,
                relevance_explanation=f"Knowledge graph connection: {score:.3f}",
                medical_entities=data.get('entities', []),
                confidence=score
            )
            results.append(retrieval_result)
        
        return results
    
    def _enhance_query(self, query: str, context: RetrievalContext) -> str:
        enhanced = query
        
        # Add medical context
        if context.query_type == QueryType.DIAGNOSIS:
            enhanced = f"Diagnosis: {query}"
        elif context.query_type == QueryType.TREATMENT:
            enhanced = f"Treatment protocol: {query}"
        elif context.query_type == QueryType.LAB_INTERPRETATION:
            enhanced = f"Lab results interpretation: {query}"
        
        # Add temporal context
        if context.temporal_context:
            if 'acute' in context.temporal_context:
                enhanced += " (acute presentation)"
            elif 'chronic' in context.temporal_context:
                enhanced += " (chronic condition)"
        
        # Add patient context
        if context.patient_context:
            if 'age' in context.patient_context:
                age = context.patient_context['age']
                if age < 18:
                    enhanced += " (pediatric)"
                elif age > 65:
                    enhanced += " (geriatric)"
        
        return enhanced
    
    def _extract_keywords(self, query: str, context: RetrievalContext) -> str:
        keywords = []
        
        # Add medical entities
        for entity in context.medical_entities:
            if entity.get('type') in ['DISEASE', 'SYMPTOM', 'MEDICATION', 'PROCEDURE']:
                keywords.append(entity['text'])
        
        # Add query type specific keywords
        type_keywords = {
            QueryType.DIAGNOSIS: ['diagnosis', 'differential', 'assessment'],
            QueryType.TREATMENT: ['treatment', 'management', 'therapy'],
            QueryType.LAB_INTERPRETATION: ['lab', 'test', 'result', 'value']
        }
        
        if context.query_type in type_keywords:
            keywords.extend(type_keywords[context.query_type])
        
        # Add original query terms
        doc = self.query_analyzer.nlp(query)
        for token in doc:
            if not token.is_stop and token.pos_ in ['NOUN', 'VERB', 'ADJ']:
                keywords.append(token.lemma_)
        
        return ' '.join(list(set(keywords)))
    
    def _combine_results(self,
                        all_results: List[List[RetrievalResult]],
                        context: RetrievalContext) -> List[RetrievalResult]:
        
        # Reciprocal Rank Fusion
        doc_scores = defaultdict(float)
        doc_data = {}
        
        k = 60  # RRF parameter
        
        for results in all_results:
            for rank, result in enumerate(results, 1):
                doc_scores[result.doc_id] += 1 / (k + rank)
                
                # Keep the result with highest individual score
                if result.doc_id not in doc_data or result.score > doc_data[result.doc_id].score:
                    doc_data[result.doc_id] = result
        
        # Combine scores
        combined_results = []
        for doc_id, rrf_score in doc_scores.items():
            result = doc_data[doc_id]
            result.score = rrf_score  # Use RRF score
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results
    
    def _rerank_results(self,
                       query: str,
                       results: List[RetrievalResult],
                       context: RetrievalContext) -> List[RetrievalResult]:
        
        if not results:
            return results
        
        # Prepare pairs for reranking
        pairs = [(query, result.content) for result in results]
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)
        
        # Apply medical-specific boosting
        for i, result in enumerate(results):
            base_score = float(rerank_scores[i])
            
            # Boost based on query type match
            if self._matches_query_type(result.content, context.query_type):
                base_score *= 1.2
            
            # Boost for entity overlap
            entity_overlap = self._calculate_entity_overlap(result, context.medical_entities)
            base_score *= (1 + entity_overlap * 0.3)
            
            # Boost for recency if temporal context indicates recent
            if context.temporal_context and 'recent' in context.temporal_context:
                if 'date' in result.metadata:
                    doc_date = datetime.fromisoformat(result.metadata['date'])
                    days_old = (datetime.now() - doc_date).days
                    recency_boost = max(0, 1 - days_old / 365)
                    base_score *= (1 + recency_boost * 0.1)
            
            # Update score
            result.score = base_score
            result.confidence = min(1.0, base_score)
        
        # Sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def _matches_query_type(self, content: str, query_type: QueryType) -> bool:
        content_lower = content.lower()
        
        type_keywords = {
            QueryType.DIAGNOSIS: ['diagnosis', 'diagnosed', 'assessment', 'findings'],
            QueryType.TREATMENT: ['treatment', 'therapy', 'management', 'intervention'],
            QueryType.LAB_INTERPRETATION: ['lab', 'test', 'result', 'value', 'range'],
            QueryType.MEDICATION_INFO: ['medication', 'drug', 'dose', 'prescription'],
            QueryType.PROCEDURE_DETAILS: ['procedure', 'surgery', 'operation']
        }
        
        if query_type in type_keywords:
            return any(keyword in content_lower for keyword in type_keywords[query_type])
        
        return False
    
    def _calculate_entity_overlap(self,
                                  result: RetrievalResult,
                                  query_entities: List[Dict[str, Any]]) -> float:
        
        if not query_entities:
            return 0.0
        
        result_text_lower = result.content.lower()
        overlap_count = 0
        
        for entity in query_entities:
            if entity['text'].lower() in result_text_lower:
                overlap_count += 1
        
        return overlap_count / len(query_entities)
    
    def _apply_medical_filters(self,
                              results: List[RetrievalResult],
                              context: RetrievalContext) -> List[RetrievalResult]:
        
        filtered = []
        
        for result in results:
            # Check confidence threshold
            if result.confidence < context.confidence_threshold:
                continue
            
            # Check for required expertise
            if context.required_expertise:
                doc_expertise = result.metadata.get('expertise', [])
                if not any(exp in doc_expertise for exp in context.required_expertise):
                    result.score *= 0.8  # Penalty for missing expertise
            
            # Check for contradictions in critical cases
            if context.urgency_level == 'critical':
                if self._contains_contradiction(result.content):
                    result.confidence *= 0.7
                    result.relevance_explanation += " (contains potential contradictions)"
            
            filtered.append(result)
        
        return filtered
    
    def _contains_contradiction(self, content: str) -> bool:
        contradiction_patterns = [
            r'\b(?:not recommended|contraindicated|avoid|do not)\b',
            r'\b(?:however|but|although|despite)\b.*\b(?:not|no|avoid)\b'
        ]
        
        content_lower = content.lower()
        for pattern in contradiction_patterns:
            if re.search(pattern, content_lower):
                return True
        
        return False
    
    def _generate_cache_key(self,
                           query: str,
                           top_k: int,
                           filters: Optional[Dict[str, Any]]) -> str:
        
        key_data = {
            'query': query,
            'top_k': top_k,
            'filters': filters or {}
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

class BM25Retriever:
    def __init__(self):
        self.documents = []
        self.doc_ids = []
        self.bm25 = None
    
    def index_documents(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in documents]
        
        # Tokenize documents
        tokenized_docs = [doc['content'].lower().split() for doc in documents]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(tokenized_docs)
        
        logger.info(f"Indexed {len(documents)} documents with BM25")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, str]]:
        if not self.bm25:
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((
                    self.doc_ids[idx],
                    float(scores[idx]),
                    self.documents[idx]['content']
                ))
        
        return results

class MedicalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self._initialize_medical_ontology()
    
    def _initialize_medical_ontology(self):
        # Add sample medical relationships
        medical_relationships = [
            ('myocardial_infarction', 'troponin', 'biomarker'),
            ('myocardial_infarction', 'chest_pain', 'symptom'),
            ('myocardial_infarction', 'aspirin', 'treatment'),
            ('diabetes', 'metformin', 'treatment'),
            ('diabetes', 'glucose', 'biomarker'),
            ('hypertension', 'lisinopril', 'treatment'),
            ('hypertension', 'blood_pressure', 'measurement')
        ]
        
        for source, target, relation in medical_relationships:
            self.graph.add_edge(source, target, relation=relation)
            self.graph.nodes[source]['type'] = 'condition'
            self.graph.nodes[target]['type'] = relation
    
    def query_entities(self,
                       entities: List[Dict[str, Any]],
                       top_k: int = 10) -> List[Tuple[str, Dict[str, Any], float]]:
        
        results = []
        
        for entity in entities:
            entity_text = entity['text'].lower().replace(' ', '_')
            
            if entity_text in self.graph:
                # Get neighbors
                neighbors = list(self.graph.neighbors(entity_text))
                
                for neighbor in neighbors[:top_k]:
                    edge_data = self.graph.edges[entity_text, neighbor]
                    node_data = self.graph.nodes[neighbor]
                    
                    score = 1.0 / (1 + len(list(nx.shortest_path(self.graph, entity_text, neighbor))) - 1)
                    
                    results.append((
                        neighbor,
                        {'content': neighbor, 'type': node_data.get('type'), 'relation': edge_data.get('relation')},
                        score
                    ))
        
        # Sort by score
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:top_k]

if __name__ == "__main__":
    # Example usage
    from rag_chatbot.app.core.medical_embeddings import MedicalEmbeddingEngine, QdrantVectorStore, EmbeddingModel
    
    # Initialize components
    embedding_engine = MedicalEmbeddingEngine(model_type=EmbeddingModel.PUBMEDBERT)
    dense_retriever = QdrantVectorStore("medical_docs", embedding_engine)
    
    # Initialize hybrid retriever
    retriever = HybridMedicalRetriever(
        dense_retriever=dense_retriever,
        reranker_model='cross-encoder/ms-marco-MiniLM-L-6-v2'
    )
    
    # Example query
    query = "Patient with elevated troponin I at 2.5 ng/mL and chest pain. What is the diagnosis?"
    
    # Retrieve documents
    async def test_retrieval():
        results = await retriever.retrieve(query, top_k=5)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Document ID: {result.doc_id}")
            print(f"   Score: {result.score:.3f} | Confidence: {result.confidence:.3f}")
            print(f"   Method: {result.retrieval_method}")
            print(f"   Explanation: {result.relevance_explanation}")
            print(f"   Content: {result.content[:200]}...")
    
    # Run test
    asyncio.run(test_retrieval())