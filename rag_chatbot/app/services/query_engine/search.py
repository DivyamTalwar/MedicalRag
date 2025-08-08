import os
import json
import logging
import numpy as np
import re
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from rag_chatbot.app.models.data_models import Document
from rag_chatbot.app.core.embeddings import get_embedding_model
import pinecone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MedicalQueryClassifier:
    """Classify medical queries to determine optimal retrieval strategy"""
    
    def __init__(self):
        self.query_patterns = {
            'lab_results': [
                r'\b(lab|test|result|value|level|count|analysis)\b',
                r'\b(glucose|cholesterol|hemoglobin|hba1c|tsh|hdl|ldl)\b',
                r'\b(blood|urine|serum|plasma)\b',
                r'\d+\.\d+\s*(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL|%)',
                r'\b(normal|abnormal|high|low|elevated)\b'
            ],
            'procedures': [
                r'\b(procedure|scan|exam|study|imaging|radiology)\b',
                r'\b(ct|mri|xray|ultrasound|pacs|ris)\b',
                r'\b(workflow|turnaround|time|tat|scheduling)\b',
                r'\b(dicom|worklist|technologist|radiologist)\b'
            ],
            'symptoms': [
                r'\b(symptom|pain|ache|fever|nausea|fatigue)\b',
                r'\b(patient|clinical|diagnosis|treatment)\b',
                r'\b(condition|disease|disorder|syndrome)\b'
            ],
            'reference_ranges': [
                r'\b(reference|range|normal|limit)\b',
                r'\d+\s*-\s*\d+(?:\.\d+)?',
                r'\b(within|outside|above|below)\b'
            ]
        }
    
    def classify_query(self, query: str) -> Tuple[str, float]:
        """Classify query and return type with confidence score"""
        query_lower = query.lower()
        scores = {}
        
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower, re.IGNORECASE))
                score += matches
            scores[query_type] = score
        
        if not any(scores.values()):
            return 'general', 0.5
        
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        confidence = min(1.0, max_score / 3.0)  # Normalize confidence
        
        return best_type, confidence

class DenseSearchEngine:
    """Enhanced search engine with dynamic top_k and medical query awareness"""
    
    def __init__(self, index_name: str = "children"):
        self.embeddings = get_embedding_model()
        self.pinecone_client = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pinecone_client.Index(index_name)
        self.query_classifier = MedicalQueryClassifier()
        
        # Dynamic top_k configuration based on query type
        self.top_k_config = {
            'lab_results': {'base': 20, 'max': 40},
            'procedures': {'base': 25, 'max': 50},
            'symptoms': {'base': 30, 'max': 60},
            'reference_ranges': {'base': 15, 'max': 30},
            'general': {'base': 25, 'max': 45}
        }
        
        # Similarity threshold for filtering
        self.similarity_threshold = 0.3

    def _determine_dynamic_top_k(self, query: str, query_type: str, confidence: float) -> int:
        """Determine optimal top_k based on query complexity and type"""
        config = self.top_k_config[query_type]
        base_k = config['base']
        max_k = config['max']
        
        # Adjust based on query complexity
        query_complexity = self._calculate_query_complexity(query)
        complexity_multiplier = 1.0 + (query_complexity * 0.5)
        
        # Adjust based on confidence
        confidence_multiplier = 0.8 + (confidence * 0.4)
        
        dynamic_k = int(base_k * complexity_multiplier * confidence_multiplier)
        return min(dynamic_k, max_k)
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity based on various factors"""
        factors = {
            'length': len(query.split()) / 10.0,  # Normalize by typical query length
            'medical_terms': len(re.findall(r'\b(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL|TSH|HDL|LDL|HbA1c|PACS|RIS)\b', query, re.IGNORECASE)) / 5.0,
            'numerical_values': len(re.findall(r'\d+\.\d+|\d+\s*-\s*\d+', query)) / 3.0,
            'question_words': len(re.findall(r'\b(?:what|how|why|when|where|which|who)\b', query, re.IGNORECASE)) / 2.0
        }
        
        # Weighted average of complexity factors
        weights = {'length': 0.3, 'medical_terms': 0.4, 'numerical_values': 0.2, 'question_words': 0.1}
        complexity = sum(min(1.0, factors[factor]) * weights[factor] for factor in factors)
        
        return min(1.0, complexity)
    
    def _filter_by_similarity_threshold(self, documents: List[Document]) -> List[Document]:
        """Filter documents by similarity threshold"""
        if not documents:
            return documents
        
        filtered_docs = []
        for doc in documents:
            similarity_score = doc.metadata.get('score', 0.0)
            if similarity_score >= self.similarity_threshold:
                filtered_docs.append(doc)
        
        # If filtering removes too many results, keep top results anyway
        if len(filtered_docs) < 5 and len(documents) >= 5:
            return documents[:10]  # Keep at least 10 documents
        
        return filtered_docs

    def search(self, query_text: str, top_k: int = None) -> List[Document]:
        """Enhanced search with dynamic top_k and similarity filtering"""
        try:
            # Classify query to determine optimal strategy
            query_type, confidence = self.query_classifier.classify_query(query_text)
            
            # Determine dynamic top_k if not provided
            if top_k is None:
                top_k = self._determine_dynamic_top_k(query_text, query_type, confidence)
            
            logging.info(f"Query type: {query_type}, confidence: {confidence:.2f}, using top_k: {top_k}")
            
            # Perform vector search
            query_vector = self.embeddings.embed_query(query_text)
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            results_dict = results.to_dict()
            
            if "matches" not in results_dict:
                logging.error(f"Unexpected response format from Pinecone: {results}")
                return []
            
            # Create Document objects with similarity scores
            documents = []
            for match in results_dict["matches"]:
                doc = Document(
                    id=match['id'],
                    text=match['metadata'].get("text", ""),
                    metadata=match['metadata']
                )
                # Add similarity score to metadata
                doc.metadata['score'] = match.get('score', 0.0)
                doc.metadata['query_type'] = query_type
                documents.append(doc)
            
            # Apply similarity threshold filtering
            filtered_documents = self._filter_by_similarity_threshold(documents)
            
            logging.info(f"Retrieved {len(documents)} documents, filtered to {len(filtered_documents)}")
            return filtered_documents
            
        except Exception as e:
            logging.error(f"An error occurred during Pinecone query: {e}")
            return []

class MedicalReranker:
    """Enhanced reranker with medical-specific relevance boosting"""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        try:
            self.model = CrossEncoder(model_name)
            logging.info(f"Cross-encoder model '{model_name}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load cross-encoder model: {e}")
            self.model = None
        
        # Medical relevance patterns based on your sample documents
        self.medical_patterns = {
            'lab_values': [
                r'\d+\.\d+\s*(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL|%)',
                r'\d+\s*-\s*\d+(?:\.\d+)?',  # Reference ranges
                r'\b(?:HbA1c|TSH|HDL|LDL|CBC|ESR|WBC|RBC|T3|T4)\b'
            ],
            'measurements': [
                r'\b(?:pH|PCO2|PO2|HCO3|TCO2)\b',  # Blood gas
                r'\b(?:P2|P3)\s*(?:Peak|Window)\b',  # Hemoglobin electrophoresis
                r'\b(?:Hb\s*[AF]2?|Hemoglobin\s*[AF]2?)\b'
            ],
            'procedures': [
                r'\b(?:PACS|RIS|DICOM|Worklist)\b',
                r'\b(?:Radiology|Imaging|CT|MRI|Ultrasound|X-ray)\b',
                r'\b(?:TAT|Turnaround|Time|Workflow)\b'
            ],
            'clinical_terms': [
                r'\b(?:TORCH|Toxoplasma|Rubella|Cytomegalovirus|Herpes)\b',
                r'\b(?:Positive|Negative|Reactive|Non.?Reactive)\b',
                r'\b(?:Normal|Abnormal|Elevated|High|Low)\b'
            ]
        }
    
    def _calculate_medical_relevance_boost(self, text: str, query: str) -> float:
        """Calculate medical relevance boost score"""
        text_lower = text.lower()
        query_lower = query.lower()
        boost_score = 0.0
        
        # Boost for matching medical patterns
        for category, patterns in self.medical_patterns.items():
            category_matches = 0
            for pattern in patterns:
                text_matches = len(re.findall(pattern, text, re.IGNORECASE))
                query_matches = len(re.findall(pattern, query, re.IGNORECASE))
                
                # Boost if both query and text contain similar medical terms
                if text_matches > 0 and query_matches > 0:
                    category_matches += min(text_matches, query_matches)
            
            # Weight different categories
            category_weights = {
                'lab_values': 0.4,
                'measurements': 0.3,
                'procedures': 0.2,
                'clinical_terms': 0.1
            }
            boost_score += category_matches * category_weights.get(category, 0.1)
        
        # Additional boost for exact value matches (critical for lab results)
        value_pattern = r'\d+\.\d+'
        query_values = set(re.findall(value_pattern, query))
        text_values = set(re.findall(value_pattern, text))
        exact_value_matches = len(query_values.intersection(text_values))
        boost_score += exact_value_matches * 0.5
        
        return min(1.0, boost_score)  # Cap at 1.0
    
    def _calculate_medical_entity_preservation(self, text: str, query: str) -> float:
        """Calculate how well medical entities are preserved"""
        # Extract medical entities from query
        entity_patterns = [
            r'\b[A-Z]{2,}\b',  # Abbreviations
            r'\b(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL|%)\b',  # Units
            r'\b(?:HbA1c|TSH|HDL|LDL|P2|P3|pH|PCO2|PO2)\b'  # Specific terms
        ]
        
        query_entities = set()
        text_entities = set()
        
        for pattern in entity_patterns:
            query_entities.update(re.findall(pattern, query, re.IGNORECASE))
            text_entities.update(re.findall(pattern, text, re.IGNORECASE))
        
        if not query_entities:
            return 1.0  # No entities to preserve
        
        preserved_entities = query_entities.intersection(text_entities)
        preservation_ratio = len(preserved_entities) / len(query_entities)
        
        return preservation_ratio
    
    def _apply_similarity_threshold_filtering(self, documents: List[Document], threshold: float = 0.1) -> List[Document]:
        """Remove documents with very low rerank scores"""
        if not documents:
            return documents
        
        filtered_docs = [doc for doc in documents if doc.metadata.get('rerank_score', 0.0) >= threshold]
        
        # Ensure we keep at least a few documents
        if len(filtered_docs) < 3 and len(documents) >= 3:
            return sorted(documents, key=lambda x: x.metadata.get('rerank_score', 0.0), reverse=True)[:5]
        
        return filtered_docs

    def rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
        """Enhanced reranking with medical-specific relevance and entity preservation"""
        if not self.model or not documents:
            return documents[:top_k] if documents else []

        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, doc.text] for doc in documents]
            
            # Get base reranking scores
            base_scores = self.model.predict(pairs, show_progress_bar=False)
            
            if not isinstance(base_scores, (list, np.ndarray)):
                logging.error(f"Reranker did not return a list or numpy array: {base_scores}")
                return documents[:top_k]
            
            # Apply medical-specific enhancements
            for doc, base_score in zip(documents, base_scores):
                # Calculate medical relevance boost
                medical_boost = self._calculate_medical_relevance_boost(doc.text, query)
                
                # Calculate entity preservation score
                entity_preservation = self._calculate_medical_entity_preservation(doc.text, query)
                
                # Combine scores with weights
                final_score = (
                    base_score * 0.6 +  # Base cross-encoder score
                    medical_boost * 0.3 +  # Medical relevance boost
                    entity_preservation * 0.1  # Entity preservation
                )
                
                # Store all scores in metadata for debugging
                doc.metadata.update({
                    'rerank_score': float(final_score),
                    'base_score': float(base_score),
                    'medical_boost': float(medical_boost),
                    'entity_preservation': float(entity_preservation)
                })
            
            # Sort by final score
            ranked_documents = sorted(documents, key=lambda x: x.metadata['rerank_score'], reverse=True)
            
            # Apply similarity threshold filtering
            filtered_documents = self._apply_similarity_threshold_filtering(ranked_documents)
            
            logging.info(f"Reranked {len(documents)} documents, filtered to {len(filtered_documents)}, returning top {min(top_k, len(filtered_documents))}")
            
            return filtered_documents[:top_k]
            
        except Exception as e:
            logging.error(f"Error during medical reranking: {e}")
            return documents[:top_k]

# Backward compatibility - alias the enhanced reranker
Reranker = MedicalReranker
