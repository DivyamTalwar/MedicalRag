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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DenseSearchEngine:    
    def __init__(self, index_name: str = "children"):
        self.embeddings = get_embedding_model()
        self.pinecone_client = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pinecone_client.Index(index_name)
        
    def search(self, query_text: str, top_k: int = 10) -> List[Document]:
        try:
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
            
            documents = []
            for match in results_dict["matches"]:
                doc = Document(
                    id=match['id'],
                    text=match['metadata'].get("text", ""),
                    metadata=match['metadata']
                )
                doc.metadata['score'] = match.get('score', 0.0)
                documents.append(doc)
            
            logging.info(f"Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logging.error(f"An error occurred during Pinecone query: {e}")
            return []

class MedicalReranker:    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        try:
            self.model = CrossEncoder(model_name)
            logging.info(f"Cross-encoder model '{model_name}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load cross-encoder model: {e}")
            self.model = None
        
        self.medical_patterns = {
            'lab_values': [
                r'\d+\.\d+\s*(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL|%)',
                r'\d+\s*-\s*\d+(?:\.\d+)?', 
                r'\b(?:HbA1c|TSH|HDL|LDL|CBC|ESR|WBC|RBC|T3|T4)\b'
            ],
            'measurements': [
                r'\b(?:pH|PCO2|PO2|HCO3|TCO2)\b',  
                r'\b(?:P2|P3)\s*(?:Peak|Window)\b',
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
        text_lower = text.lower()
        query_lower = query.lower()
        boost_score = 0.0
        
        for category, patterns in self.medical_patterns.items():
            category_matches = 0
            for pattern in patterns:
                text_matches = len(re.findall(pattern, text, re.IGNORECASE))
                query_matches = len(re.findall(pattern, query, re.IGNORECASE))
                
                if text_matches > 0 and query_matches > 0:
                    category_matches += min(text_matches, query_matches)
            
            category_weights = {
                'lab_values': 0.4,
                'measurements': 0.3,
                'procedures': 0.2,
                'clinical_terms': 0.1
            }
            boost_score += category_matches * category_weights.get(category, 0.1)
        
        value_pattern = r'\d+\.\d+'
        query_values = set(re.findall(value_pattern, query))
        text_values = set(re.findall(value_pattern, text))
        exact_value_matches = len(query_values.intersection(text_values))
        boost_score += exact_value_matches * 0.5
        
        return min(1.0, boost_score) 
    
    def _calculate_medical_entity_preservation(self, text: str, query: str) -> float:
        entity_patterns = [
            r'\b[A-Z]{2,}\b',
            r'\b(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL|%)\b',
            r'\b(?:HbA1c|TSH|HDL|LDL|P2|P3|pH|PCO2|PO2)\b' 
        ]
        
        query_entities = set()
        text_entities = set()
        
        for pattern in entity_patterns:
            query_entities.update(re.findall(pattern, query, re.IGNORECASE))
            text_entities.update(re.findall(pattern, text, re.IGNORECASE))
        
        if not query_entities:
            return 1.0 
        
        preserved_entities = query_entities.intersection(text_entities)
        preservation_ratio = len(preserved_entities) / len(query_entities)
        
        return preservation_ratio
    
    def _apply_similarity_threshold_filtering(self, documents: List[Document], threshold: float = 0.1) -> List[Document]:
        if not documents:
            return documents
        
        filtered_docs = [doc for doc in documents if doc.metadata.get('rerank_score', 0.0) >= threshold]
        
        if len(filtered_docs) < 5 and len(documents) >= 5:
            return sorted(documents, key=lambda x: x.metadata.get('rerank_score', 0.0), reverse=True)[:5]
        
        return filtered_docs

    def rerank(self, query: str, documents: List[Document], top_k: int = 10) -> List[Document]:
        if not self.model or not documents:
            return documents[:top_k] if documents else []

        try:
            pairs = [[query, doc.text] for doc in documents]
            
            base_scores = self.model.predict(pairs, show_progress_bar=False)
            
            if not isinstance(base_scores, (list, np.ndarray)):
                logging.error(f"Reranker did not return a list or numpy array: {base_scores}")
                return documents[:top_k]
            
            for doc, base_score in zip(documents, base_scores):
                medical_boost = self._calculate_medical_relevance_boost(doc.text, query)
                
                entity_preservation = self._calculate_medical_entity_preservation(doc.text, query)
                
                final_score = (
                    base_score * 0.6 +  
                    medical_boost * 0.3 + 
                    entity_preservation * 0.1  
                )
                
                doc.metadata.update({
                    'rerank_score': float(final_score),
                    'base_score': float(base_score),
                    'medical_boost': float(medical_boost),
                    'entity_preservation': float(entity_preservation)
                })
            
            ranked_documents = sorted(documents, key=lambda x: x.metadata['rerank_score'], reverse=True)
            
            filtered_documents = self._apply_similarity_threshold_filtering(ranked_documents)
            
            logging.info(f"Reranked {len(documents)} documents, filtered to {len(filtered_documents)}, returning top {min(top_k, len(filtered_documents))}")
            
            return filtered_documents[:top_k]
            
        except Exception as e:
            logging.error(f"Error during medical reranking: {e}")
            return documents[:top_k]

Reranker = MedicalReranker
