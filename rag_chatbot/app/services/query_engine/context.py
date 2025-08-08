import os
import logging
import re
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
from difflib import SequenceMatcher
from rag_chatbot.app.models.data_models import Document
import pinecone

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from pymongo import MongoClient
from bson.codec_options import CodecOptions, UuidRepresentation
from ...core.config import MONGO_DB_NAME

class MedicalContextAssembler:
    """
    Enhanced context assembler with medical-specific deduplication, highlighting, and quality scoring
    """
    def __init__(self):
        self.mongo_client = MongoClient(os.getenv("MONGO_URI"))
        codec_options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
        self.db = self.mongo_client.get_database(MONGO_DB_NAME, codec_options=codec_options)
        self.collection = self.db.get_collection(MONGO_DB_NAME)
        
        # Medical entity patterns based on your sample documents
        self.medical_patterns = {
            'lab_values': [
                r'\d+\.\d+\s*(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL|%)',
                r'\d+\s*-\s*\d+(?:\.\d+)?',  # Reference ranges
                r'[<>]\s*\d+\.\d+'  # Comparison values
            ],
            'medical_terms': [
                r'\b(?:HbA1c|TSH|HDL|LDL|CBC|ESR|WBC|RBC|T3|T4)\b',
                r'\b(?:pH|PCO2|PO2|HCO3|TCO2|SBC|BE)\b',  # Blood gas
                r'\b(?:Toxoplasma|Rubella|Cytomegalovirus|Herpes)\b',  # TORCH
                r'\b(?:P2|P3)\s*(?:Peak|Window)\b',  # Hemoglobin electrophoresis
                r'\b(?:PACS|RIS|DICOM|TAT|SLA)\b'  # Radiology workflow
            ],
            'clinical_states': [
                r'\b(?:Positive|Negative|Reactive|Non-Reactive)\b',
                r'\b(?:Normal|Abnormal|High|Low|Elevated)\b',
                r'\b(?:Deficiency|Insufficiency|Sufficiency|Toxicity)\b'
            ],
            'medical_units': [
                r'\b(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL|ng/mL|pg/mL|micromol/L|fL|g/dL)\b'
            ]
        }
        
        # Medical terminology normalization mappings
        self.terminology_normalizations = {
            # Lab test variations
            'hba1c': ['glycosylated hemoglobin', 'glycated hemoglobin', 'hemoglobin a1c'],
            'tsh': ['thyroid stimulating hormone', 'thyrotropin'],
            'hdl': ['high density lipoprotein', 'good cholesterol'],
            'ldl': ['low density lipoprotein', 'bad cholesterol'],
            'cbc': ['complete blood count', 'full blood count'],
            'esr': ['erythrocyte sedimentation rate', 'sed rate'],
            
            # Units normalization
            'mg/dl': ['mg/dL', 'milligrams per deciliter'],
            'mmhg': ['mmHg', 'mm Hg', 'millimeters mercury'],
            'meq/l': ['mEq/L', 'milliequivalents per liter'],
            
            # Clinical terms
            'positive': ['reactive', '+'],
            'negative': ['non-reactive', 'non reactive', '-'],
            'normal': ['within normal limits', 'wnl', 'reference range'],
            'abnormal': ['outside normal', 'elevated', 'decreased']
        }
        
        # Similarity threshold for deduplication
        self.semantic_similarity_threshold = 0.85
        self.coherence_threshold = 0.6

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two text chunks"""
        # Normalize texts for comparison
        normalized_text1 = self._normalize_medical_terminology(text1.lower())
        normalized_text2 = self._normalize_medical_terminology(text2.lower())
        
        # Use SequenceMatcher for basic similarity
        sequence_similarity = SequenceMatcher(None, normalized_text1, normalized_text2).ratio()
        
        # Extract medical entities for enhanced comparison
        entities1 = self._extract_medical_entities(text1)
        entities2 = self._extract_medical_entities(text2)
        
        # Calculate entity overlap
        if entities1 or entities2:
            all_entities = entities1.union(entities2)
            common_entities = entities1.intersection(entities2)
            entity_similarity = len(common_entities) / len(all_entities) if all_entities else 0
        else:
            entity_similarity = 0
        
        # Weighted combination
        final_similarity = (sequence_similarity * 0.7) + (entity_similarity * 0.3)
        return final_similarity

    def _extract_medical_entities(self, text: str) -> Set[str]:
        """Extract medical entities from text"""
        entities = set()
        
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities.update([match.lower().strip() for match in matches])
        
        return entities

    def _normalize_medical_terminology(self, text: str) -> str:
        """Normalize medical terminology across chunks"""
        normalized_text = text
        
        for standard_term, variations in self.terminology_normalizations.items():
            for variation in variations:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(variation) + r'\b'
                normalized_text = re.sub(pattern, standard_term, normalized_text, flags=re.IGNORECASE)
        
        return normalized_text

    def _semantic_deduplication(self, chunks: List[Document]) -> List[Document]:
        """Remove semantically similar chunks while preserving the most informative ones"""
        if len(chunks) <= 1:
            return chunks
        
        deduplicated_chunks = []
        similarity_matrix = {}
        
        # Calculate similarity matrix
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                similarity = self._calculate_semantic_similarity(chunk1.text, chunk2.text)
                similarity_matrix[(i, j)] = similarity
        
        # Find duplicates and select the best representative
        used_indices = set()
        duplicate_groups = defaultdict(list)
        
        for i, chunk in enumerate(chunks):
            if i in used_indices:
                continue
            
            group = [i]
            for j in range(i+1, len(chunks)):
                if j in used_indices:
                    continue
                
                similarity = similarity_matrix.get((i, j), 0)
                if similarity >= self.semantic_similarity_threshold:
                    group.append(j)
                    used_indices.add(j)
            
            # Select the best representative from the group
            if len(group) > 1:
                best_idx = self._select_best_representative(chunks, group)
                deduplicated_chunks.append(chunks[best_idx])
                logging.info(f"Deduplicated {len(group)} similar chunks, kept chunk {best_idx}")
            else:
                deduplicated_chunks.append(chunks[i])
            
            used_indices.add(i)
        
        return deduplicated_chunks

    def _select_best_representative(self, chunks: List[Document], group_indices: List[int]) -> int:
        """Select the most informative chunk from a group of similar chunks"""
        best_idx = group_indices[0]
        best_score = 0
        
        for idx in group_indices:
            chunk = chunks[idx]
            
            # Score based on multiple factors
            score = 0
            
            # Length score (longer chunks often more informative)
            score += len(chunk.text.split()) * 0.1
            
            # Medical entity density
            entities = self._extract_medical_entities(chunk.text)
            score += len(entities) * 2
            
            # Rerank score if available
            rerank_score = chunk.metadata.get('rerank_score', 0)
            score += rerank_score * 10
            
            # Recent document preference
            page_no = chunk.metadata.get('page_no', 0)
            score += (1.0 / (page_no + 1)) * 0.5
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        return best_idx

    def _highlight_medical_entities(self, text: str) -> str:
        """Highlight medical entities in the context"""
        highlighted_text = text
        
        # Track highlighted positions to avoid overlapping
        highlights = []
        
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start, end = match.span()
                    # Check for overlaps
                    if not any(s <= start < e or s < end <= e for s, e in highlights):
                        highlights.append((start, end))
        
        # Sort by position (reverse order for replacement)
        highlights.sort(reverse=True)
        
        # Apply highlighting
        for start, end in highlights:
            entity = text[start:end]
            highlighted_text = highlighted_text[:start] + f"**{entity}**" + highlighted_text[end:]
        
        return highlighted_text

    def _calculate_context_quality_score(self, chunks: List[Document]) -> float:
        """Calculate overall quality score for the assembled context"""
        if not chunks:
            return 0.0
        
        quality_factors = {
            'medical_entity_density': 0,
            'coherence_score': 0,
            'completeness_score': 0,
            'relevance_score': 0
        }
        
        total_text = " ".join([chunk.text for chunk in chunks])
        total_entities = self._extract_medical_entities(total_text)
        
        # Medical entity density
        total_words = len(total_text.split())
        entity_density = len(total_entities) / total_words if total_words > 0 else 0
        quality_factors['medical_entity_density'] = min(1.0, entity_density * 50)  # Scale appropriately
        
        # Coherence score (semantic flow between chunks)
        coherence_scores = []
        for i in range(len(chunks) - 1):
            similarity = self._calculate_semantic_similarity(chunks[i].text, chunks[i+1].text)
            coherence_scores.append(similarity)
        
        quality_factors['coherence_score'] = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0
        
        # Completeness score (presence of reference ranges, normal values)
        completeness_patterns = [
            r'\d+\s*-\s*\d+',  # Reference ranges
            r'normal|reference',  # Reference mentions
            r'interpretation|conclusion'  # Clinical interpretation
        ]
        completeness_matches = 0
        for pattern in completeness_patterns:
            if re.search(pattern, total_text, re.IGNORECASE):
                completeness_matches += 1
        
        quality_factors['completeness_score'] = completeness_matches / len(completeness_patterns)
        
        # Relevance score (average rerank scores)
        rerank_scores = [chunk.metadata.get('rerank_score', 0.5) for chunk in chunks]
        quality_factors['relevance_score'] = sum(rerank_scores) / len(rerank_scores)
        
        # Weighted average
        weights = {
            'medical_entity_density': 0.3,
            'coherence_score': 0.25,
            'completeness_score': 0.25,
            'relevance_score': 0.2
        }
        
        overall_score = sum(quality_factors[factor] * weights[factor] for factor in quality_factors)
        return min(1.0, max(0.0, overall_score))

    def _validate_context_coherence(self, chunks: List[Document]) -> Tuple[bool, List[str]]:
        """Validate logical flow and coherence of assembled context"""
        if len(chunks) < 2:
            return True, []
        
        issues = []
        
        # Check for logical flow between adjacent chunks
        for i in range(len(chunks) - 1):
            current_entities = self._extract_medical_entities(chunks[i].text)
            next_entities = self._extract_medical_entities(chunks[i+1].text)
            
            # Check semantic continuity
            similarity = self._calculate_semantic_similarity(chunks[i].text, chunks[i+1].text)
            if similarity < 0.1:  # Very low similarity might indicate poor flow
                issues.append(f"Low coherence between chunk {i+1} and {i+2} (similarity: {similarity:.2f})")
            
            # Check for contradictory values
            if self._detect_contradictory_values(chunks[i].text, chunks[i+1].text):
                issues.append(f"Potential contradictory values between chunk {i+1} and {i+2}")
        
        # Check for missing critical context
        all_text = " ".join([chunk.text for chunk in chunks])
        if not self._has_sufficient_context(all_text):
            issues.append("Context may lack sufficient medical detail")
        
        coherence_valid = len(issues) == 0
        return coherence_valid, issues

    def _detect_contradictory_values(self, text1: str, text2: str) -> bool:
        """Detect contradictory medical values between chunks"""
        # Extract numerical values with same units
        value_pattern = r'(\w+)\s*:?\s*(\d+(?:\.\d+)?)\s*(mg/dL|mmHg|mEq/L|%|IU/mL|AU/mL|U/mL)'
        
        values1 = {}
        values2 = {}
        
        for match in re.finditer(value_pattern, text1, re.IGNORECASE):
            parameter, value, unit = match.groups()
            key = f"{parameter.lower()}_{unit.lower()}"
            values1[key] = float(value)
        
        for match in re.finditer(value_pattern, text2, re.IGNORECASE):
            parameter, value, unit = match.groups()
            key = f"{parameter.lower()}_{unit.lower()}"
            values2[key] = float(value)
        
        # Check for significant differences in same parameters
        for key in values1:
            if key in values2:
                diff_ratio = abs(values1[key] - values2[key]) / max(values1[key], values2[key])
                if diff_ratio > 0.5:  # More than 50% difference
                    return True
        
        return False

    def _has_sufficient_context(self, text: str) -> bool:
        """Check if context has sufficient medical detail"""
        # Count medical entities
        entities = self._extract_medical_entities(text)
        
        # Minimum requirements for sufficient context
        min_entities = 3
        min_words = 100
        
        word_count = len(text.split())
        
        return len(entities) >= min_entities and word_count >= min_words

    def assemble(self, reranked_child_chunks: List[Document]) -> Tuple[List[Document], str]:
        """Enhanced context assembly with all medical-specific improvements"""
        if not reranked_child_chunks:
            return [], ""
        
        try:
            # Extract unique parent IDs
            parent_ids = []
            for chunk in reranked_child_chunks:
                parent_id = chunk.metadata.get('parent_id')
                if parent_id and parent_id not in parent_ids:
                    parent_ids.append(parent_id)
            
            if not parent_ids:
                return [], ""
            
            logging.info(f"Fetching {len(parent_ids)} parent chunks from MongoDB")
            
            # Fetch parent chunks from MongoDB
            parent_chunks_cursor = self.collection.find({"metadata.chunk_id": {"$in": parent_ids}})
            
            parent_chunks_map = {chunk['metadata']['chunk_id']: chunk for chunk in parent_chunks_cursor}
            
            final_context_chunks = []
            for pid in parent_ids:
                chunk_data = parent_chunks_map.get(pid)
                if chunk_data:
                    chunk = Document(
                        id=chunk_data['metadata']['chunk_id'],
                        text=chunk_data['text'],
                        metadata=chunk_data['metadata']
                    )
                    # Preserve rerank scores from child chunks
                    matching_child = next((c for c in reranked_child_chunks if c.metadata.get('parent_id') == pid), None)
                    if matching_child:
                        chunk.metadata.update({
                            'rerank_score': matching_child.metadata.get('rerank_score', 0),
                            'query_type': matching_child.metadata.get('query_type', 'general')
                        })
                    final_context_chunks.append(chunk)
            
            if not final_context_chunks:
                return [], ""
            
            logging.info(f"Retrieved {len(final_context_chunks)} parent chunks")
            
            # Apply semantic deduplication
            deduplicated_chunks = self._semantic_deduplication(final_context_chunks)
            logging.info(f"After deduplication: {len(deduplicated_chunks)} chunks")
            
            # Sort chunks by document order
            sorted_chunks = sorted(
                deduplicated_chunks, 
                key=lambda x: (x.metadata.get('page_no', 0), x.metadata.get('order_idx', 0))
            )
            
            # Assemble context with medical entity highlighting
            context_parts = []
            for i, chunk in enumerate(sorted_chunks):
                # Normalize medical terminology
                normalized_text = self._normalize_medical_terminology(chunk.text)
                
                # Highlight medical entities
                highlighted_text = self._highlight_medical_entities(normalized_text)
                
                # Add chunk with metadata
                chunk_header = f"Chunk {i+1}"
                context_parts.append(f"{chunk_header}: {highlighted_text}")
                
            assembled_context = "\n\n".join(context_parts)
            
            total_entities = len(self._extract_medical_entities(assembled_context))
            logging.info(f"Final assembled context: {len(sorted_chunks)} chunks, {total_entities} medical entities")
            
            return sorted_chunks, assembled_context
            
        except Exception as e:
            logging.error(f"Context assembly failed: {e}")
            return [], "Context assembly failed due to technical error."

# Backward compatibility
ContextAssembler = MedicalContextAssembler
