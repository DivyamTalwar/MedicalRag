"""
LEGENDARY Medical Text Chunking Strategy
Advanced, context-aware chunking for medical documents
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import logging

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkType(Enum):
    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    LAB_RESULTS = "lab_results"
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    HISTORY = "history"
    EXAMINATION = "examination"
    IMAGING = "imaging"
    NARRATIVE = "narrative"
    TABLE = "table"
    MIXED = "mixed"

@dataclass
class MedicalChunk:
    chunk_id: str
    content: str
    chunk_type: ChunkType
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    token_count: int = 0
    char_count: int = 0
    sentence_count: int = 0
    entities: List[Dict[str, Any]] = field(default_factory=list)
    semantic_density: float = 0.0
    coherence_score: float = 0.0
    importance_score: float = 0.0
    overlap_prev: Optional[str] = None
    overlap_next: Optional[str] = None
    relationships: List[Dict[str, Any]] = field(default_factory=list)

class SemanticChunker:
    def __init__(self, 
                 target_chunk_size: int = 512,
                 min_chunk_size: int = 128,
                 max_chunk_size: int = 1024,
                 overlap_size: int = 128,
                 semantic_threshold: float = 0.7):
        
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.semantic_threshold = semantic_threshold
        
        self.medical_section_markers = {
            'CHIEF COMPLAINT': ChunkType.HISTORY,
            'HISTORY OF PRESENT ILLNESS': ChunkType.HISTORY,
            'PAST MEDICAL HISTORY': ChunkType.HISTORY,
            'MEDICATIONS': ChunkType.MEDICATION,
            'ALLERGIES': ChunkType.HISTORY,
            'PHYSICAL EXAMINATION': ChunkType.EXAMINATION,
            'LABORATORY DATA': ChunkType.LAB_RESULTS,
            'IMAGING': ChunkType.IMAGING,
            'ASSESSMENT': ChunkType.DIAGNOSIS,
            'PLAN': ChunkType.TREATMENT,
            'DIAGNOSIS': ChunkType.DIAGNOSIS,
            'PROCEDURE': ChunkType.PROCEDURE,
            'OPERATIVE NOTE': ChunkType.PROCEDURE,
            'DISCHARGE SUMMARY': ChunkType.MIXED,
        }
        
        self.preserve_patterns = [
            r'(?:Diagnosis|DX|Dx):.*?(?=\n\n|\n[A-Z]|\Z)',
            r'(?:Medications?|MEDS?):.*?(?=\n\n|\n[A-Z]|\Z)', 
            r'(?:Lab(?:oratory)?\s+(?:Results?|Values?)):.*?(?=\n\n|\n[A-Z]|\Z)',
            r'(?:Vital Signs?):.*?(?=\n\n|\n[A-Z]|\Z)',
            r'(?:Physical Exam(?:ination)?):.*?(?=\n\n|\n[A-Z]|\Z)',
            r'\d+\.\s+.*?(?=\n\d+\.|\n\n|\Z)',  # Numbered lists
            r'(?:•|\*|-)\s+.*?(?=\n(?:•|\*|-)|\n\n|\Z)',  # Bullet lists
        ]
        
        self.critical_relationships = [
            ('diagnosis', 'treatment'),
            ('symptom', 'diagnosis'),
            ('lab_result', 'diagnosis'),
            ('medication', 'dosage'),
            ('procedure', 'outcome'),
            ('allergy', 'medication'),
        ]
    
    def _calculate_chunk_id(self, content: str, parent_id: Optional[str] = None) -> str:
        base = hashlib.md5(content.encode()).hexdigest()[:12]
        if parent_id:
            return f"{parent_id}_{base}"
        return base
    
    def _classify_chunk_type(self, text: str) -> ChunkType:
        text_upper = text.upper()
        
        for marker, chunk_type in self.medical_section_markers.items():
            if marker in text_upper:
                return chunk_type
        
        # Pattern-based classification
        medication_pattern = r'\b(?:mg|mcg|ml|tab|cap|dose|bid|tid|qid|prn)\b'
        lab_pattern = r'\b(?:WBC|RBC|Hgb|Hct|PLT|Na|K|Cl|CO2|BUN|Cr|Glucose)\b'
        procedure_pattern = r'\b(?:procedure|surgery|operation|incision|suture|anesthesia)\b'
        diagnosis_pattern = r'\b(?:diagnosed|diagnosis|assessment|impression|findings)\b'
        
        patterns = {
            ChunkType.MEDICATION: medication_pattern,
            ChunkType.LAB_RESULTS: lab_pattern,
            ChunkType.PROCEDURE: procedure_pattern,
            ChunkType.DIAGNOSIS: diagnosis_pattern,
        }
        
        scores = {}
        for chunk_type, pattern in patterns.items():
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            if matches > 0:
                scores[chunk_type] = matches
        
        if scores:
            return max(scores, key=scores.get)
        
        return ChunkType.NARRATIVE
    
    def _calculate_semantic_density(self, text: str) -> float:
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        medical_terms = re.findall(
            r'\b(?:[A-Z]{2,}|(?:mg|mcg|ml|mmHg|bpm|IU|mEq|ng|pg)/(?:dL|L|mL))\b',
            text
        )
        
        total_words = len(word_tokenize(text))
        if total_words == 0:
            return 0.0
        
        term_density = len(medical_terms) / total_words
        sentence_complexity = np.mean([len(word_tokenize(s)) for s in sentences])
        
        return min(1.0, term_density * 2 + sentence_complexity / 50)
    
    def _calculate_coherence_score(self, sentences: List[str]) -> float:
        if len(sentences) < 2:
            return 1.0
        
        # Simple coherence based on word overlap between consecutive sentences
        coherence_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(word_tokenize(sentences[i].lower()))
            words2 = set(word_tokenize(sentences[i + 1].lower()))
            
            # Remove common words
            stopwords = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 
                        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
                        'should', 'may', 'might', 'must', 'can', 'could'}
            words1 = words1 - stopwords
            words2 = words2 - stopwords
            
            if words1 and words2:
                overlap = len(words1.intersection(words2))
                min_len = min(len(words1), len(words2))
                coherence_scores.append(overlap / min_len if min_len > 0 else 0)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _calculate_importance_score(self, chunk: MedicalChunk) -> float:
        score = 0.0
        
        # Type-based importance
        type_weights = {
            ChunkType.DIAGNOSIS: 1.0,
            ChunkType.TREATMENT: 0.9,
            ChunkType.LAB_RESULTS: 0.8,
            ChunkType.MEDICATION: 0.8,
            ChunkType.PROCEDURE: 0.7,
            ChunkType.IMAGING: 0.7,
            ChunkType.EXAMINATION: 0.6,
            ChunkType.HISTORY: 0.5,
            ChunkType.NARRATIVE: 0.3,
            ChunkType.MIXED: 0.5,
            ChunkType.TABLE: 0.6,
        }
        score += type_weights.get(chunk.chunk_type, 0.3) * 0.3
        
        # Entity density
        entity_score = min(1.0, len(chunk.entities) / 10)
        score += entity_score * 0.2
        
        # Semantic density
        score += chunk.semantic_density * 0.2
        
        # Coherence
        score += chunk.coherence_score * 0.1
        
        # Critical information patterns
        critical_patterns = [
            r'\b(?:critical|severe|acute|emergency|urgent)\b',
            r'\b(?:abnormal|elevated|decreased|positive|negative)\b',
            r'\b(?:diagnosed?|confirmed|ruled out)\b',
            r'\b(?:allerg(?:y|ic)|contraindicated)\b',
        ]
        
        critical_count = sum(
            len(re.findall(pattern, chunk.content, re.IGNORECASE))
            for pattern in critical_patterns
        )
        critical_score = min(1.0, critical_count / 5)
        score += critical_score * 0.2
        
        return min(1.0, score)
    
    def _find_natural_boundaries(self, text: str) -> List[int]:
        boundaries = []
        
        # Sentence boundaries
        sentences = sent_tokenize(text)
        current_pos = 0
        for sentence in sentences:
            current_pos = text.find(sentence, current_pos) + len(sentence)
            boundaries.append(current_pos)
        
        # Section boundaries
        section_pattern = r'\n\n|\n(?=[A-Z][A-Z\s]+:)|(?<=\.)\s+(?=[A-Z][a-z]+\s+[A-Z][a-z]+:)'
        for match in re.finditer(section_pattern, text):
            boundaries.append(match.start())
        
        # List boundaries
        list_pattern = r'\n(?=\s*(?:\d+\.|\*|•|-)\s)'
        for match in re.finditer(list_pattern, text):
            boundaries.append(match.start())
        
        return sorted(set(boundaries))
    
    def _preserve_medical_context(self, text: str, start: int, end: int) -> Tuple[int, int]:
        # Adjust boundaries to preserve medical context
        
        # Don't split in the middle of a measurement
        measurement_pattern = r'\d+\.?\d*\s*(?:mg|mcg|mL|L|mmHg|bpm|%|units?)'
        for match in re.finditer(measurement_pattern, text):
            if match.start() < end < match.end():
                end = match.end()
        
        # Don't split medication instructions
        med_instruction_pattern = r'[A-Za-z]+\s+\d+\.?\d*\s*(?:mg|mcg|mL).*?(?:daily|bid|tid|qid|prn)'
        for match in re.finditer(med_instruction_pattern, text, re.IGNORECASE):
            if match.start() < end < match.end():
                end = match.end()
        
        # Keep diagnosis-treatment pairs together
        dx_tx_pattern = r'(?:Diagnosis|Assessment):.*?(?:Plan|Treatment):[^.]+\.'
        for match in re.finditer(dx_tx_pattern, text, re.IGNORECASE | re.DOTALL):
            if match.start() <= start < match.end() or match.start() < end <= match.end():
                start = min(start, match.start())
                end = max(end, match.end())
        
        return start, end
    
    def _create_overlap(self, chunks: List[MedicalChunk]):
        for i in range(len(chunks)):
            if i > 0:
                # Previous overlap
                prev_sentences = sent_tokenize(chunks[i-1].content)
                if prev_sentences:
                    overlap_sentences = prev_sentences[-2:] if len(prev_sentences) >= 2 else prev_sentences
                    chunks[i].overlap_prev = ' '.join(overlap_sentences)[:self.overlap_size]
            
            if i < len(chunks) - 1:
                # Next overlap
                next_sentences = sent_tokenize(chunks[i+1].content)
                if next_sentences:
                    overlap_sentences = next_sentences[:2] if len(next_sentences) >= 2 else next_sentences
                    chunks[i].overlap_next = ' '.join(overlap_sentences)[:self.overlap_size]
    
    def _identify_relationships(self, chunks: List[MedicalChunk]):
        for i, chunk in enumerate(chunks):
            relationships = []
            
            # Sequential relationships
            if i > 0:
                relationships.append({
                    'type': 'sequential_prev',
                    'target': chunks[i-1].chunk_id,
                    'strength': 1.0
                })
            if i < len(chunks) - 1:
                relationships.append({
                    'type': 'sequential_next',
                    'target': chunks[i+1].chunk_id,
                    'strength': 1.0
                })
            
            # Medical concept relationships
            for j, other_chunk in enumerate(chunks):
                if i != j:
                    # Check for diagnosis-treatment relationship
                    if (chunk.chunk_type == ChunkType.DIAGNOSIS and 
                        other_chunk.chunk_type == ChunkType.TREATMENT):
                        relationships.append({
                            'type': 'diagnosis_treatment',
                            'target': other_chunk.chunk_id,
                            'strength': 0.8
                        })
                    
                    # Check for lab-diagnosis relationship
                    if (chunk.chunk_type == ChunkType.LAB_RESULTS and 
                        other_chunk.chunk_type == ChunkType.DIAGNOSIS):
                        relationships.append({
                            'type': 'lab_diagnosis',
                            'target': other_chunk.chunk_id,
                            'strength': 0.7
                        })
                    
                    # Check for medication-related chunks
                    if (chunk.chunk_type == ChunkType.MEDICATION and 
                        other_chunk.chunk_type in [ChunkType.DIAGNOSIS, ChunkType.TREATMENT]):
                        relationships.append({
                            'type': 'medication_indication',
                            'target': other_chunk.chunk_id,
                            'strength': 0.6
                        })
            
            chunk.relationships = relationships
    
    def chunk_medical_document(self, 
                               text: str, 
                               doc_metadata: Optional[Dict[str, Any]] = None,
                               entities: Optional[List[Dict[str, Any]]] = None) -> List[MedicalChunk]:
        
        if not text or len(text.strip()) == 0:
            return []
        
        chunks = []
        
        # Try to preserve special medical patterns first
        preserved_sections = []
        remaining_text = text
        
        for pattern in self.preserve_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                preserved_sections.append({
                    'start': match.start(),
                    'end': match.end(),
                    'content': match.group(0),
                    'preserve': True
                })
        
        # Sort preserved sections by start position
        preserved_sections.sort(key=lambda x: x['start'])
        
        # Process text in segments
        current_pos = 0
        for section in preserved_sections:
            # Process text before preserved section
            if current_pos < section['start']:
                segment = text[current_pos:section['start']]
                if segment.strip():
                    segment_chunks = self._chunk_segment(segment, doc_metadata, entities)
                    chunks.extend(segment_chunks)
            
            # Add preserved section as single chunk if it fits size constraints
            preserved_content = section['content'].strip()
            if preserved_content:
                if len(preserved_content.split()) <= self.max_chunk_size:
                    chunk = self._create_chunk(preserved_content, doc_metadata, entities)
                    chunks.append(chunk)
                else:
                    # If too large, apply smart chunking
                    segment_chunks = self._chunk_segment(preserved_content, doc_metadata, entities)
                    chunks.extend(segment_chunks)
            
            current_pos = section['end']
        
        # Process remaining text
        if current_pos < len(text):
            segment = text[current_pos:]
            if segment.strip():
                segment_chunks = self._chunk_segment(segment, doc_metadata, entities)
                chunks.extend(segment_chunks)
        
        # If no preserved sections, chunk entire text
        if not chunks:
            chunks = self._chunk_segment(text, doc_metadata, entities)
        
        # Create overlaps
        self._create_overlap(chunks)
        
        # Identify relationships
        self._identify_relationships(chunks)
        
        # Calculate importance scores
        for chunk in chunks:
            chunk.importance_score = self._calculate_importance_score(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from document")
        return chunks
    
    def _chunk_segment(self, 
                      text: str, 
                      doc_metadata: Optional[Dict[str, Any]] = None,
                      entities: Optional[List[Dict[str, Any]]] = None) -> List[MedicalChunk]:
        
        chunks = []
        sentences = sent_tokenize(text)
        
        if not sentences:
            return chunks
        
        current_chunk_sentences = []
        current_chunk_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(word_tokenize(sentence))
            
            # Check if adding this sentence would exceed max size
            if current_chunk_tokens + sentence_tokens > self.max_chunk_size and current_chunk_sentences:
                # Create chunk from accumulated sentences
                chunk_content = ' '.join(current_chunk_sentences)
                chunk = self._create_chunk(chunk_content, doc_metadata, entities)
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk_sentences[-2:] if len(current_chunk_sentences) >= 2 else []
                current_chunk_sentences = overlap_sentences + [sentence]
                current_chunk_tokens = sum(len(word_tokenize(s)) for s in current_chunk_sentences)
            else:
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens
                
                # Check if we've reached target size with good coherence
                if current_chunk_tokens >= self.target_chunk_size:
                    coherence = self._calculate_coherence_score(current_chunk_sentences)
                    
                    # If coherence is good or we're at a natural boundary, create chunk
                    if coherence >= self.semantic_threshold or sentence.endswith(('.', '!', '?')):
                        chunk_content = ' '.join(current_chunk_sentences)
                        chunk = self._create_chunk(chunk_content, doc_metadata, entities)
                        chunks.append(chunk)
                        
                        # Start new chunk with overlap
                        overlap_sentences = current_chunk_sentences[-2:] if len(current_chunk_sentences) >= 2 else []
                        current_chunk_sentences = overlap_sentences
                        current_chunk_tokens = sum(len(word_tokenize(s)) for s in current_chunk_sentences)
        
        # Create final chunk if there's remaining content
        if current_chunk_sentences and len(current_chunk_sentences) > 2:  # Avoid duplicate from overlap
            chunk_content = ' '.join(current_chunk_sentences)
            if len(word_tokenize(chunk_content)) >= self.min_chunk_size:
                chunk = self._create_chunk(chunk_content, doc_metadata, entities)
                chunks.append(chunk)
        
        return chunks
    
    def _create_chunk(self, 
                     content: str, 
                     doc_metadata: Optional[Dict[str, Any]] = None,
                     entities: Optional[List[Dict[str, Any]]] = None) -> MedicalChunk:
        
        chunk_id = self._calculate_chunk_id(content)
        chunk_type = self._classify_chunk_type(content)
        
        # Extract entities that belong to this chunk
        chunk_entities = []
        if entities:
            content_lower = content.lower()
            for entity in entities:
                if entity.get('text', '').lower() in content_lower:
                    chunk_entities.append(entity)
        
        # Calculate metrics
        sentences = sent_tokenize(content)
        
        chunk = MedicalChunk(
            chunk_id=chunk_id,
            content=content,
            chunk_type=chunk_type,
            metadata=doc_metadata or {},
            token_count=len(word_tokenize(content)),
            char_count=len(content),
            sentence_count=len(sentences),
            entities=chunk_entities,
            semantic_density=self._calculate_semantic_density(content),
            coherence_score=self._calculate_coherence_score(sentences)
        )
        
        return chunk

class HierarchicalMedicalChunker:
    def __init__(self, base_chunker: SemanticChunker):
        self.base_chunker = base_chunker
        self.summary_max_length = 256
    
    def create_hierarchical_chunks(self, 
                                   text: str,
                                   doc_metadata: Optional[Dict[str, Any]] = None,
                                   entities: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        
        # Create base chunks
        base_chunks = self.base_chunker.chunk_medical_document(text, doc_metadata, entities)
        
        # Group chunks into semantic clusters
        clusters = self._create_semantic_clusters(base_chunks)
        
        # Create parent chunks from clusters
        parent_chunks = []
        for cluster_id, cluster_chunks in clusters.items():
            parent_chunk = self._create_parent_chunk(cluster_chunks, cluster_id)
            parent_chunks.append(parent_chunk)
            
            # Link children to parent
            for chunk in cluster_chunks:
                chunk.parent_id = parent_chunk.chunk_id
                parent_chunk.child_ids.append(chunk.chunk_id)
        
        # Create document-level summary
        doc_summary = self._create_document_summary(parent_chunks, doc_metadata)
        
        return {
            'document_summary': doc_summary,
            'parent_chunks': parent_chunks,
            'child_chunks': base_chunks,
            'hierarchy_depth': 3,
            'total_chunks': len(base_chunks) + len(parent_chunks) + 1
        }
    
    def _create_semantic_clusters(self, chunks: List[MedicalChunk]) -> Dict[str, List[MedicalChunk]]:
        clusters = {}
        current_cluster_id = 0
        current_cluster = []
        current_cluster_type = None
        
        for chunk in chunks:
            # Start new cluster if type changes significantly
            if current_cluster_type is None:
                current_cluster_type = chunk.chunk_type
                current_cluster = [chunk]
            elif chunk.chunk_type == current_cluster_type or chunk.chunk_type == ChunkType.MIXED:
                current_cluster.append(chunk)
            else:
                # Save current cluster
                clusters[f"cluster_{current_cluster_id}"] = current_cluster
                current_cluster_id += 1
                
                # Start new cluster
                current_cluster_type = chunk.chunk_type
                current_cluster = [chunk]
        
        # Save final cluster
        if current_cluster:
            clusters[f"cluster_{current_cluster_id}"] = current_cluster
        
        return clusters
    
    def _create_parent_chunk(self, child_chunks: List[MedicalChunk], cluster_id: str) -> MedicalChunk:
        # Combine content from child chunks
        combined_content = '\n'.join([chunk.content[:100] + '...' for chunk in child_chunks])
        
        # Determine parent chunk type
        chunk_types = [chunk.chunk_type for chunk in child_chunks]
        most_common_type = max(set(chunk_types), key=chunk_types.count)
        
        # Aggregate entities
        all_entities = []
        for chunk in child_chunks:
            all_entities.extend(chunk.entities)
        
        # Calculate aggregated metrics
        avg_semantic_density = np.mean([chunk.semantic_density for chunk in child_chunks])
        avg_coherence = np.mean([chunk.coherence_score for chunk in child_chunks])
        avg_importance = np.mean([chunk.importance_score for chunk in child_chunks])
        
        parent_chunk = MedicalChunk(
            chunk_id=f"parent_{cluster_id}",
            content=combined_content[:self.summary_max_length],
            chunk_type=most_common_type,
            metadata={'cluster_id': cluster_id, 'num_children': len(child_chunks)},
            token_count=sum(chunk.token_count for chunk in child_chunks),
            char_count=sum(chunk.char_count for chunk in child_chunks),
            sentence_count=sum(chunk.sentence_count for chunk in child_chunks),
            entities=all_entities[:20],  # Limit entities
            semantic_density=avg_semantic_density,
            coherence_score=avg_coherence,
            importance_score=avg_importance
        )
        
        return parent_chunk
    
    def _create_document_summary(self, 
                                 parent_chunks: List[MedicalChunk],
                                 doc_metadata: Optional[Dict[str, Any]] = None) -> MedicalChunk:
        
        # Extract key information from parent chunks
        key_info = []
        for chunk in sorted(parent_chunks, key=lambda x: x.importance_score, reverse=True)[:5]:
            key_info.append(f"[{chunk.chunk_type.value}]: {chunk.content[:50]}...")
        
        summary_content = '\n'.join(key_info)
        
        doc_summary = MedicalChunk(
            chunk_id=f"doc_summary_{hashlib.md5(summary_content.encode()).hexdigest()[:8]}",
            content=summary_content,
            chunk_type=ChunkType.MIXED,
            metadata=doc_metadata or {},
            child_ids=[chunk.chunk_id for chunk in parent_chunks],
            token_count=len(word_tokenize(summary_content)),
            char_count=len(summary_content),
            sentence_count=len(sent_tokenize(summary_content)),
            entities=[],
            semantic_density=1.0,
            coherence_score=1.0,
            importance_score=1.0
        )
        
        return doc_summary

if __name__ == "__main__":
    # Example usage
    sample_text = """
    CHIEF COMPLAINT: Chest pain and shortness of breath
    
    HISTORY OF PRESENT ILLNESS: The patient is a 65-year-old male who presents with 
    acute onset chest pain that started 2 hours ago. The pain is described as crushing, 
    radiating to the left arm. Associated with diaphoresis and nausea.
    
    MEDICATIONS:
    - Aspirin 81mg daily
    - Lisinopril 10mg daily
    - Metoprolol 25mg BID
    - Atorvastatin 40mg daily
    
    LABORATORY DATA:
    - Troponin I: 2.5 ng/mL (elevated)
    - CK-MB: 45 U/L (elevated)
    - BNP: 850 pg/mL
    - Creatinine: 1.2 mg/dL
    
    ASSESSMENT: Acute ST-elevation myocardial infarction (STEMI)
    
    PLAN: 
    1. Immediate cardiac catheterization
    2. Continue dual antiplatelet therapy
    3. Start heparin infusion
    4. Admit to CCU for monitoring
    """
    
    # Initialize chunkers
    semantic_chunker = SemanticChunker(
        target_chunk_size=256,
        overlap_size=50
    )
    hierarchical_chunker = HierarchicalMedicalChunker(semantic_chunker)
    
    # Create chunks
    result = hierarchical_chunker.create_hierarchical_chunks(
        sample_text,
        doc_metadata={'doc_type': 'emergency_note', 'department': 'ER'}
    )
    
    print(f"Created {result['total_chunks']} total chunks")
    print(f"Hierarchy depth: {result['hierarchy_depth']}")
    print(f"Parent chunks: {len(result['parent_chunks'])}")
    print(f"Child chunks: {len(result['child_chunks'])}")
    print("\nDocument Summary:")
    print(result['document_summary'].content)