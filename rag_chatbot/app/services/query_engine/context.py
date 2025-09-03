import os
import logging
import re
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
from difflib import SequenceMatcher
from rag_chatbot.app.models.data_models import Document
import pinecone

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from pymongo import MongoClient
from bson.codec_options import CodecOptions, UuidRepresentation
from ...core.config import MONGO_DB_NAME

class MedicalContextAssembler:
    def __init__(self):
        self.mongo_client = MongoClient(os.getenv("MONGO_URI"))
        codec_options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
        self.db = self.mongo_client.get_database(MONGO_DB_NAME, codec_options=codec_options)
        self.collection = self.db.get_collection(MONGO_DB_NAME)
        
        self.medical_patterns = {
            'lab_values': [
                r'\d+\.\d+\s*(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL|%)',
                r'\d+\s*-\s*\d+(?:\.\d+)?', 
                r'[<>]\s*\d+\.\d+'
            ],
            'medical_terms': [
                r'\b(?:HbA1c|TSH|HDL|LDL|CBC|ESR|WBC|RBC|T3|T4)\b',
                r'\b(?:pH|PCO2|PO2|HCO3|TCO2|SBC|BE)\b', 
                r'\b(?:Toxoplasma|Rubella|Cytomegalovirus|Herpes)\b', 
                r'\b(?:P2|P3)\s*(?:Peak|Window)\b', 
                r'\b(?:PACS|RIS|DICOM|TAT|SLA)\b'  
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
        
        self.terminology_normalizations = {
            'hba1c': ['glycosylated hemoglobin', 'glycated hemoglobin', 'hemoglobin a1c'],
            'tsh': ['thyroid stimulating hormone', 'thyrotropin'],
            'hdl': ['high density lipoprotein', 'good cholesterol'],
            'ldl': ['low density lipoprotein', 'bad cholesterol'],
            'cbc': ['complete blood count', 'full blood count'],
            'esr': ['erythrocyte sedimentation rate', 'sed rate'],
            
            'mg/dl': ['mg/dL', 'milligrams per deciliter'],
            'mmhg': ['mmHg', 'mm Hg', 'millimeters mercury'],
            'meq/l': ['mEq/L', 'milliequivalents per liter'],
            
            'positive': ['reactive', '+'],
            'negative': ['non-reactive', 'non reactive', '-'],
            'normal': ['within normal limits', 'wnl', 'reference range'],
            'abnormal': ['outside normal', 'elevated', 'decreased']
        }
        
        self.semantic_similarity_threshold = 0.85
        self.coherence_threshold = 0.6

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        normalized_text1 = self._normalize_medical_terminology(text1.lower())
        normalized_text2 = self._normalize_medical_terminology(text2.lower())
        
        sequence_similarity = SequenceMatcher(None, normalized_text1, normalized_text2).ratio()
        
        entities1 = self._extract_medical_entities(text1)
        entities2 = self._extract_medical_entities(text2)
        
        if entities1 or entities2:
            all_entities = entities1.union(entities2)
            common_entities = entities1.intersection(entities2)
            entity_similarity = len(common_entities) / len(all_entities) if all_entities else 0
        else:
            entity_similarity = 0
        
        final_similarity = (sequence_similarity * 0.7) + (entity_similarity * 0.3)
        return final_similarity

    def _extract_medical_entities(self, text: str) -> Set[str]:
        entities = set()
        
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                entities.update([match.lower().strip() for match in matches])
        
        return entities

    def _normalize_medical_terminology(self, text: str) -> str:
        normalized_text = text
        
        for standard_term, variations in self.terminology_normalizations.items():
            for variation in variations:
                pattern = r'\b' + re.escape(variation) + r'\b'
                normalized_text = re.sub(pattern, standard_term, normalized_text, flags=re.IGNORECASE)
        
        return normalized_text

    def _semantic_deduplication(self, chunks: List[Document]) -> List[Document]:
        if len(chunks) <= 1:
            return chunks
        
        deduplicated_chunks = []
        similarity_matrix = {}
        
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                similarity = self._calculate_semantic_similarity(chunk1.text, chunk2.text)
                similarity_matrix[(i, j)] = similarity
        
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
            
            if len(group) > 1:
                best_idx = self._select_best_representative(chunks, group)
                deduplicated_chunks.append(chunks[best_idx])
                logging.info(f"Deduplicated {len(group)} similar chunks, kept chunk {best_idx}")
            else:
                deduplicated_chunks.append(chunks[i])
            
            used_indices.add(i)
        
        return deduplicated_chunks

    def _select_best_representative(self, chunks: List[Document], group_indices: List[int]) -> int:
        best_idx = group_indices[0]
        best_score = 0
        
        for idx in group_indices:
            chunk = chunks[idx]
            
            score = 0
            
            score += len(chunk.text.split()) * 0.1
            
            entities = self._extract_medical_entities(chunk.text)
            score += len(entities) * 2
            
            rerank_score = chunk.metadata.get('rerank_score', 0)
            score += rerank_score * 10
            
            page_no = chunk.metadata.get('page_no', 0)
            score += (1.0 / (page_no + 1)) * 0.5
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        return best_idx

    def _highlight_medical_entities(self, text: str) -> str:
        highlighted_text = text
        
        highlights = []
        
        for category, patterns in self.medical_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start, end = match.span()
                    if not any(s <= start < e or s < end <= e for s, e in highlights):
                        highlights.append((start, end))
        
        highlights.sort(reverse=True)
        
        for start, end in highlights:
            entity = text[start:end]
            highlighted_text = highlighted_text[:start] + f"**{entity}**" + highlighted_text[end:]
        
        return highlighted_text

    def _calculate_context_quality_score(self, chunks: List[Document]) -> float:
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
        
        total_words = len(total_text.split())
        entity_density = len(total_entities) / total_words if total_words > 0 else 0
        quality_factors['medical_entity_density'] = min(1.0, entity_density * 50)  
        
        coherence_scores = []
        for i in range(len(chunks) - 1):
            similarity = self._calculate_semantic_similarity(chunks[i].text, chunks[i+1].text)
            coherence_scores.append(similarity)
        
        quality_factors['coherence_score'] = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0
        
        completeness_patterns = [
            r'\d+\s*-\s*\d+', 
            r'normal|reference',
            r'interpretation|conclusion' 
        ]
        completeness_matches = 0
        for pattern in completeness_patterns:
            if re.search(pattern, total_text, re.IGNORECASE):
                completeness_matches += 1
        
        quality_factors['completeness_score'] = completeness_matches / len(completeness_patterns)
        
        rerank_scores = [chunk.metadata.get('rerank_score', 0.5) for chunk in chunks]
        quality_factors['relevance_score'] = sum(rerank_scores) / len(rerank_scores)
        
        weights = {
            'medical_entity_density': 0.3,
            'coherence_score': 0.25,
            'completeness_score': 0.25,
            'relevance_score': 0.2
        }
        
        overall_score = sum(quality_factors[factor] * weights[factor] for factor in quality_factors)
        return min(1.0, max(0.0, overall_score))

    def _validate_context_coherence(self, chunks: List[Document]) -> Tuple[bool, List[str]]:
        if len(chunks) < 2:
            return True, []
        
        issues = []
        
        for i in range(len(chunks) - 1):
            current_entities = self._extract_medical_entities(chunks[i].text)
            next_entities = self._extract_medical_entities(chunks[i+1].text)
            
            similarity = self._calculate_semantic_similarity(chunks[i].text, chunks[i+1].text)
            if similarity < 0.1: 
                issues.append(f"Low coherence between chunk {i+1} and {i+2} (similarity: {similarity:.2f})")
            
            if self._detect_contradictory_values(chunks[i].text, chunks[i+1].text):
                issues.append(f"Potential contradictory values between chunk {i+1} and {i+2}")
        
        all_text = " ".join([chunk.text for chunk in chunks])
        if not self._has_sufficient_context(all_text):
            issues.append("Context may lack sufficient medical detail")
        
        coherence_valid = len(issues) == 0
        return coherence_valid, issues

    def _detect_contradictory_values(self, text1: str, text2: str) -> bool:
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
        
        for key in values1:
            if key in values2:
                diff_ratio = abs(values1[key] - values2[key]) / max(values1[key], values2[key])
                if diff_ratio > 0.5: 
                    return True
        
        return False

    def _has_sufficient_context(self, text: str) -> bool:
        entities = self._extract_medical_entities(text)
        
        min_entities = 3
        min_words = 100
        
        word_count = len(text.split())
        
        return len(entities) >= min_entities and word_count >= min_words

    def assemble(self, child_chunks: List[Document]) -> Tuple[List[Document], str]:
        if not child_chunks:
            return [], ""

        # Get parent documents
        parent_doc_ids = []
        for child_doc in child_chunks:
            try:
                parent_id = child_doc.id.split('_')[0]
                parent_doc_ids.append(parent_id)
            except (AttributeError, IndexError):
                logging.warning(f"Could not parse parent ID from child ID: {getattr(child_doc, 'id', 'N/A')}")

        unique_parent_ids = list(set(parent_doc_ids))
        
        parent_docs = []
        if unique_parent_ids:
            try:
                results = self.collection.find({"id": {"$in": unique_parent_ids}})
                doc_map = {doc['id']: Document(**doc) for doc in results}
                parent_docs = [doc_map[pid] for pid in unique_parent_ids if pid in doc_map]
            except Exception as e:
                logging.error(f"Failed to retrieve parent documents from MongoDB: {e}")

        context_str = "\n\n---\n\n".join([doc.text for doc in parent_docs])
        
        return parent_docs, context_str

    def assemble_comprehensive_context(
        self, 
        original_query: str, 
        original_query_results: List[Dict[str, Any]], 
        sub_query_results: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        context_str = "### Primary Context (from original query)\n"
        context_str += f"Original Query: {original_query}\n"
        if original_query_results:
            for i, result in enumerate(original_query_results):
                text = result.get('text', 'N/A')
                score = result.get('rerank_score', 0.0)
                context_str += f"{i+1}. (Score: {score:.2f}) {text}\n"
        else:
            context_str += "- No results found for the original query.\n"
        
        context_str += "\n### Supplementary Context (from sub-queries)\n"
        if sub_query_results:
            for sub_query, results in sub_query_results.items():
                context_str += f"Sub-query: {sub_query}\n"
                if results:
                    for i, result in enumerate(results):
                        text = result.get('text', 'N/A')
                        score = result.get('rerank_score', 0.0)
                        context_str += f"- (Score: {score:.2f}) {text}\n"
                else:
                    context_str += "- No results found for this sub-query.\n"
                context_str += "\n"
        else:
            context_str += "- No sub-queries were generated or they produced no results.\n"
            
        return context_str

ContextAssembler = MedicalContextAssembler
