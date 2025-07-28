import os
import sys
import asyncio
import logging
import math
import re
from uuid import uuid4, UUID
from collections import defaultdict
from typing import List, Dict, Any

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_pinecone import PineconeVectorStore
from app.core.embeddings import get_embedding_model
from langchain_core.documents import Document as LangchainDocument
from llama_index.core import SimpleDirectoryReader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException
from pymongo import MongoClient
from bson.codec_options import CodecOptions, UuidRepresentation
from app.models.data_models import DocumentChunk, EnhancedMedicalMetadata
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode, TextNode

load_dotenv()
nest_asyncio.apply()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
PARENT_INDEX_NAME = "parent"
CHILD_INDEX_NAME = "children"
MONGO_DB_NAME = "AdvanceRag"
DIMENSION = 1024

def create_pinecone_indexes(pinecone_client: Pinecone, vector_size: int):
    for index_name in [PARENT_INDEX_NAME, CHILD_INDEX_NAME]:
        try:
            if index_name not in pinecone_client.list_indexes().names():
                logging.info(f"Creating Pinecone index: '{index_name}'")
                pinecone_client.create_index(
                    name=index_name,
                    dimension=vector_size,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            else:
                logging.info(f"Pinecone index '{index_name}' already exists.")
        except Exception as e:
            logging.error(f"Failed to create/check Pinecone index '{index_name}': {e}")
            raise

def clear_pinecone_index(pinecone_client: Pinecone, index_name: str):
    try:
        if index_name in pinecone_client.list_indexes().names():
            index = pinecone_client.Index(index_name)
            index.delete(delete_all=True)
            logging.info(f"Cleared all vectors from index '{index_name}'.")
    except PineconeException:
        logging.warning(f"Index '{index_name}' was empty or namespace not found. Nothing to clear.")
    except Exception as e:
        logging.error(f"Failed to clear Pinecone index '{index_name}': {e}")


def clean_formatting_artifacts(text: str) -> str:
    original_length = len(text)
    
    html_entities = {
        '&#xA;': '\n', 
        '&amp;': '&',    
        '&lt;': '<',     
        '&gt;': '>',     
        '&quot;': '"',   
        '&#39;': "'", 
        '&nbsp;': ' ', 
        '&#8217;': "'",
        '&#8220;': '"', 
        '&#8221;': '"', 
        '&#8211;': '–', 
        '&#8212;': '—',
        '&#8226;': '•', 
        '&#174;': '®', 
        '&#169;': '©', 
        '&#8482;': '™'
    }
    
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)
    
    text = re.sub(r'\|\s*\|\s*\|+', '| | |', text)
    text = re.sub(r'\|\s*$', '|', text, flags=re.MULTILINE)
    
    text = re.sub(r'(\d+)\s*%', r'\1%', text)
    text = re.sub(r'\$\s*(\d)', r'$\1', text)
    text = re.sub(r'(\d+)\s*(minutes?|hrs?|hours?|days?)', r'\1 \2', text, flags=re.IGNORECASE)
    
    text = re.sub(r'\(\s*(\d{3})\s*\)\s*(\d{3})\s*-?\s*(\d{4})', r'(\1) \2-\3', text)
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()

def enhance_table_structure(text: str) -> str:
    lines = text.split('\n')
    enhanced_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        if '|' in line and line.count('|') >= 2:
            table_lines = []
            j = i
            while j < len(lines) and ('|' in lines[j] or lines[j].strip() == ''):
                if '|' in lines[j]:
                    table_lines.append(lines[j])
                j += 1
            
            if len(table_lines) >= 2:
                processed_table = process_complete_table(table_lines)
                enhanced_lines.extend(processed_table)
                i = j
            else:
                enhanced_lines.append(line)
                i += 1
        else:
            enhanced_lines.append(line)
            i += 1
    
    return '\n'.join(enhanced_lines)

def process_complete_table(table_lines: List[str]) -> List[str]:
    processed = []
    
    if table_lines:
        header = table_lines[0]
        cells = [cell.strip() for cell in header.split('|')]
        num_cols = len(cells) - 2
        
        processed.append('| ' + ' | '.join(cells[1:-1]) + ' |')
        
        if len(table_lines) < 2 or not re.match(r'\s*\|(\s*[-:]+\s*\|)+\s*', table_lines[1]):
            separator = '|' + '|'.join(['---' for _ in range(num_cols)]) + '|'
            processed.append(separator)
        
        for line in table_lines[1:]:
            if not re.match(r'\s*\|(\s*[-:]+\s*\|)+\s*', line):
                cells = [cell.strip() for cell in line.split('|')]
                while len(cells) < num_cols + 2:
                    cells.append('')
                processed.append('| ' + ' | '.join(cells[1:num_cols+1]) + ' |')
            else:
                processed.append(line)
    
    return processed


class MedicalEntityExtractor:    
    def __init__(self):
        self.medical_patterns = {
            'measurements': r'(\d+(?:\.\d+)?)\s*(minutes?|hours?|days?|%|mm|cm|mg|ml)',
            'phone_numbers': r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'medical_codes': r'[A-Z]{2,}-\d{4,}',
            'dates': r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            'times': r'\d{1,2}:\d{2}(?:\s*[AP]M)?',
            'radiologist_names': r'Dr\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            'medical_facilities': r'(Hospital|Medical Center|Clinic|RadPod|CIVIE)',
            'tat_data': r'(\d+)\s*(min|minutes?|hrs?|hours?)',
            'percentages': r'(\d+(?:\.\d+)?)\s*%',
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        entities = {}
        
        for entity_type, pattern in self.medical_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                if matches and isinstance(matches[0], tuple):
                    entities[entity_type] = [' '.join(match) if isinstance(match, tuple) else match for match in matches]
                else:
                    entities[entity_type] = matches
            else:
                entities[entity_type] = []
        
        return entities
    
class MedicalSemanticChunker():    
    def __init__(self, parent_chunk_size: int = 2000,child_chunk_size: int = 350,chunk_overlap: int = 40):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.chunk_overlap = chunk_overlap
        self.entity_extractor = MedicalEntityExtractor()
    
    def _identify_semantic_boundaries(self, text: str) -> List[int]:
        boundaries = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if re.match(r'^#{1,3}\s+', line):
                boundaries.append(i)
            elif re.match(r'^(FINDINGS?|IMPRESSION|CONCLUSION|HISTORY|TECHNIQUE|COMPARISON):', line, re.IGNORECASE):
                boundaries.append(i)
            elif '|' in line and i > 0 and '|' not in lines[i-1]:
                boundaries.append(i)
            elif re.match(r'^(RadPod|CIVIE|RADPOD)', line):
                boundaries.append(i)
            elif re.match(r'^\s*[•\-\*]\s+', line) and i > 0 and not re.match(r'^\s*[•\-\*]\s+', lines[i-1]):
                boundaries.append(i)
        
        return boundaries
    
    def _classify_section_type(self, text: str) -> str:
        if re.search(r'^(FINDINGS?|IMPRESSION|CONCLUSION)', text, re.IGNORECASE):
            return "clinical"
        elif '|' in text:
            return "table"
        else:
            return "standard"

    def _extract_section_title(self, text: str) -> str:
        lines = text.strip().split('\n')
        first_line = lines[0].strip() if lines else ""
        
        markdown_match = re.match(r'^#{1,6}\s+(.+?)(?:\s+#{1,6}.*)?(?:\s+##.*)?$', first_line)
        if markdown_match:
            title = markdown_match.group(1).strip()
            title = re.split(r'\s+[-–—]\s+|\s+##\s+', title)[0]
            title_words = title.split()
            if len(title_words) > 8:
                return ' '.join(title_words[:8]) + "..."
            return title
        
        medical_match = re.match(r'^([A-Z][A-Z\s&]+?):', first_line)
        if medical_match:
            return medical_match.group(1).strip()
        
        company_match = re.match(r'^(RadPod|CIVIE|RADPOD)(?:\s+[-–—]\s+(.+?))?', first_line, re.IGNORECASE)
        if company_match:
            if company_match.group(2):
                return f"{company_match.group(1)} - {company_match.group(2)[:50]}"
            return company_match.group(1)
        
        if (re.match(r'^[A-Z][a-z].*[a-z]$', first_line) and 
            len(first_line.split()) <= 6 and 
            len(first_line) <= 50):
            return first_line
        
        if first_line:
            clean_title = re.sub(r'^#{1,6}\s*', '', first_line)
            clean_title = re.sub(r'[^\w\s&-]', ' ', clean_title)
            clean_title = re.sub(r'\s+', ' ', clean_title).strip()
            
            # Take first meaningful words
            words = clean_title.split()
            if words:
                if len(words) > 6:
                    return ' '.join(words[:6]) + "..."
                return clean_title
        
        return "Untitled Section"

    def create_parent_chunks(self, text: str, doc_metadata: Dict) -> List[TextNode]:
        boundaries = self._identify_semantic_boundaries(text)
        
        if not boundaries:
            return self._size_based_parent_chunks(text, doc_metadata)
        
        parent_chunks = []
        lines = text.split('\n')
        
        for i, start_boundary in enumerate(boundaries):
            end_boundary = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
            
            chunk_lines = lines[start_boundary:end_boundary]
            chunk_text = '\n'.join(chunk_lines).strip()
            
            if len(chunk_text) > 50:
                entities = self.entity_extractor.extract_entities(chunk_text)
                
                section_type = self._classify_section_type(chunk_text)
                
                parent_metadata = EnhancedMedicalMetadata(
                    doc_id=doc_metadata['doc_id'],
                    parent_id=None,
                    pdf_name=doc_metadata['pdf_name'],
                    page_no=doc_metadata.get('page_no', 1),
                    order_idx=i,
                    chunk_type=f"parent_{section_type}",
                    section_title=self._extract_section_title(chunk_text),
                    medical_entities=self._flatten_entities(entities),
                    numerical_data=self._extract_numerical_context(entities),
                    contains_phi=self._detect_phi(chunk_text),
                    references_table='|' in chunk_text,
                    table_type=self._classify_table_type(chunk_text) if '|' in chunk_text else None,
                    primary_topics=self._extract_topics(chunk_text),
                    searchable_terms=self._create_searchable_terms(entities, chunk_text)
                )
                
                parent_chunks.append(TextNode(
                    text=chunk_text,
                    metadata=parent_metadata.model_dump()
                ))
        
        return parent_chunks

    def _size_based_parent_chunks(self, text: str, doc_metadata: Dict) -> List[TextNode]:
        chunks = []
        for i in range(0, len(text), self.parent_chunk_size):
            chunk_text = text[i:i+self.parent_chunk_size]
            parent_metadata = EnhancedMedicalMetadata(
                doc_id=doc_metadata['doc_id'],
                parent_id=None,
                pdf_name=doc_metadata['pdf_name'],
                page_no=doc_metadata.get('page_no', 1),
                order_idx=i,
                chunk_type="parent_standard",
                section_title="Untitled Section"
            )
            chunks.append(TextNode(text=chunk_text, metadata=parent_metadata.model_dump()))
        return chunks

    def _flatten_entities(self, entities: Dict) -> List[str]:
        return [item for sublist in entities.values() for item in sublist if item]

    def _extract_numerical_context(self, entities: Dict) -> List[Dict]:
        numerical_data = []
        for entity_type in ['measurements', 'tat_data', 'percentages']:
            for value in entities.get(entity_type, []):
                numerical_data.append({'value': value, 'type': entity_type})
        return numerical_data

    def _detect_phi(self, text: str) -> bool:
        phi_patterns = [
            r'Dr\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',
            r'\b[A-Z][a-z]+,?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            r'\(\d{3}\)\s*\d{3}-\d{4}',
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\b\d{3}-\d{2}-\d{4}\b', 
            r'\b[A-Z]{2}\d{8}\b',
            r'\b\d{4}\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b',
        ]
        
        phi_score = 0
        for pattern in phi_patterns:
            matches = len(re.findall(pattern, text))
            phi_score += matches
        
        return phi_score >= 1
    
    def _classify_table_type(self, text: str) -> str:
        text_lower = text.lower()
        if any(term in text_lower for term in ['tat', 'turnaround', 'minutes', 'hours', 'response time']):
            return 'performance_metrics'
        elif any(term in text_lower for term in ['patient', 'demographics', 'age', 'gender']):
            return 'patient_demographics'  
        elif any(term in text_lower for term in ['radiologist', 'physician', 'doctor', 'md', 'staff']):
            return 'medical_staff'
        elif any(term in text_lower for term in ['diagnosis', 'findings', 'impression', 'results']):
            return 'clinical_findings'
        elif any(term in text_lower for term in ['billing', 'cost', 'revenue', 'payment', '$']):
            return 'financial_data'
        elif any(term in text_lower for term in ['ct', 'mri', 'ultrasound', 'x-ray', 'modality']):
            return 'imaging_modalities'
        else:
            return 'general_medical'

    def _extract_topics(self, text: str) -> List[str]:
        topics = []
        
        topics.append(self._classify_section_type(text))
        
        if 'radpod' in text.lower() or 'civie' in text.lower():
            topics.append('company_info')
        if '|' in text:
            topics.append('tabular_data')
        if re.search(r'\d+\s*(minutes?|hours?)', text, re.IGNORECASE):
            topics.append('performance_metrics')
        
        return list(set(topics))

    def _create_searchable_terms(self, entities: Dict, text: str) -> List[str]:
        terms = self._flatten_entities(entities)
        
        medical_keywords = re.findall(r'\b(radiology|imaging|scan|CT|MRI|ultrasound|X-ray|diagnosis|findings|impression|RadPod|CIVIE)\b', text, re.IGNORECASE)
        terms.extend([term.lower() for term in medical_keywords])
        
        return list(set(terms))



class ContextAwareChildChunker:    
    def __init__(self, child_chunk_size: int = 350):
        self.child_chunk_size = child_chunk_size

    def create_child_chunks(self, parent_node: TextNode) -> List[TextNode]:
        """Create child chunks with intelligent medical context overlap"""
        text = parent_node.text
        parent_metadata = parent_node.metadata
        
        if self._is_table_section(text):
            return self._chunk_table_section(text, parent_metadata)
        
        if self._is_clinical_section(text):
            return self._chunk_clinical_section(text, parent_metadata)
        
        return self._semantic_chunk_standard(text, parent_metadata)

    def _is_table_section(self, text: str) -> bool:
        return text.count('|') > 10

    def _is_clinical_section(self, text: str) -> bool:
        return bool(re.search(r'^(FINDINGS?|IMPRESSION|CONCLUSION)', text, re.IGNORECASE))

    def _chunk_table_section(self, text: str, parent_metadata: Dict) -> List[TextNode]:
        return self._create_child_nodes([text], parent_metadata)

    def _chunk_clinical_section(self, text: str, parent_metadata: Dict) -> List[TextNode]:
        delimiters = [
            r'\n(?=FINDINGS?:)',
            r'\n(?=IMPRESSION:)',
            r'\n(?=CONCLUSION:)',
            r'\n(?=\d+\.)',  
            r'\n(?=[A-Z][^a-z]{10,}:)'
        ]
        
        chunks = []
        current_chunk = ""
        
        for line in text.split('\n'):
            if len(current_chunk) + len(line) > self.child_chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                context = self._extract_medical_context(current_chunk)
                current_chunk = context + line
            else:
                current_chunk += '\n' + line
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return self._create_child_nodes(chunks, parent_metadata)

    def _semantic_chunk_standard(self, text: str, parent_metadata: Dict) -> List[TextNode]:
        chunks = []
        for i in range(0, len(text), self.child_chunk_size):
            chunks.append(text[i:i+self.child_chunk_size])
        return self._create_child_nodes(chunks, parent_metadata)

    def _extract_medical_context(self, text: str) -> str:
        match = re.search(r'^(FINDINGS?|IMPRESSION|CONCLUSION):', text, re.IGNORECASE)
        if match:
            return match.group(0) + " "
        return ""

    def _create_child_nodes(self, chunks: List[str], parent_metadata: Dict) -> List[TextNode]:
        nodes = []
        for i, chunk_text in enumerate(chunks):
            child_metadata = parent_metadata.copy()
            child_metadata.update({
                "order_idx": i + 1,
                "chunk_type": "child_chunk"
            })
            nodes.append(TextNode(text=chunk_text, metadata=child_metadata))
        return nodes

def merge_extraction_results(results: Dict[str, Any]) -> List[Any]:
    merged_nodes = []
    
    logging.info(f"Merge function received keys: {list(results.keys())}")
    
    files = set()
    for key in results.keys():
        if key.endswith('_standard'):
            files.add(key[:-9]) 
        elif key.endswith('_tables'):
            files.add(key[:-7]) 
    
    logging.info(f"Identified files for merging: {files}")
    
    for file_base in files:
        standard_key = f"{file_base}_standard"
        tables_key = f"{file_base}_tables"
        
        standard_nodes = results.get(standard_key, [])
        table_nodes = results.get(tables_key, [])
        
        logging.info(f"Processing {file_base}:")
        logging.info(f"  Standard nodes: {len(standard_nodes)}")
        logging.info(f"  Table nodes: {len(table_nodes)}")
        
        if not standard_nodes and not table_nodes:
            logging.warning(f"No nodes found for {file_base}")
            continue
        
        if not standard_nodes:
            logging.info(f"Using only table nodes for {file_base}")
            merged_nodes.extend(table_nodes)
            continue
        
        if not table_nodes:
            logging.info(f"Using only standard nodes for {file_base}")
            merged_nodes.extend(standard_nodes)
            continue
        
        final_nodes = []
        max_nodes = max(len(standard_nodes), len(table_nodes))
        
        for i in range(max_nodes):
            std_node = standard_nodes[i] if i < len(standard_nodes) else None
            table_node = table_nodes[i] if i < len(table_nodes) else None
            
            if not std_node:
                final_nodes.append(table_node)
                continue
            if not table_node:
                final_nodes.append(std_node)
                continue
            
            try:
                std_text = std_node.get_content()
                table_text = table_node.get_content()
                
                if '|' in std_text or '|' in table_text:
                    std_table_cells = std_text.count('|')
                    table_table_cells = table_text.count('|')
                    
                    std_empty = std_text.count('| |') + std_text.count('||')
                    table_empty = table_text.count('| |') + table_text.count('||')
                    
                    if table_table_cells > std_table_cells or (table_table_cells == std_table_cells and table_empty < std_empty):
                        final_nodes.append(table_node)
                        logging.debug(f"Used table version for node {i} (better table data)")
                    else:
                        final_nodes.append(std_node)
                        logging.debug(f"Used standard version for node {i}")
                else:
                    final_nodes.append(std_node)
                    logging.debug(f"Used standard version for node {i} (no tables)")
                    
            except Exception as e:
                logging.warning(f"Error comparing nodes {i}: {e}, using standard")
                final_nodes.append(std_node)
        
        logging.info(f"Merged {len(final_nodes)} nodes for {file_base}")
        merged_nodes.extend(final_nodes)
    
    logging.info(f"Final merge result: {len(merged_nodes)} total nodes")
    return merged_nodes
async def multi_pass_processing(pdf_files: List[str]) -> List[Any]:
    parser_standard = LlamaParse(
        api_key=os.getenv("llama_index_api_key"),
        result_type="markdown",
        premium_mode=True,
        table_structure_recognition=True,
    )
    
    parser_tables = LlamaParse(
        api_key=os.getenv("llama_index_api_key"),
        result_type="markdown",
        premium_mode=True,
        table_structure_recognition=True,
        complemental_formatting_instruction="Focus exclusively on extracting complete table data with all cell values populated."
    )
    
    results = {}
    
    for pdf_file in pdf_files:
        logging.info(f"Multi-pass processing: {pdf_file}")
        
        try:
            standard_result = await asyncio.to_thread(
                SimpleDirectoryReader(input_files=[pdf_file], file_extractor={".pdf": parser_standard}).load_data
            )
            logging.info(f"Standard pass extracted {len(standard_result)} nodes from {pdf_file}")
            for i, node in enumerate(standard_result):
                logging.info(f"   Node {i}: {len(node.get_content())} chars")
            results[f"{pdf_file}_standard"] = standard_result
        except Exception as e:
            logging.error(f"Standard pass failed for {pdf_file}: {e}")
        
        # Table-focused pass
        try:
            table_result = await asyncio.to_thread(
                SimpleDirectoryReader(input_files=[pdf_file], file_extractor={".pdf": parser_tables}).load_data
            )
            logging.info(f"Table pass extracted {len(table_result)} nodes from {pdf_file}")
            results[f"{pdf_file}_tables"] = table_result
        except Exception as e:
            logging.error(f"Table pass failed for {pdf_file}: {e}")
    
    logging.info(f"Results dictionary keys: {list(results.keys())}")
    for key, value in results.items():
        logging.info(f"  {key}: {len(value)} nodes")
    
    merged = merge_extraction_results(results)
    logging.info(f"After merging: {len(merged)} total nodes")
    return merged

async def robust_extraction_with_retry(pdf_files: List[str], max_retries: int = 3) -> List[Any]:
    all_nodes = []
    
    fallback_parser = LlamaParse(
        api_key=os.getenv("llama_index_api_key"),
        result_type="markdown",
        premium_mode=True,
    )

    for pdf_file in pdf_files:
        success = False
        for attempt in range(max_retries):
            try:
                logging.info(f"Processing {pdf_file} (attempt {attempt + 1}/{max_retries})")
                
                if attempt == 0:
                    nodes = await multi_pass_processing([pdf_file])
                else:
                    logging.warning(f"Falling back to single-pass processing for {pdf_file}")
                    nodes = await asyncio.to_thread(
                        SimpleDirectoryReader(input_files=[pdf_file], file_extractor={".pdf": fallback_parser}).load_data
                    )
                
                if nodes:
                    all_nodes.extend(nodes)
                    success = True
                    logging.info(f"Successfully processed {pdf_file}")
                    break
                else:
                    logging.warning(f"No nodes extracted for {pdf_file} on attempt {attempt + 1}")

            except Exception as e:
                logging.error(f"Attempt {attempt + 1} failed for {pdf_file}: {e}")
        
        if not success:
            logging.error(f"Failed to process {pdf_file} after all attempts")
    
    return all_nodes

def track_extraction_metrics(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chunks:
        logging.warning("No chunks provided for metrics tracking.")
        return {}

    num_parent_chunks = sum(1 for c in chunks if 'parent' in c['metadata']['chunk_type'])
    num_child_chunks = sum(1 for c in chunks if 'child' in c['metadata']['chunk_type'])
    
    total_word_count = sum(len(c['text'].split()) for c in chunks)
    avg_word_count_per_chunk = total_word_count / len(chunks) if chunks else 0
    
    chunks_with_tables = sum(1 for c in chunks if '|' in c['text'])
    chunks_with_numbers = sum(1 for c in chunks if re.search(r'\d+', c['text']))
    chunks_with_entities = sum(1 for c in chunks if c['metadata'].get('medical_entities'))
    chunks_with_phi = sum(1 for c in chunks if c['metadata'].get('contains_phi', False))
    
    table_chunks = [c for c in chunks if 'table' in c['metadata']['chunk_type']]
    table_types = {}
    for chunk in table_chunks:
        table_type = chunk['metadata'].get('table_type', 'unknown')
        table_types[table_type] = table_types.get(table_type, 0) + 1
    
    doc_ids = {c['metadata']['doc_id'] for c in chunks}
    
    all_entities = []
    for chunk in chunks:
        entities = chunk['metadata'].get('medical_entities', [])
        all_entities.extend(entities)
    
    unique_entities = len(set(all_entities))
    
    metrics = {
        "total_chunks": len(chunks),
        "parent_chunks": num_parent_chunks,
        "child_chunks": num_child_chunks,
        "total_documents_processed": len(doc_ids),
        "total_word_count": total_word_count,
        "avg_word_count_per_chunk": round(avg_word_count_per_chunk, 2),
        "chunks_with_tables": chunks_with_tables,
        "table_preservation_rate": round(chunks_with_tables / len(chunks) * 100, 1) if chunks else 0,
        "chunks_with_numerical_data": chunks_with_numbers,
        "chunks_with_medical_entities": chunks_with_entities,
        "chunks_with_phi": chunks_with_phi,
        "unique_medical_entities": unique_entities,
        "total_medical_entities": len(all_entities),
        "table_types_detected": table_types,
        "content_coverage": {
            "clinical_sections": sum(1 for c in chunks if 'clinical' in c['metadata']['chunk_type']),
            "table_sections": len(table_chunks),
            "standard_sections": sum(1 for c in chunks if 'standard' in c['metadata']['chunk_type'])
        }
    }
    
    logging.info("COMPREHENSIVE EXTRACTION METRICS:")
    for key, value in metrics.items():
        logging.info(f"   {key}: {value}")
    
    return metrics

async def main():
    logging.info("Starting ingestion with LlamaParse")

    try:
        mongo_client = MongoClient(os.getenv("MONGO_URI"))
        codec_options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
        db = mongo_client.get_database(MONGO_DB_NAME, codec_options=codec_options)
        collection = db.get_collection(MONGO_DB_NAME)
        logging.info(f"Successfully connected to MongoDB.")
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        return

    try:
        pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        logging.info("Successfully connected to Pinecone.")
    except Exception as e:
        logging.error(f"Failed to connect to Pinecone: {e}")
        return

    embeddings_model = get_embedding_model()
    create_pinecone_indexes(pinecone_client, DIMENSION)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    pdf_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        logging.warning("No PDF files found. Exiting.")
        return

    try:
        raw_nodes = await robust_extraction_with_retry(pdf_files)
        if not raw_nodes:
            logging.error("Robust extraction failed to produce any nodes. Exiting.")
            return
        logging.info(f"Successfully parsed {len(raw_nodes)} nodes using robust extraction.")
    except Exception as e:
        logging.error(f"Failed during robust extraction: {e}")
        return

    all_mongo_chunks = []
    parent_docs_for_pinecone = []
    child_docs_for_pinecone = []

    semantic_chunker = MedicalSemanticChunker()
    child_chunker = ContextAwareChildChunker()

    nodes_by_doc = defaultdict(list)
    for node in raw_nodes:
        nodes_by_doc[node.metadata.get("file_name")].append(node)

    for pdf_name, nodes in nodes_by_doc.items():
        doc_id = uuid4()
        logging.info(f"Processing document: {pdf_name} (ID: {doc_id})")
        
        cleaned_nodes = [clean_formatting_artifacts(node.get_content()) for node in nodes]
        full_text = "\n\n".join(cleaned_nodes)
        full_text = enhance_table_structure(full_text)
        
        doc_metadata = {"doc_id": doc_id, "pdf_name": pdf_name}
        parent_nodes = semantic_chunker.create_parent_chunks(full_text, doc_metadata)

        for parent_node in parent_nodes:
            parent_id = uuid4()
            parent_node.metadata["chunk_id"] = str(parent_id)
            
            parent_chunk = DocumentChunk(
                text=parent_node.text,
                metadata=EnhancedMedicalMetadata(**parent_node.metadata)
            )
            all_mongo_chunks.append(parent_chunk.model_dump())
            
            pinecone_parent_metadata = {k: str(v) for k, v in parent_chunk.metadata.model_dump().items() if v is not None}
            parent_docs_for_pinecone.append(LangchainDocument(page_content=parent_chunk.text, metadata=pinecone_parent_metadata))

            child_nodes = child_chunker.create_child_chunks(parent_node)
            for child_node in child_nodes:
                child_id = uuid4()
                child_node.metadata["chunk_id"] = str(child_id)
                child_node.metadata["parent_id"] = str(parent_id)

                child_chunk = DocumentChunk(
                    text=child_node.text,
                    metadata=EnhancedMedicalMetadata(**child_node.metadata)
                )
                all_mongo_chunks.append(child_chunk.model_dump())

                pinecone_child_metadata = {k: str(v) for k, v in child_chunk.metadata.model_dump().items() if v is not None}
                child_docs_for_pinecone.append(LangchainDocument(page_content=child_chunk.text, metadata=pinecone_child_metadata))

    logging.info(f"Created {len(parent_docs_for_pinecone)} parent chunks and {len(child_docs_for_pinecone)} child chunks.")

    if all_mongo_chunks:
        try:
            collection.delete_many({})
            collection.insert_many(all_mongo_chunks)
            logging.info(f"Inserted {len(all_mongo_chunks)} chunks into MongoDB.")
        except Exception as e:
            logging.error(f"Failed to insert data into MongoDB: {e}")
            return

    logging.info("Upserting vectors to Pinecone in batches")
    try:
        clear_pinecone_index(pinecone_client, PARENT_INDEX_NAME)
        clear_pinecone_index(pinecone_client, CHILD_INDEX_NAME)
        
        async def upsert_parents():
            if parent_docs_for_pinecone:
                logging.info(f"Upserting {len(parent_docs_for_pinecone)} parent vectors to '{PARENT_INDEX_NAME}'...")
                await asyncio.to_thread(PineconeVectorStore.from_documents, documents=parent_docs_for_pinecone, embedding=embeddings_model, index_name=PARENT_INDEX_NAME, batch_size=100)
                logging.info(f"Finished upserting to '{PARENT_INDEX_NAME}'.")

        async def upsert_children():
            if child_docs_for_pinecone:
                logging.info(f"Upserting {len(child_docs_for_pinecone)} child vectors to '{CHILD_INDEX_NAME}'...")
                await asyncio.to_thread(PineconeVectorStore.from_documents, documents=child_docs_for_pinecone, embedding=embeddings_model, index_name=CHILD_INDEX_NAME, batch_size=100)
                logging.info(f"Finished upserting to '{CHILD_INDEX_NAME}'.")

        results = await asyncio.gather(upsert_parents(), upsert_children(), return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logging.error(f"An exception occurred during Pinecone upsert: {result}")

    except Exception as e:
        logging.error(f"Failed to upsert vectors to Pinecone: {e}")
        return

    track_extraction_metrics(all_mongo_chunks)
    logging.info("Ingestion process finished successfully.")

if __name__ == "__main__":
    asyncio.run(main())
