"""
LEGENDARY Medical Data Ingestion Pipeline
Production-ready, scalable, medical-domain optimized
"""

import asyncio
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pytesseract
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentType(Enum):
    LAB_REPORT = "lab_report"
    CLINICAL_NOTE = "clinical_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    PRESCRIPTION = "prescription"
    IMAGING_REPORT = "imaging_report"
    PATHOLOGY_REPORT = "pathology_report"
    CONSULTATION = "consultation"
    OPERATIVE_NOTE = "operative_note"
    UNKNOWN = "unknown"

@dataclass
class MedicalEntity:
    text: str
    type: str
    confidence: float
    start_idx: int
    end_idx: int
    normalized_value: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[Tuple[float, float]] = None
    icd_codes: List[str] = field(default_factory=list)
    snomed_codes: List[str] = field(default_factory=list)

@dataclass
class MedicalDocument:
    doc_id: str
    content: str
    doc_type: DocumentType
    metadata: Dict[str, Any]
    entities: List[MedicalEntity]
    sections: Dict[str, str]
    tables: List[pd.DataFrame]
    checksum: str
    processed_at: datetime
    quality_score: float
    phi_redacted: bool = False

class MedicalNERExtractor:
    def __init__(self):
        self.biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
        self.biobert_model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1")
        self.ner_pipeline = pipeline(
            "ner",
            model=self.biobert_model,
            tokenizer=self.biobert_tokenizer,
            aggregation_strategy="simple"
        )
        
        try:
            self.spacy_model = spacy.load("en_core_sci_md")
        except:
            logger.warning("Medical spaCy model not found, using default")
            self.spacy_model = spacy.load("en_core_web_sm")
        
        self.medical_patterns = self._compile_medical_patterns()
    
    def _compile_medical_patterns(self) -> Dict[str, re.Pattern]:
        return {
            'lab_value': re.compile(
                r'(\w+[\w\s]*?)[\s:]+(\d+\.?\d*)\s*(mg/dL|mmol/L|g/dL|IU/mL|ng/mL|pg/mL|%|mmHg|mEq/L|U/L)',
                re.IGNORECASE
            ),
            'vital_sign': re.compile(
                r'(BP|Blood Pressure|HR|Heart Rate|RR|Respiratory Rate|Temp|Temperature|SpO2|O2 Sat)[\s:]+(\d+\.?\d*(?:/\d+\.?\d*)?)\s*(\w+)?',
                re.IGNORECASE
            ),
            'medication': re.compile(
                r'(\w+(?:\s+\w+)?)\s+(\d+\.?\d*)\s*(mg|mcg|g|mL|units?)\s*(?:(PO|IV|IM|SC|PR|SL|TD|INH|TOP))?',
                re.IGNORECASE
            ),
            'icd_code': re.compile(r'[A-Z]\d{2}(?:\.\d{1,2})?'),
            'date': re.compile(
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}'
            ),
            'measurement': re.compile(
                r'(\d+\.?\d*)\s*(?:x|×)\s*(\d+\.?\d*)\s*(?:x|×)?\s*(\d+\.?\d*)?\s*(cm|mm|inches?)',
                re.IGNORECASE
            )
        }
    
    def extract_entities(self, text: str) -> List[MedicalEntity]:
        entities = []
        
        # BioBERT NER
        try:
            bert_entities = self.ner_pipeline(text[:512])  # BERT token limit
            for ent in bert_entities:
                entities.append(MedicalEntity(
                    text=ent['word'],
                    type=ent['entity_group'],
                    confidence=ent['score'],
                    start_idx=ent['start'],
                    end_idx=ent['end']
                ))
        except Exception as e:
            logger.error(f"BioBERT NER failed: {e}")
        
        # spaCy NER
        doc = self.spacy_model(text)
        for ent in doc.ents:
            entities.append(MedicalEntity(
                text=ent.text,
                type=ent.label_,
                confidence=0.8,
                start_idx=ent.start_char,
                end_idx=ent.end_char
            ))
        
        # Pattern-based extraction
        for pattern_name, pattern in self.medical_patterns.items():
            for match in pattern.finditer(text):
                entity = MedicalEntity(
                    text=match.group(0),
                    type=pattern_name,
                    confidence=0.9,
                    start_idx=match.start(),
                    end_idx=match.end()
                )
                
                if pattern_name == 'lab_value' and len(match.groups()) >= 3:
                    entity.normalized_value = match.group(2)
                    entity.unit = match.group(3)
                    entity.reference_range = self._get_reference_range(match.group(1), match.group(3))
                
                entities.append(entity)
        
        return self._deduplicate_entities(entities)
    
    def _get_reference_range(self, test_name: str, unit: str) -> Optional[Tuple[float, float]]:
        reference_ranges = {
            ('glucose', 'mg/dL'): (70, 100),
            ('hemoglobin', 'g/dL'): (12, 16),
            ('wbc', 'cells/mcL'): (4500, 11000),
            ('platelet', 'cells/mcL'): (150000, 450000),
            ('creatinine', 'mg/dL'): (0.6, 1.2),
            ('cholesterol', 'mg/dL'): (0, 200),
            ('hdl', 'mg/dL'): (40, 60),
            ('ldl', 'mg/dL'): (0, 100),
            ('triglycerides', 'mg/dL'): (0, 150),
        }
        
        test_lower = test_name.lower()
        for (test, test_unit), range_vals in reference_ranges.items():
            if test in test_lower and unit.lower() == test_unit.lower():
                return range_vals
        return None
    
    def _deduplicate_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        seen = set()
        unique_entities = []
        for entity in sorted(entities, key=lambda x: x.confidence, reverse=True):
            key = (entity.text.lower(), entity.type, entity.start_idx)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        return unique_entities

class MedicalDocumentParser:
    def __init__(self):
        self.ner_extractor = MedicalNERExtractor()
        self.section_patterns = {
            'chief_complaint': re.compile(r'chief complaint[s]?[:]\s*(.*?)(?=\n[A-Z]|\n\n|\Z)', re.IGNORECASE | re.DOTALL),
            'history_present_illness': re.compile(r'history of present illness[:]\s*(.*?)(?=\n[A-Z]|\n\n|\Z)', re.IGNORECASE | re.DOTALL),
            'past_medical_history': re.compile(r'past medical history[:]\s*(.*?)(?=\n[A-Z]|\n\n|\Z)', re.IGNORECASE | re.DOTALL),
            'medications': re.compile(r'medication[s]?[:]\s*(.*?)(?=\n[A-Z]|\n\n|\Z)', re.IGNORECASE | re.DOTALL),
            'allergies': re.compile(r'allergies[:]\s*(.*?)(?=\n[A-Z]|\n\n|\Z)', re.IGNORECASE | re.DOTALL),
            'physical_exam': re.compile(r'physical exam(?:ination)?[:]\s*(.*?)(?=\n[A-Z]|\n\n|\Z)', re.IGNORECASE | re.DOTALL),
            'assessment_plan': re.compile(r'assessment(?:\s+and|\s*/\s*)?plan[:]\s*(.*?)(?=\n[A-Z]|\n\n|\Z)', re.IGNORECASE | re.DOTALL),
            'lab_results': re.compile(r'lab(?:oratory)?\s+results?[:]\s*(.*?)(?=\n[A-Z]|\n\n|\Z)', re.IGNORECASE | re.DOTALL),
            'imaging': re.compile(r'imaging\s+(?:results?|findings?)[:]\s*(.*?)(?=\n[A-Z]|\n\n|\Z)', re.IGNORECASE | re.DOTALL),
            'diagnosis': re.compile(r'diagnos[ie]s[:]\s*(.*?)(?=\n[A-Z]|\n\n|\Z)', re.IGNORECASE | re.DOTALL),
        }
    
    def classify_document(self, text: str, metadata: Dict[str, Any]) -> DocumentType:
        text_lower = text.lower()
        
        type_indicators = {
            DocumentType.LAB_REPORT: ['hemoglobin', 'hematocrit', 'wbc', 'platelet', 'glucose', 'creatinine'],
            DocumentType.DISCHARGE_SUMMARY: ['discharge summary', 'discharge diagnosis', 'hospital course'],
            DocumentType.PRESCRIPTION: ['rx', 'sig:', 'dispense', 'refills'],
            DocumentType.IMAGING_REPORT: ['ct scan', 'mri', 'x-ray', 'ultrasound', 'impression:', 'findings:'],
            DocumentType.PATHOLOGY_REPORT: ['microscopic', 'gross description', 'specimen', 'histologic'],
            DocumentType.CONSULTATION: ['consultation', 'referring physician', 'reason for consult'],
            DocumentType.OPERATIVE_NOTE: ['procedure:', 'anesthesia:', 'surgeon:', 'operative findings'],
        }
        
        scores = {}
        for doc_type, indicators in type_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            if score > 0:
                scores[doc_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        
        if 'clinical' in text_lower or 'patient' in text_lower:
            return DocumentType.CLINICAL_NOTE
        
        return DocumentType.UNKNOWN
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        sections = {}
        for section_name, pattern in self.section_patterns.items():
            match = pattern.search(text)
            if match:
                sections[section_name] = match.group(1).strip()
        
        if not sections:
            sections['full_text'] = text
        
        return sections
    
    def extract_tables(self, pdf_path: str) -> List[pd.DataFrame]:
        tables = []
        try:
            import camelot
            extracted_tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
            for table in extracted_tables:
                df = table.df
                if not df.empty and df.shape[0] > 1 and df.shape[1] > 1:
                    tables.append(df)
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
            
        return tables
    
    def calculate_quality_score(self, doc: MedicalDocument) -> float:
        score = 0.0
        
        # Entity richness
        entity_density = len(doc.entities) / max(len(doc.content.split()), 1) * 100
        score += min(entity_density * 10, 30)  
        
        # Section completeness
        important_sections = ['diagnosis', 'medications', 'lab_results', 'assessment_plan']
        section_score = sum(10 for section in important_sections if section in doc.sections)
        score += section_score  
        
        # Metadata completeness
        metadata_fields = ['patient_id', 'date', 'provider', 'facility']
        metadata_score = sum(5 for field in metadata_fields if field in doc.metadata)
        score += metadata_score  
        
        # Content length and structure
        if 100 < len(doc.content.split()) < 10000:
            score += 10
        
        # Table presence
        if doc.tables:
            score += 10
        
        return min(score / 100, 1.0)

class LegendaryMedicalIngestionPipeline:
    def __init__(self, 
                 data_dir: str,
                 output_dir: str,
                 batch_size: int = 32,
                 max_workers: int = 8):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        self.parser = MedicalDocumentParser()
        self.processed_checksums = self._load_processed_checksums()
        
        # Stats tracking
        self.stats = {
            'total_processed': 0,
            'duplicates_skipped': 0,
            'errors': 0,
            'total_entities': 0,
            'processing_time': 0
        }
    
    def _load_processed_checksums(self) -> set:
        checksum_file = self.output_dir / 'processed_checksums.json'
        if checksum_file.exists():
            with open(checksum_file, 'r') as f:
                return set(json.load(f))
        return set()
    
    def _save_processed_checksums(self):
        checksum_file = self.output_dir / 'processed_checksums.json'
        with open(checksum_file, 'w') as f:
            json.dump(list(self.processed_checksums), f)
    
    def _calculate_checksum(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
        text = ""
        metadata = {'source_file': str(pdf_path), 'pages': 0}
        
        try:
            pdf_document = fitz.open(pdf_path)
            metadata['pages'] = len(pdf_document)
            
            for page_num, page in enumerate(pdf_document, 1):
                page_text = page.get_text()
                
                # OCR fallback for scanned pages
                if len(page_text.strip()) < 50:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_text = pytesseract.image_to_string(img)
                
                text += f"\n--- Page {page_num} ---\n{page_text}"
            
            pdf_document.close()
            
            # Extract metadata
            doc_info = pdf_document.metadata
            if doc_info:
                metadata.update({
                    'title': doc_info.get('title', ''),
                    'author': doc_info.get('author', ''),
                    'creation_date': str(doc_info.get('creationDate', '')),
                })
            
        except Exception as e:
            logger.error(f"PDF extraction failed for {pdf_path}: {e}")
            raise
        
        return text, metadata
    
    def _redact_phi(self, text: str) -> str:
        # Basic PHI redaction (should use proper de-identification service in production)
        patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]'),  # SSN
            (r'\b\d{10}\b', '[MRN]'),  # Medical Record Number
            (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[NAME]'),  # Names (basic)
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]'),  # Phone numbers
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),  # Email
        ]
        
        redacted = text
        for pattern, replacement in patterns:
            redacted = re.sub(pattern, replacement, redacted)
        
        return redacted
    
    async def process_document(self, file_path: Path) -> Optional[MedicalDocument]:
        try:
            start_time = datetime.now()
            
            # Extract text
            if file_path.suffix.lower() == '.pdf':
                text, metadata = self._extract_text_from_pdf(file_path)
            else:
                text = file_path.read_text(encoding='utf-8', errors='ignore')
                metadata = {'source_file': str(file_path)}
            
            # Check for duplicates
            checksum = self._calculate_checksum(text)
            if checksum in self.processed_checksums:
                self.stats['duplicates_skipped'] += 1
                logger.info(f"Skipping duplicate: {file_path}")
                return None
            
            # Classify document
            doc_type = self.parser.classify_document(text, metadata)
            
            # Extract sections
            sections = self.parser.extract_sections(text)
            
            # Extract entities
            entities = self.parser.ner_extractor.extract_entities(text)
            
            # Extract tables
            tables = []
            if file_path.suffix.lower() == '.pdf':
                tables = self.parser.extract_tables(str(file_path))
            
            # PHI redaction
            redacted_text = self._redact_phi(text)
            
            # Create document
            doc = MedicalDocument(
                doc_id=checksum[:16],
                content=redacted_text,
                doc_type=doc_type,
                metadata=metadata,
                entities=entities,
                sections=sections,
                tables=tables,
                checksum=checksum,
                processed_at=datetime.now(),
                quality_score=0.0,
                phi_redacted=True
            )
            
            # Calculate quality score
            doc.quality_score = self.parser.calculate_quality_score(doc)
            
            # Update stats
            self.stats['total_processed'] += 1
            self.stats['total_entities'] += len(entities)
            self.stats['processing_time'] += (datetime.now() - start_time).total_seconds()
            
            # Mark as processed
            self.processed_checksums.add(checksum)
            
            logger.info(f"Processed: {file_path.name} | Type: {doc_type.value} | Entities: {len(entities)} | Quality: {doc.quality_score:.2f}")
            
            return doc
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            self.stats['errors'] += 1
            return None
    
    async def process_batch(self, file_paths: List[Path]) -> List[MedicalDocument]:
        tasks = [self.process_document(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks)
        return [doc for doc in results if doc is not None]
    
    def save_documents(self, documents: List[MedicalDocument], batch_num: int):
        output_file = self.output_dir / f'medical_documents_batch_{batch_num:04d}.json'
        
        serializable_docs = []
        for doc in documents:
            doc_dict = {
                'doc_id': doc.doc_id,
                'content': doc.content,
                'doc_type': doc.doc_type.value,
                'metadata': doc.metadata,
                'entities': [
                    {
                        'text': e.text,
                        'type': e.type,
                        'confidence': e.confidence,
                        'normalized_value': e.normalized_value,
                        'unit': e.unit,
                        'reference_range': e.reference_range
                    } for e in doc.entities
                ],
                'sections': doc.sections,
                'tables': [table.to_dict() for table in doc.tables],
                'checksum': doc.checksum,
                'processed_at': doc.processed_at.isoformat(),
                'quality_score': doc.quality_score,
                'phi_redacted': doc.phi_redacted
            }
            serializable_docs.append(doc_dict)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_docs, f, indent=2)
        
        logger.info(f"Saved batch {batch_num} with {len(documents)} documents to {output_file}")
    
    async def run(self):
        logger.info(f"Starting legendary medical ingestion pipeline...")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Collect all files
        file_paths = []
        for ext in ['*.pdf', '*.txt', '*.csv', '*.json']:
            file_paths.extend(self.data_dir.glob(ext))
        
        logger.info(f"Found {len(file_paths)} files to process")
        
        # Process in batches
        for i in range(0, len(file_paths), self.batch_size):
            batch = file_paths[i:i+self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1}/{(len(file_paths)-1)//self.batch_size + 1}")
            
            documents = await self.process_batch(batch)
            
            if documents:
                self.save_documents(documents, i//self.batch_size + 1)
        
        # Save checksums
        self._save_processed_checksums()
        
        # Print stats
        logger.info("=" * 50)
        logger.info("INGESTION COMPLETE - STATISTICS:")
        logger.info(f"Total documents processed: {self.stats['total_processed']}")
        logger.info(f"Duplicates skipped: {self.stats['duplicates_skipped']}")
        logger.info(f"Errors encountered: {self.stats['errors']}")
        logger.info(f"Total entities extracted: {self.stats['total_entities']}")
        logger.info(f"Average processing time: {self.stats['processing_time']/max(self.stats['total_processed'], 1):.2f}s")
        logger.info("=" * 50)

if __name__ == "__main__":
    pipeline = LegendaryMedicalIngestionPipeline(
        data_dir="rag_chatbot/data",
        output_dir="rag_chatbot/processed_data",
        batch_size=32,
        max_workers=8
    )
    
    asyncio.run(pipeline.run())