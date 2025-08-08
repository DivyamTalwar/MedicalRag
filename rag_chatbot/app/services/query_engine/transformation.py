import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from rag_chatbot.app.core.llm import CustomLLM
from rag_chatbot.app.core.extractor import MedicalEntityExtractor
from dataclasses import dataclass
from difflib import SequenceMatcher

@dataclass
class QueryProcessingConfig:
    max_sub_queries: int = 5
    min_sub_queries: int = 2
    default_weight: float = 1.0
    max_contradiction_confidence: float = 1.0
    enable_caching: bool = True
    cache_size: int = 100

class QueryCondenser:
    def __init__(self, llm: CustomLLM):
        self.llm = llm
        self._template = self._create_template()
        self.critical_medical_patterns = [
            r'\b\d+\.\d+\s*(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL|%)\b',  # Lab values with units
            r'\b(?:HbA1c|TSH|HDL|LDL|CBC|ESR|WBC|RBC)\b',  # Critical lab markers
            r'\b\d+\s*-\s*\d+(?:\.\d+)?\b',  # Reference ranges
            r'\b(?:pH|PCO2|PO2|HCO3|TCO2)\b',  # Blood gas parameters
            r'\b(?:Toxoplasma|Rubella|Cytomegalovirus|Herpes)\b',  # TORCH infections
        ]

    def _create_template(self):
        return (
            "Given the following conversation and a follow-up question, rephrase the "
            "follow-up question to be a standalone question that is precise and "
            "medically relevant. **CRITICAL**: Preserve ALL medical terminology, lab values, "
            "units, reference ranges, and numerical data exactly as written.\n\n"
            "Chat History:\n"
            "{chat_history}\n\n"
            "Follow-up Input: {question}\n"
            "Standalone Medical Question:"
        )

    def _preserve_medical_context(self, original: str, condensed: str) -> str:
        """Ensure critical medical terms are preserved in condensed query"""
        for pattern in self.critical_medical_patterns:
            original_matches = set(re.findall(pattern, original, re.IGNORECASE))
            condensed_matches = set(re.findall(pattern, condensed, re.IGNORECASE))
            
            missing_terms = original_matches - condensed_matches
            if missing_terms:
                condensed += f" {' '.join(missing_terms)}"
        
        return condensed

    def condense(self, question: str, chat_history: List[BaseMessage]) -> str:
        if not chat_history:
            return question

        try:
            history_str = "\n".join(
                [f"{msg.role.capitalize()}: {msg.content}" for msg in chat_history]
            )
            
            prompt = self._template.format(chat_history=history_str, question=question)
            response = self.llm.invoke(prompt).content.strip()
            
            # Preserve critical medical context
            response = self._preserve_medical_context(question, response)
            
            return response
            
        except Exception as e:
            logging.error(f"Query condensation failed: {e}")
            return question

from rag_chatbot.app.models.agent_models import SubQueryGeneration

class SubQueryGenerator:
    def __init__(self, llm: CustomLLM):
        self.llm = llm

    def _create_subquery_prompt(self, question: str) -> str:
        return f"""
You are a medical information assistant. Generate exactly 2 subqueries from the main question.

Main Question: {question}

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
    "query1": "First subquery here",
    "query2": "Second subquery here"
}}

Example Response:
{{
    "query1": "What are the key features of the 5-step order process?",
    "query2": "How does the order journey work in CIVIE system?"
}}

Your JSON Response:
"""

    def _extract_llm_content(self, response) -> str:
        try:
            if hasattr(response, 'choices'):
                content = response.choices[0].message.content
            elif hasattr(response, 'content'):
                content = response.content
            elif isinstance(response, str):
                content = response
            else:
                content = str(response)
            return content.strip()
        except Exception as e:
            logging.error(f"Content extraction error: {e}")
            return ""

    def _extract_queries_with_regex(self, text: str) -> dict:
        patterns = [
            r'"query1":\s*"([^"]+)".*?"query2":\s*"([^"]+)"',
            r'query1["\s:]*([^,\n}]+).*?query2["\s:]*([^,\n}]+)',
            r'1[.)]\s*([^\n2]+).*?2[.)]\s*([^\n]+)',
            r'First.*?:\s*([^\n]+).*?Second.*?:\s*([^\n]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                query1 = match.group(1).strip().strip('"').strip("'")
                query2 = match.group(2).strip().strip('"').strip("'")
                
                if len(query1) >= 5 and len(query2) >= 5:
                    return {"query1": query1, "query2": query2}
        
        return {
            "query1": f"What are the main components of {text[:50]}?",
            "query2": f"How does the process work for {text[:50]}?"
        }

    async def generate(self, question: str) -> SubQueryGeneration:
        """Absolutely bulletproof subquery generation"""
        content = ""
        # Attempt 1: Structured LLM call
        try:
            prompt = self._create_subquery_prompt(question)
            response = self.llm.invoke(prompt)
            content = self._extract_llm_content(response)
            
            if content:
                try:
                    json_data = json.loads(content)
                    return SubQueryGeneration.model_validate(json_data)
                except json.JSONDecodeError:
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        json_data = json.loads(json_match.group())
                        return SubQueryGeneration.model_validate(json_data)
                        
        except Exception as e:
            logging.error(f"Attempt 1 failed: {e}")
        
        # Attempt 2: Regex extraction
        try:
            queries_dict = self._extract_queries_with_regex(content or question)
            return SubQueryGeneration.model_validate(queries_dict)
        except Exception as e:
            logging.error(f"Attempt 2 failed: {e}")
        
        # Attempt 3: LLM repair
        try:
            repair_prompt = f"""
            Fix this broken JSON to match the required format:
            Broken: {content}
            
            Required format:
            {{"query1": "...", "query2": "..."}}
            
            Fixed JSON:
            """
            repaired = self.llm.invoke(repair_prompt)
            repaired_content = self._extract_llm_content(repaired)
            json_data = json.loads(repaired_content)
            return SubQueryGeneration.model_validate(json_data)
        except Exception as e:
            logging.error(f"Attempt 3 failed: {e}")
        
        # Final fallback - NEVER fails
        return SubQueryGeneration(
            query1=f"What are the main aspects of: {question}",
            query2=f"How does the following work: {question}"
        )

class ContradictionDetector:
    def __init__(self, llm: CustomLLM):
        self.llm = llm
        self._template = self._create_template()

    def _create_template(self):
        return (
            "You are an expert at detecting contradictions in text. "
            "Given a list of text chunks, identify any chunks that contradict each other. "
            "Return a JSON object with two keys: 'contradictory_indices' (a list of the indices of the contradictory chunks) "
            "and 'confidence_score' (a float from 0.0 to 1.0 indicating your confidence in the detection). "
            "If there are no contradictions, return an empty list for 'contradictory_indices'.\n\n"
            "Chunks:\n{chunks}\n\n"
            "JSON Output:"
        )

    def _regex_confidence_extraction(self, response: str) -> float:
        """Extract confidence score using regex when JSON parsing fails"""
        confidence_patterns = [
            r'confidence[_\s]*score[\'\":\s]*(\d*\.?\d+)',
            r'confidence[\'\":\s]*(\d*\.?\d+)',
            r'score[\'\":\s]*(\d*\.?\d+)',
            r'(\d+\.?\d*)(?:\s*%)?(?:\s*confidence)',
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    # Convert percentage to decimal if needed
                    if score > 1.0:
                        score = score / 100.0
                    return max(0.0, min(1.0, score))
                except ValueError:
                    continue
        
        # Default confidence based on response content
        if any(word in response.lower() for word in ['contradict', 'conflict', 'inconsistent']):
            return 0.7
        elif any(word in response.lower() for word in ['no contradiction', 'consistent', 'agree']):
            return 0.1
        
        return 0.5  # Default fallback

    def _regex_indices_extraction(self, response: str, num_chunks: int) -> List[int]:
        """Extract contradictory indices using regex when JSON parsing fails"""
        indices = []
        
        # Look for numbered indices
        index_patterns = [
            r'indices?[\'\":\s]*\[([^\]]+)\]',
            r'chunks?[\'\":\s]*\[([^\]]+)\]',
            r'contradictory[\'\":\s]*\[([^\]]+)\]',
            r'(\d+)(?:\s*,\s*(\d+))*',
        ]
        
        for pattern in index_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0] if match[0] else match[1] if len(match) > 1 else ""
                
                # Extract numbers from the match
                numbers = re.findall(r'\d+', str(match))
                for num in numbers:
                    try:
                        idx = int(num) - 1  # Convert to 0-based indexing
                        if 0 <= idx < num_chunks:
                            indices.append(idx)
                    except ValueError:
                        continue
        
        return list(set(indices))  # Remove duplicates

    def detect(self, chunks: List[str]) -> Dict[str, Any]:
        if len(chunks) < 2:
            return {"contradictory_indices": [], "confidence_score": 0.0}
        
        prompt = self._template.format(
            chunks="\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)])
        )
        
        try:
            response = self.llm.invoke(prompt).content
            
            # Try JSON parsing first
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            contradictory_indices = result.get("contradictory_indices", [])
            valid_indices = [i for i in contradictory_indices if 0 <= i < len(chunks)]
            
            return {
                "contradictory_indices": valid_indices,
                "confidence_score": max(0.0, min(1.0, result.get("confidence_score", 0.0))),
                "explanation": result.get("explanation", "")
            }
            
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            logging.warning(f"JSON parsing failed, using regex extraction: {e}")
            
            # Regex-based fallback extraction
            confidence_score = self._regex_confidence_extraction(response)
            contradictory_indices = self._regex_indices_extraction(response, len(chunks))
            
            return {
                "contradictory_indices": contradictory_indices,
                "confidence_score": confidence_score,
                "explanation": "Extracted using regex fallback"
            }

class MedicalQueryExpander:
    def __init__(self):
        self.entity_extractor = MedicalEntityExtractor()
        # Comprehensive synonym map based on your sample PDFs
        self.synonym_map = {
            # Imaging modalities
            "ct": ["computed tomography", "cat scan", "ct scan"],
            "mri": ["magnetic resonance imaging", "nmr", "mri scan"],
            "us": ["ultrasound", "sonography", "echo", "ultrasonography"],
            "xray": ["x-ray", "radiograph", "roentgen", "radiography"],
            
            # Medical abbreviations
            "dx": ["diagnosis", "diagnostic", "diagnose"],
            "fx": ["fracture", "break", "broken bone", "bone fracture"],
            "hx": ["history", "medical history", "patient history"],
            "tx": ["treatment", "therapy", "therapeutic"],
            
            # Body parts and systems
            "abd": ["abdomen", "abdominal", "belly"],
            "chest": ["thorax", "thoracic", "lung", "pulmonary"],
            
            # Lab tests and markers (from your PDFs)
            "cbc": ["complete blood count", "full blood count", "blood count"],
            "hba1c": ["glycosylated hemoglobin", "glycated hemoglobin", "hemoglobin a1c"],
            "tsh": ["thyroid stimulating hormone", "thyrotropin"],
            "t3": ["triiodothyronine", "tri-iodothyronine"],
            "t4": ["thyroxine", "tetraiodothyronine"],
            "hdl": ["high density lipoprotein", "good cholesterol"],
            "ldl": ["low density lipoprotein", "bad cholesterol"],
            "esr": ["erythrocyte sedimentation rate", "sed rate"],
            "wbc": ["white blood cell", "white blood count", "leukocyte"],
            "rbc": ["red blood cell", "red blood count", "erythrocyte"],
            
            # TORCH profile (from your PDFs)
            "torch": ["toxoplasma rubella cytomegalovirus herpes", "torch panel", "torch screen"],
            "toxoplasma": ["toxoplasmosis", "toxoplasma gondii"],
            "rubella": ["german measles", "rubella virus"],
            "cmv": ["cytomegalovirus", "human cytomegalovirus"],
            "hsv": ["herpes simplex virus", "herpes simplex"],
            
            # Blood gas parameters (from your PDFs)
            "abg": ["arterial blood gas", "blood gas analysis", "arterial blood gases"],
            "pco2": ["partial pressure carbon dioxide", "carbon dioxide pressure"],
            "po2": ["partial pressure oxygen", "oxygen pressure"],
            "hco3": ["bicarbonate", "hydrogen carbonate"],
            "tco2": ["total carbon dioxide", "total co2"],
            
            # Medical units and values
            "mg/dl": ["milligrams per deciliter", "mg per dl"],
            "mmhg": ["millimeters mercury", "mm hg", "torr"],
            "meq/l": ["milliequivalents per liter", "meq per liter"],
            "iu/ml": ["international units per milliliter"],
            "au/ml": ["arbitrary units per milliliter"],
            "u/ml": ["units per milliliter"],
            
            # Hemoglobin electrophoresis (from your PDFs)
            "hplc": ["high performance liquid chromatography", "liquid chromatography"],
            "p2": ["p2 peak", "p2 window", "p2 hemoglobin"],
            "p3": ["p3 peak", "p3 window", "p3 hemoglobin"],
            "hbf": ["fetal hemoglobin", "hemoglobin f"],
            "hba2": ["hemoglobin a2", "hb a2"],
            
            # Radiology workflow terms (from CIVIE PDF)
            "pacs": ["picture archiving communication system", "medical imaging system"],
            "ris": ["radiology information system", "radiology system"],
            "dicom": ["digital imaging communications medicine", "medical imaging standard"],
            "worklist": ["work list", "task list", "patient list"],
            "tat": ["turnaround time", "turn around time"],
            "sla": ["service level agreement", "service agreement"],
            
            # Procedure and workflow terms
            "radflow": ["radiology workflow", "rad flow", "radiology flow"],
            "qr": ["quality review", "peer review"],
            "stat": ["urgent", "emergency", "immediate", "priority"],
            
            # General medical terms
            "ref": ["reference", "normal range", "reference range"],
            "nl": ["normal", "within normal limits", "wnl"],
            "abn": ["abnormal", "outside normal", "irregular"],
        }
    
    def _extract_lab_values_and_units(self, query: str) -> List[str]:
        """Extract lab values and units from query"""
        value_patterns = [
            r'\d+\.\d+\s*(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL|%)',
            r'\d+\s*-\s*\d+(?:\.\d+)?',  # Reference ranges
            r'(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL)',  # Units
        ]
        
        extracted = []
        for pattern in value_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            extracted.extend(matches)
        
        return extracted
    
    def _extract_procedure_names(self, query: str) -> List[str]:
        """Extract procedure names from query"""
        procedure_patterns = [
            r'(?:blood gas|torch|hemoglobin electrophoresis|lipid profile)',
            r'(?:complete blood count|thyroid function|iron studies)',
            r'(?:liver function|kidney function|cardiac markers)',
        ]
        
        extracted = []
        for pattern in procedure_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            extracted.extend(matches)
        
        return extracted

    def expand(self, query: str) -> List[str]:
        try:
            entities = self.entity_extractor.extract_entities(query)
            flat_entities = self.entity_extractor.flatten_entities(entities)
            
            expanded_terms = set(flat_entities)
            expanded_terms.update(query.lower().split())
            
            # Add lab values and units
            lab_values = self._extract_lab_values_and_units(query)
            expanded_terms.update(lab_values)
            
            # Add procedure names
            procedures = self._extract_procedure_names(query)
            expanded_terms.update(procedures)
            
            # Enhanced synonym expansion
            for term in query.lower().split():
                clean_term = term.strip(".,?!()")
                if clean_term in self.synonym_map:
                    if isinstance(self.synonym_map[clean_term], list):
                        expanded_terms.update(self.synonym_map[clean_term])
                    else:
                        expanded_terms.add(self.synonym_map[clean_term].lower())
            
            return list(expanded_terms)
            
        except Exception as e:
            logging.error(f"Query expansion failed: {e}")
            return query.lower().split()
