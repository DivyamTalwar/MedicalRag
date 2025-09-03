import os
import re
import json
import logging
from typing import List, Dict, Any, Optional
from llama_index.core.base.llms.types import ChatMessage
from rag_chatbot.app.core.llm import CustomLLM, XMLExtractionSystem
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
            r'\b\d+\.\d+\s*(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL|%)\b',  
            r'\b(?:HbA1c|TSH|HDL|LDL|CBC|ESR|WBC|RBC)\b',  
            r'\b\d+\s*-\s*\d+(?:\.\d+)?\b',  
            r'\b(?:pH|PCO2|PO2|HCO3|TCO2)\b',  
            r'\b(?:Toxoplasma|Rubella|Cytomegalovirus|Herpes)\b',  
        ]

    def _create_template(self):
        return (
            "You are an expert medical assistant. Your task is to rephrase a follow-up question into a standalone, medically precise question based on the provided chat history. "
            "It is **CRITICAL** to preserve all medical terminology, lab values, units, and numerical data exactly as they appear in the original question. Do not alter or omit any details.\n\n"
            "Chat History:\n"
            "{chat_history}\n\n"
            "Follow-up Input: {question}\n\n"
            "Example:\n"
            "Chat History: User: What does a high TSH level of 4.5 mIU/L indicate?\n"
            "Follow-up Input: What about the T4 levels?\n"
            "Standalone Medical Question: Based on a TSH level of 4.5 mIU/L, what do the corresponding T4 levels indicate?\n\n"
            "Your Turn:\n"
            "Standalone Medical Question:"
        )

    def _preserve_medical_context(self, original: str, condensed: str) -> str:
        for pattern in self.critical_medical_patterns:
            original_matches = set(re.findall(pattern, original, re.IGNORECASE))
            condensed_matches = set(re.findall(pattern, condensed, re.IGNORECASE))
            
            missing_terms = original_matches - condensed_matches
            if missing_terms:
                condensed += f" {' '.join(missing_terms)}"
        
        return condensed

    def condense(self, question: str, chat_history: List[ChatMessage]) -> str:
        if not chat_history:
            return question

        try:
            history_str = "\n".join(
                [f"{msg.role.capitalize()}: {msg.content}" for msg in chat_history]
            )
            
            prompt = self._template.format(chat_history=history_str, question=question)
            response = self.llm.invoke(prompt)['content'].strip()
            
            response = self._preserve_medical_context(question, response)
            
            return response
            
        except Exception as e:
            logging.error(f"Query condensation failed: {e}")
            return question

from rag_chatbot.app.models.agent_models import SubQueryGeneration

class SubQueryGenerator:
    def __init__(self, llm: CustomLLM):
        self.llm = llm
        self.query_condenser = QueryCondenser(llm)
        self.xml_extractor = XMLExtractionSystem(llm)

    def _create_xml_subquery_prompt(self, question: str) -> str:
        return f"""
**MEDICAL SUBQUERY DECOMPOSITION TASK**

You are a medical expert specializing in breaking down complex clinical questions into focused, actionable subqueries.

**Primary Question to Analyze:**
"{question}"

**Decomposition Requirements:**
1. **Comprehensive Coverage:** Ensure all aspects of the main question are addressed
2. **Medical Precision:** Preserve exact medical terminology, values, and units
3. **Non-Overlapping:** Each subquery must address a distinct aspect
4. **Clinical Relevance:** Focus on clinically significant elements
5. **Optimal Count:** Generate 2-5 subqueries (quality over quantity)

**Medical Analysis Guidelines:**
- Identify specific medical conditions, symptoms, or findings mentioned
- Preserve laboratory values, measurements, and units exactly
- Consider diagnostic, therapeutic, and prognostic aspects
- Include relevant clinical context and relationships
- Address both direct questions and implied clinical concerns

**Examples of Quality Decomposition:**

**Example 1 - Lab Results Analysis:**
Main Question: "Analyze the patient's arterial blood gas (ABG) results: pH 7.35, PCO2 45 mmHg, PO2 95 mmHg, and HCO3 24 mEq/L. What is the acid-base status, and how does it relate to the patient's respiratory and metabolic condition?"

Quality Subqueries:
1. "What is the patient's acid-base status based on a pH of 7.35?"
2. "How does a PCO2 of 45 mmHg indicate the patient's respiratory status?"
3. "What does an HCO3 of 24 mEq/L reveal about the patient's metabolic condition?"
4. "Is there evidence of compensation in these ABG results?"
5. "What is the patient's oxygenation status given a PO2 of 95 mmHg?"

**Example 2 - Symptom Analysis:**
Main Question: "Patient presents with chest pain, shortness of breath, and elevated troponin levels. What could be the diagnosis?"

Quality Subqueries:
1. "What are the potential causes of chest pain in this clinical context?"
2. "How do shortness of breath symptoms relate to possible cardiac conditions?"
3. "What do elevated troponin levels indicate about cardiac muscle damage?"
4. "What diagnostic workup would be appropriate for this presentation?"

**Your Task:**
Generate your subquery breakdown using the XML format below. Focus on creating the most clinically relevant and comprehensive set of subqueries.
"""

    def _create_subquery_prompt(self, question: str) -> str:
        return f"""
You are an expert medical analyst. Your task is to decompose a complex medical question into 2 to 5 precise, non-overlapping subqueries.

**Main Question:**
"{question}"

**Your Task:**
Generate a JSON object containing a list of 2 to 5 subqueries that break down the main question.

**Strict Rules:**
1.  **JSON Output Only:** The output must be a single, valid JSON object with a "queries" key containing a list of strings.
2.  **No Overlap:** Each subquery must be distinct and cover a unique aspect of the main question.
3.  **Medical Precision:** Use exact medical terminology. Preserve all values, units, and terms from the original question.
4.  **Clarity and Conciseness:** Each subquery should be a clear, direct question.

**Good Example:**
Main Question: "Analyze the patient's arterial blood gas (ABG) results: pH 7.35, PCO2 45 mmHg, PO2 95 mmHg, and HCO3 24 mEq/L. What is the acid-base status, and how does it relate to the patient's respiratory and metabolic condition?"
Response:
{{
    "queries": [
        "What is the patient's acid-base status based on a pH of 7.35?",
        "How does a PCO2 of 45 mmHg indicate the patient's respiratory status?",
        "What does an HCO3 of 24 mEq/L reveal about the patient's metabolic condition?",
        "Is there evidence of compensation in the ABG results?",
        "What is the patient's oxygenation status given a PO2 of 95 mmHg?"
    ]
}}

**Bad Example (Vague and Overlapping):**
Main Question: "What's up with the patient's lab work?"
Response:
{{
    "queries": [
        "What are the lab results?",
        "Tell me about the blood test.",
        "Are the labs normal?"
    ]
}}

**Your JSON Response:**
"""

    def _extract_llm_content(self, response) -> str:
        try:
            if isinstance(response, dict) and 'content' in response:
                return response['content'].strip()
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

    def _parse_json_from_response(self, text: str) -> Optional[Dict[str, Any]]:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                logging.warning("Failed to parse JSON from markdown block.")

        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                logging.warning("Failed to parse raw JSON object from response.")

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logging.warning("Failed to parse the entire response string as JSON.")
            return None

    async def generate(self, question: str, chat_history: List = [], use_xml: bool = True) -> SubQueryGeneration:
        original_question = question

        if chat_history:
            condensed_question = self.query_condenser.condense(question, chat_history)
        else:
            condensed_question = question

        if not condensed_question or not condensed_question.strip():
            final_query = original_question
        else:
            final_query = condensed_question

        # Try XML-based extraction first
        if use_xml:
            try:
                xml_prompt = self._create_xml_subquery_prompt(final_query)
                
                result = await self.xml_extractor.get_structured_xml_response(
                    prompt=xml_prompt,
                    model_class=SubQueryGeneration,
                    context="You are a medical subquery generation specialist. Create precise, non-overlapping medical subqueries that comprehensively address the main question."
                )
                
                # Validate that we have actual queries
                if result.queries and len(result.queries) > 0:
                    validated_queries = [q for q in result.queries if isinstance(q, str) and q.strip()]
                    if validated_queries:
                        result.queries = validated_queries
                        logging.info(f"Successfully generated {len(validated_queries)} subqueries using XML")
                        return result
                
            except Exception as e:
                logging.error(f"XML subquery generation failed: {e}, falling back to JSON")
                use_xml = False

        # Fallback to JSON-based approach
        if not use_xml:
            try:
                prompt = self._create_subquery_prompt(final_query)
                response = self.llm.invoke(prompt)
                content = self._extract_llm_content(response)
                
                if content:
                    json_data = self._parse_json_from_response(content)
                    if json_data and json_data.get("queries"):
                        validated_queries = [q for q in json_data.get("queries", []) if isinstance(q, str) and q.strip()]
                        if validated_queries:
                            json_data["queries"] = validated_queries
                            return SubQueryGeneration.model_validate(json_data)
                    logging.warning("LLM response did not contain valid and non-empty JSON for subqueries.")
                            
            except Exception as e:
                logging.error(f"Subquery generation with LLM failed: {e}")
        
        # Final fallback: return the original query as a single subquery
        logging.warning("All subquery generation methods failed, using original query")
        return SubQueryGeneration(queries=[final_query])

class ContradictionDetector:
    def __init__(self, llm: CustomLLM):
        self.llm = llm
        self.xml_extractor = XMLExtractionSystem(llm)
        self._template = self._create_template()
        self._xml_template = self._create_xml_template()

    def _create_template(self):
        return (
            "You are a clinical data analyst specializing in identifying contradictions in medical information. "
            "Review the numbered text chunks below and determine if any of them contradict each other.\n\n"
            "**Chunks:**\n{chunks}\n\n"
            "Your response must be a JSON object with three keys:\n"
            "1. `contradictory_indices`: A list of the 1-based indices of the chunks that contradict each other. If no contradiction is found, this should be an empty list `[]`.\n"
            "2. `confidence_score`: A float between 0.0 (no confidence) and 1.0 (absolute confidence) indicating your certainty about the contradiction.\n"
            "3. `explanation`: A brief, clear explanation of why the chunks are contradictory.\n\n"
            "**Example 1 (Contradiction):**\n"
            "Chunks:\n"
            "1. Patient's TSH level is 5.2 mIU/L, which is elevated.\n"
            "2. The thyroid function test is within the normal range.\n"
            "JSON Output:\n"
            '{{\n'
            '    "contradictory_indices": [1, 2],\n'
            '    "confidence_score": 0.95,\n'
            '    "explanation": "An elevated TSH level of 5.2 mIU/L is inconsistent with a normal thyroid function test."\n'
            '}}\n\n'
            "**Example 2 (No Contradiction):**\n"
            "Chunks:\n"
            "1. The patient's fasting glucose is 98 mg/dL.\n"
            "2. The HbA1c level is 5.5%.\n"
            "JSON Output:\n"
            '{{\n'
            '    "contradictory_indices": [],\n'
            '    "confidence_score": 0.9,\n'
            '    "explanation": "Both fasting glucose and HbA1c levels are within the normal, non-diabetic range."\n'
            '}}\n\n'
            "**Your Analysis:**"
        )

    def _create_xml_template(self):
        return """
**MEDICAL CONTRADICTION ANALYSIS TASK**

You are a clinical data analyst specializing in identifying contradictions and inconsistencies in medical information.

**Medical Information Chunks to Analyze:**
{chunks}

**Analysis Instructions:**
1. **Systematic Review:** Examine each chunk for medical facts, values, and statements
2. **Contradiction Identification:** Look for conflicting information between chunks
3. **Clinical Context:** Consider medical knowledge and normal/abnormal ranges
4. **Confidence Assessment:** Evaluate certainty level of detected contradictions

**Types of Medical Contradictions to Detect:**
- Conflicting laboratory values or reference ranges
- Inconsistent diagnostic conclusions
- Contradictory treatment recommendations
- Opposing clinical assessments
- Incompatible timeline information
- Conflicting patient status descriptions

**Examples of Medical Contradictions:**

**Example 1 - Clear Contradiction:**
Chunk 1: "Patient's TSH level is 5.2 mIU/L, which is elevated indicating hypothyroidism."
Chunk 2: "Thyroid function tests are completely normal with no abnormalities detected."
→ **Contradiction:** Elevated TSH contradicts normal thyroid function assessment

**Example 2 - No Contradiction:**
Chunk 1: "Patient's fasting glucose is 98 mg/dL, within normal range."
Chunk 2: "HbA1c level is 5.5%, indicating good long-term glucose control."
→ **No Contradiction:** Both values support normal glucose metabolism

**Your Analysis Requirements:**
- Identify specific contradictory information with medical reasoning
- Provide confidence level based on clinical significance
- List the chunk indices that contain contradictory information
- Explain the medical basis for contradiction detection
"""

    def _regex_confidence_extraction(self, response: str) -> float:
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
                    if score > 1.0:
                        score = score / 100.0
                    return max(0.0, min(1.0, score))
                except ValueError:
                    continue
        
        if any(word in response.lower() for word in ['contradict', 'conflict', 'inconsistent']):
            return 0.7
        elif any(word in response.lower() for word in ['no contradiction', 'consistent', 'agree']):
            return 0.1
        
        return 0.5 

    def _regex_indices_extraction(self, response: str, num_chunks: int) -> List[int]:
        indices = []
        
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
                
                numbers = re.findall(r'\d+', str(match))
                for num in numbers:
                    try:
                        idx = int(num) - 1  
                        if 0 <= idx < num_chunks:
                            indices.append(idx)
                    except ValueError:
                        continue
        
        return list(set(indices)) 

    async def detect(self, chunks: List[str], use_xml: bool = True) -> Dict[str, Any]:
        if len(chunks) < 2:
            return {"contradictory_indices": [], "confidence_score": 0.0, "explanation": "Insufficient chunks for contradiction analysis"}
        
        numbered_chunks = "\n".join([f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)])
        
        if use_xml:
            try:
                # Define a simple Pydantic model for contradiction results
                from pydantic import BaseModel
                from typing import List as TypingList
                
                class ContradictionResult(BaseModel):
                    contradictory_indices: TypingList[int]
                    confidence_score: float
                    explanation: str
                
                xml_prompt = self._xml_template.format(chunks=numbered_chunks)
                
                result = await self.xml_extractor.get_structured_xml_response(
                    prompt=xml_prompt,
                    model_class=ContradictionResult,
                    context="You are a medical contradiction detection specialist. Analyze medical information for conflicting statements or values."
                )
                
                # Validate indices are within range and adjust for 0-based indexing
                valid_indices = []
                for idx in result.contradictory_indices:
                    # Convert from 1-based to 0-based indexing
                    zero_based_idx = idx - 1 if idx > 0 else idx
                    if 0 <= zero_based_idx < len(chunks):
                        valid_indices.append(zero_based_idx)
                
                return {
                    "contradictory_indices": valid_indices,
                    "confidence_score": max(0.0, min(1.0, result.confidence_score)),
                    "explanation": result.explanation
                }
                
            except Exception as e:
                logging.error(f"XML contradiction detection failed: {e}, falling back to JSON")
                use_xml = False
        
        # Fallback to JSON approach
        if not use_xml:
            prompt = self._template.format(chunks=numbered_chunks)
            
            try:
                response = self.llm.invoke(prompt)
                response_content = response.content if hasattr(response, 'content') else str(response)
                
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = json.loads(response_content)
                
                # Convert 1-based indices to 0-based and validate
                contradictory_indices = result.get("contradictory_indices", [])
                valid_indices = []
                for idx in contradictory_indices:
                    zero_based_idx = idx - 1 if idx > 0 else idx
                    if 0 <= zero_based_idx < len(chunks):
                        valid_indices.append(zero_based_idx)
                
                return {
                    "contradictory_indices": valid_indices,
                    "confidence_score": max(0.0, min(1.0, result.get("confidence_score", 0.0))),
                    "explanation": result.get("explanation", "")
                }
                
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                logging.warning(f"JSON parsing failed, using regex extraction: {e}")
                
                confidence_score = self._regex_confidence_extraction(response_content)
                contradictory_indices = self._regex_indices_extraction(response_content, len(chunks))
                
                return {
                    "contradictory_indices": contradictory_indices,
                    "confidence_score": confidence_score,
                    "explanation": "Extracted using regex fallback"
                }
    
    def detect_sync(self, chunks: List[str]) -> Dict[str, Any]:
        """Synchronous wrapper for backward compatibility."""
        import asyncio
        return asyncio.run(self.detect(chunks, use_xml=True))

class MedicalQueryExpander:
    def __init__(self):
        self.entity_extractor = MedicalEntityExtractor()
        self.synonym_map = {
            "ct": ["computed tomography", "cat scan", "ct scan"],
            "mri": ["magnetic resonance imaging", "nmr", "mri scan"],
            "us": ["ultrasound", "sonography", "echo", "ultrasonography"],
            "xray": ["x-ray", "radiograph", "roentgen", "radiography"],
            
            "dx": ["diagnosis", "diagnostic", "diagnose"],
            "fx": ["fracture", "break", "broken bone", "bone fracture"],
            "hx": ["history", "medical history", "patient history"],
            "tx": ["treatment", "therapy", "therapeutic"],
            
            "abd": ["abdomen", "abdominal", "belly"],
            "chest": ["thorax", "thoracic", "lung", "pulmonary"],
            
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
            
            "torch": ["toxoplasma rubella cytomegalovirus herpes", "torch panel", "torch screen"],
            "toxoplasma": ["toxoplasmosis", "toxoplasma gondii"],
            "rubella": ["german measles", "rubella virus"],
            "cmv": ["cytomegalovirus", "human cytomegalovirus"],
            "hsv": ["herpes simplex virus", "herpes simplex"],
            
            "abg": ["arterial blood gas", "blood gas analysis", "arterial blood gases"],
            "pco2": ["partial pressure carbon dioxide", "carbon dioxide pressure"],
            "po2": ["partial pressure oxygen", "oxygen pressure"],
            "hco3": ["bicarbonate", "hydrogen carbonate"],
            "tco2": ["total carbon dioxide", "total co2"],
            
            "mg/dl": ["milligrams per deciliter", "mg per dl"],
            "mmhg": ["millimeters mercury", "mm hg", "torr"],
            "meq/l": ["milliequivalents per liter", "meq per liter"],
            "iu/ml": ["international units per milliliter"],
            "au/ml": ["arbitrary units per milliliter"],
            "u/ml": ["units per milliliter"],
            
            "hplc": ["high performance liquid chromatography", "liquid chromatography"],
            "p2": ["p2 peak", "p2 window", "p2 hemoglobin"],
            "p3": ["p3 peak", "p3 window", "p3 hemoglobin"],
            "hbf": ["fetal hemoglobin", "hemoglobin f"],
            "hba2": ["hemoglobin a2", "hb a2"],
            
            "pacs": ["picture archiving communication system", "medical imaging system"],
            "ris": ["radiology information system", "radiology system"],
            "dicom": ["digital imaging communications medicine", "medical imaging standard"],
            "worklist": ["work list", "task list", "patient list"],
            "tat": ["turnaround time", "turn around time"],
            "sla": ["service level agreement", "service agreement"],
            
            "radflow": ["radiology workflow", "rad flow", "radiology flow"],
            "qr": ["quality review", "peer review"],
            "stat": ["urgent", "emergency", "immediate", "priority"],
            
            "ref": ["reference", "normal range", "reference range"],
            "nl": ["normal", "within normal limits", "wnl"],
            "abn": ["abnormal", "outside normal", "irregular"],
        }
    
    def _extract_lab_values_and_units(self, query: str) -> List[str]:
        value_patterns = [
            r'\d+\.\d+\s*(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL|%)',
            r'\d+\s*-\s*\d+(?:\.\d+)?', 
            r'(?:mg/dL|mmHg|mEq/L|IU/mL|AU/mL|U/mL)', 
        ]
        
        extracted = []
        for pattern in value_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            extracted.extend(matches)
        
        return extracted
    
    def _extract_procedure_names(self, query: str) -> List[str]:
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
            
            lab_values = self._extract_lab_values_and_units(query)
            expanded_terms.update(lab_values)
            
            procedures = self._extract_procedure_names(query)
            expanded_terms.update(procedures)
            
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
