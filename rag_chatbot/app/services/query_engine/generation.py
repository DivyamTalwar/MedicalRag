import re
import time
import logging
import asyncio
from pydantic import TypeAdapter
from typing import List, Any, Optional
from rag_chatbot.app.core.llm import CustomLLM, JSONExtractionSystem, XMLExtractionSystem
from rag_chatbot.app.models.data_models import Document
from rag_chatbot.app.models.agent_models import (
    RewriteOutput,
    GateInsufficient,
    SubQueryItem,
    SubQueryAnswer,
    FinalAnswer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self, llm: CustomLLM):
        self.llm = llm
        self.json_extractor = JSONExtractionSystem(llm)
        self.xml_extractor = XMLExtractionSystem(llm)
        self._prompt_template = self._create_prompt_template()
        self._xml_prompt_template = self._create_xml_prompt_template()

    def _create_prompt_template(self) -> str:
        return """You are an expert medical AI assistant. Your sole purpose is to answer questions based *strictly* on the provided context and chat history. Do not use any external knowledge.

**Rules:**
1.  **Context is King:** Your answer must be derived exclusively from the `CONTEXT` section. Do not invent or infer information.
2.  **Synthesize and Detail:** Provide a comprehensive, detailed synthesis of all relevant information. Do not simply summarize.
3.  **Acknowledge Limitations:** If the context does not contain the answer, state that clearly. For example: 'The provided context does not contain information about [topic].'
4.  **Mandatory Disclaimer:** Always conclude your entire response with the following disclaimer on a new line: `This information is for informational purposes only and does not constitute medical advice.`

---

**CHAT HISTORY:**
{chat_history}

**CONTEXT:**
{assembled_context}

**QUESTION:**
{question}

---

**Example Response:**
Based on the provided context, the patient's Arterial Blood Gas (ABG) analysis shows a pH of 7.35, which is at the lower end of the normal range (7.35-7.45). The PCO2 is 45 mmHg, and the HCO3 is 24 mEq/L, both of which are within their respective normal ranges.

This information is for informational purposes only and does not constitute medical advice.

**YOUR ANSWER:**
"""
    
    def _create_xml_prompt_template(self) -> str:
        return """You are an expert medical AI assistant. Generate comprehensive, evidence-based answers using XML-structured responses.

**CORE PRINCIPLES:**
1. **Context-Only Responses:** Base your answer EXCLUSIVELY on the provided CONTEXT section
2. **Medical Precision:** Preserve all medical terminology, lab values, units, and numerical data exactly
3. **Comprehensive Analysis:** Provide detailed synthesis, not just summaries
4. **Limitation Acknowledgment:** Clearly state when context lacks sufficient information
5. **Professional Standards:** Maintain clinical accuracy and appropriate medical language

---

**CHAT HISTORY:**
{chat_history}

**MEDICAL CONTEXT:**
{assembled_context}

**PATIENT QUESTION:**
{question}

---

**RESPONSE STRUCTURE REQUIREMENTS:**
Generate your response using the following XML structure for better information extraction:

<medical_response>
    <primary_answer>
        [Your main, comprehensive answer to the question based on the provided context]
    </primary_answer>
    
    <key_findings>
        <finding>
            <category>Lab Values/Symptoms/Diagnosis/etc.</category>
            <details>Specific details from context</details>
        </finding>
        <!-- Repeat for multiple findings -->
    </key_findings>
    
    <clinical_significance>
        [Clinical interpretation and significance of the findings]
    </clinical_significance>
    
    <limitations>
        [What information is missing or unclear from the provided context]
    </limitations>
    
    <medical_disclaimer>
        This information is for informational purposes only and does not constitute medical advice.
    </medical_disclaimer>
</medical_response>

**XML FORMATTING RULES:**
- Use proper XML tags as shown above
- Include all sections, even if some are brief
- Preserve exact medical values and terminology within XML content
- No markdown formatting inside XML tags
- Ensure XML is well-formed and complete

**EXAMPLE RESPONSE FORMAT:**
<medical_response>
    <primary_answer>Based on the provided context, the patient's Arterial Blood Gas (ABG) analysis shows a pH of 7.35, which is at the lower end of the normal range (7.35-7.45). The PCO2 is 45 mmHg and HCO3 is 24 mEq/L, both within normal ranges, suggesting compensated metabolic acidosis.</primary_answer>
    
    <key_findings>
        <finding>
            <category>ABG Results</category>
            <details>pH: 7.35, PCO2: 45 mmHg, HCO3: 24 mEq/L</details>
        </finding>
    </key_findings>
    
    <clinical_significance>The borderline pH with normal PCO2 and HCO3 suggests the patient's acid-base balance is at the lower limit of normal, requiring clinical correlation with symptoms and other laboratory findings.</clinical_significance>
    
    <limitations>Additional clinical history, symptoms, and other laboratory values would be helpful for complete assessment.</limitations>
    
    <medical_disclaimer>This information is for informational purposes only and does not constitute medical advice.</medical_disclaimer>
</medical_response>"""

    def _extract_json_from_text(self, text: str) -> Optional[str]:
        # Find the start and end of the JSON block
        start_index = text.find('{')
        end_index = text.rfind('}')
        
        if start_index != -1 and end_index != -1:
            return text[start_index:end_index+1]
        return None
    
    def _extract_and_construct_from_xml(self, xml_response: str) -> str:
        """Extract structured information from XML response and construct clean output."""
        try:
            import xml.etree.ElementTree as ET
            
            # Clean up the XML response
            xml_response = re.sub(r'```xml\s*|\s*```', '', xml_response)
            xml_response = re.sub(r'```\s*|\s*```', '', xml_response)
            
            # Parse XML
            root = ET.fromstring(xml_response)
            
            # Extract components
            components = {}
            for child in root:
                if child.tag == 'key_findings':
                    findings = []
                    for finding in child.findall('finding'):
                        category = finding.find('category')
                        details = finding.find('details')
                        if category is not None and details is not None:
                            findings.append(f"**{category.text}:** {details.text}")
                    components['findings'] = findings
                elif child.text and child.text.strip():
                    components[child.tag] = child.text.strip()
            
            # Construct final response
            response_parts = []
            
            # Primary answer
            if 'primary_answer' in components:
                response_parts.append(components['primary_answer'])
            
            # Key findings
            if 'findings' in components and components['findings']:
                response_parts.append("\n**Key Findings:**")
                for finding in components['findings']:
                    response_parts.append(f"• {finding}")
            
            # Clinical significance
            if 'clinical_significance' in components:
                response_parts.append(f"\n**Clinical Significance:**\n{components['clinical_significance']}")
            
            # Limitations
            if 'limitations' in components:
                response_parts.append(f"\n**Limitations:**\n{components['limitations']}")
            
            # Medical disclaimer
            if 'medical_disclaimer' in components:
                response_parts.append(f"\n\n{components['medical_disclaimer']}")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"XML parsing error: {e}")
            # Fallback: extract text between XML tags using regex
            return self._extract_text_from_xml_fallback(xml_response)
    
    def _extract_text_from_xml_fallback(self, xml_response: str) -> str:
        """Fallback method to extract text from XML using regex."""
        try:
            # Extract primary answer
            primary_match = re.search(r'<primary_answer>(.*?)</primary_answer>', xml_response, re.DOTALL)
            primary_text = primary_match.group(1).strip() if primary_match else ""
            
            # Extract clinical significance
            significance_match = re.search(r'<clinical_significance>(.*?)</clinical_significance>', xml_response, re.DOTALL)
            significance_text = significance_match.group(1).strip() if significance_match else ""
            
            # Extract disclaimer
            disclaimer_match = re.search(r'<medical_disclaimer>(.*?)</medical_disclaimer>', xml_response, re.DOTALL)
            disclaimer_text = disclaimer_match.group(1).strip() if disclaimer_match else "This information is for informational purposes only and does not constitute medical advice."
            
            # Combine parts
            response_parts = [primary_text]
            
            if significance_text:
                response_parts.append(f"\n**Clinical Context:**\n{significance_text}")
            
            response_parts.append(f"\n\n{disclaimer_text}")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Fallback XML parsing failed: {e}")
            # Return cleaned text as last resort
            clean_text = re.sub(r'<[^>]+>', '', xml_response).strip()
            return clean_text if clean_text else "Unable to process the medical information properly."

    async def generate(self, prompt: str, assembled_context: str, chat_history: List = [], model=None, use_xml=True, max_retries=2, timeout=30) -> Any:
        if model is None:
            # Handle non-JSON generation with XML structured responses
            history_str = "\n".join(
                [f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history]
            )
            
            if use_xml:
                # Use XML-structured prompt for better response quality
                formatted_prompt = self._xml_prompt_template.format(
                    assembled_context=assembled_context,
                    question=prompt, # in this case, prompt is the query
                    chat_history=history_str
                )
                
                try:
                    response = await self.llm.ainvoke(formatted_prompt)
                    xml_content = response.content
                    
                    # Extract and construct structured response from XML
                    structured_response = self._extract_and_construct_from_xml(xml_content)
                    return structured_response
                    
                except Exception as e:
                    logger.error(f"XML generation failed: {e}, falling back to standard prompt")
                    # Fallback to standard prompt
                    use_xml = False
            
            if not use_xml:
                # Fallback to original prompt template
                formatted_prompt = self._prompt_template.format(
                    assembled_context=assembled_context,
                    question=prompt, # in this case, prompt is the query
                    chat_history=history_str
                )
                response = await self.llm.ainvoke(formatted_prompt)
                return response.content

        try:
            # Handle JSON generation with validation and repair
            for i in range(max_retries):
                try:
                    # Add a timeout to the LLM call
                    raw_response = await asyncio.wait_for(self.llm.ainvoke(prompt), timeout=timeout)
                    raw_text = raw_response.content

                    # 1. Validate with Pydantic
                    try:
                        adapter = TypeAdapter(model)
                        validated_response = adapter.validate_json(raw_text)
                        return validated_response
                    except Exception as pydantic_error:
                        logger.warning(f"Pydantic validation failed on attempt {i+1}: {pydantic_error}")

                        # 2. Attempt to repair with regex
                        json_str = self._extract_json_from_text(raw_text)
                        if json_str:
                            try:
                                adapter = TypeAdapter(model)
                                validated_response = adapter.validate_json(json_str)
                                logger.info("Successfully repaired JSON with regex extraction.")
                                return validated_response
                            except Exception as regex_repair_error:
                                logger.warning(f"Regex repair failed: {regex_repair_error}")
                        
                        # 3. Ask LLM to repair
                        if i < max_retries - 1:
                            logger.info("Attempting to repair JSON with LLM.")
                            repair_prompt = f"""The following JSON is broken. Please fix it to conform to the {model.__name__} schema.
---
BROKEN JSON:
{raw_text}
---
FIXED JSON:"""
                            prompt = repair_prompt
                        else:
                            raise ValueError("Failed to get valid JSON after multiple retries.")

                except asyncio.TimeoutError:
                    logger.error(f"LLM invocation timed out after {timeout} seconds.")
                    if i == max_retries - 1:
                        raise
                except Exception as e:
                    logger.error(f"LLM invocation failed on attempt {i+1}: {e}")
                    if i == max_retries - 1:
                        raise e
            
            raise ValueError("Loop finished without returning, this should not be reached.")

        except Exception as e:
            logger.error(f"Generation failed, using model defaults: {e}")
            
            if model.__name__ == "RewriteOutput":
                original_query = "unknown"
                match = re.search(r"Original query: (.*)", prompt)
                if match:
                    original_query = match.group(1).strip()

                return RewriteOutput(
                    original_query=original_query,
                    dependent_on_history=False,
                    final_query=original_query,
                    reasoning="Fallback due to generation error"
                )
            
            if model.__name__ == "GateOutput":
                normalized_query = "unknown"
                match = re.search(r"Question: (.*)", prompt)
                if match:
                    normalized_query = match.group(1).strip()
                return GateInsufficient(
                    response_type="insufficient",
                    subqueries=[
                        SubQueryItem(id="1", query=f"What information is available about: {normalized_query}?"),
                        SubQueryItem(id="2", query=f"What are the key details regarding: {normalized_query}?")
                    ],
                    relevant_parent_summaries=[],
                    aggregated_parent_summaries=[]
                )

            if model.__name__ == "SubQueryAnswer":
                subquery_id = "unknown"
                subquery = "unknown"
                match_id = re.search(r'"subquery_id": "(.*?)"', prompt)
                if match_id:
                    subquery_id = match_id.group(1)
                match_query = re.search(r'Question: (.*)', prompt)
                if match_query:
                    subquery = match_query.group(1).strip()
                
                return SubQueryAnswer(
                    subquery_id=subquery_id,
                    subquery=subquery,
                    answer=f"Unable to find specific information about: {subquery}",
                    used_parent_summaries=[]
                )

            if model.__name__ == "FinalAnswer":
                return FinalAnswer(
                    final_answer="Based on the available information, an error occurred during processing.",
                    subquery_answers=[],
                    aggregated_parent_summaries=[],
                    notes="Fallback due to generation error"
                )

            raise e

    def generate_refined_answer(self, query: str, comprehensive_context: str, chat_history: List = [], use_xml: bool = True) -> str:
        """Generate refined answer with XML structured processing for enhanced quality."""
        
        if use_xml:
            refined_xml_template = """You are an expert medical AI assistant specializing in comprehensive medical analysis and synthesis.

**MISSION:** Provide the most accurate, detailed, and clinically relevant answer by synthesizing ALL available information.

**COMPREHENSIVE MEDICAL CONTEXT:**
{comprehensive_context}

**CHAT HISTORY:**
{chat_history}

**PATIENT QUESTION:**
{question}

**SYNTHESIS REQUIREMENTS:**
1. **Primary Focus:** Prioritize information most directly relevant to the patient question
2. **Comprehensive Integration:** Seamlessly weave together primary and supplementary context
3. **Clinical Accuracy:** Maintain precise medical terminology and exact values
4. **Evidence-Based:** Only include information supported by the provided context
5. **Professional Structure:** Organize information logically for clinical understanding
6. **Limitation Transparency:** Clearly acknowledge gaps in available information

**XML RESPONSE FORMAT:**
<comprehensive_medical_response>
    <executive_summary>
        [Concise overview addressing the main question directly]
    </executive_summary>
    
    <detailed_analysis>
        <primary_findings>
            [Core information directly answering the question]
        </primary_findings>
        
        <supporting_information>
            [Additional context and related findings that enhance understanding]
        </supporting_information>
        
        <clinical_interpretation>
            [Professional medical interpretation of the findings]
        </clinical_interpretation>
    </detailed_analysis>
    
    <synthesis_conclusion>
        [Integrated conclusion drawing together all relevant information]
    </synthesis_conclusion>
    
    <information_gaps>
        [What additional information would be valuable for complete assessment]
    </information_gaps>
    
    <medical_disclaimer>
        This comprehensive analysis is for informational purposes only and does not constitute medical advice. Clinical correlation and professional medical consultation are recommended.
    </medical_disclaimer>
</comprehensive_medical_response>

Generate your comprehensive medical response using this XML structure."""

            history_str = "\n".join(
                [f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history]
            )
            
            prompt = refined_xml_template.format(
                comprehensive_context=comprehensive_context,
                question=query,
                chat_history=history_str
            )
            
            try:
                response = self.llm.invoke(prompt)
                if isinstance(response, dict) and 'content' in response:
                    xml_content = response['content']
                else:
                    xml_content = response.content if hasattr(response, 'content') else str(response)
                
                # Extract and construct refined response from XML
                refined_response = self._extract_comprehensive_response_from_xml(xml_content)
                return str(refined_response).encode('utf-8', 'ignore').decode('utf-8')
                
            except Exception as e:
                logger.error(f"XML refined answer generation failed: {e}, falling back to standard")
                use_xml = False
        
        if not use_xml:
            # Fallback to standard refined template
            refined_template = """You are an expert medical AI assistant. Your primary goal is to provide a clear, accurate, and synthesized answer based *strictly* on the provided context.

**Instructions:**
1.  **Prioritize Primary Context:** Base your main answer on the `Primary Context` section. This contains the most relevant information from the original query.
2.  **Use Supplementary Context:** Use the `Supplementary Context` to add detail, nuance, or related information to your main answer. Do not introduce it as a separate topic.
3.  **Synthesize, Don't List:** Do not list the context. Synthesize the information into a coherent, easy-to-read response.
4.  **Acknowledge Limitations:** If the context does not provide an answer, state that clearly.
5.  **Mandatory Disclaimer:** Always conclude your entire response with the following disclaimer on a new line: `This information is for informational purposes only and does not constitute medical advice.`

---

**CHAT HISTORY:**
{chat_history}

**COMPREHENSIVE CONTEXT:**
{comprehensive_context}

**QUESTION:**
{question}

---

**YOUR EXPERT ANSWER:**
"""
            history_str = "\n".join(
                [f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history]
            )
            prompt = refined_template.format(
                comprehensive_context=comprehensive_context,
                question=query,
                chat_history=history_str
            )
            try:
                response = self.llm.invoke(prompt)
                if isinstance(response, dict) and 'content' in response:
                    content = response['content']
                else:
                    content = response.content if hasattr(response, 'content') else str(response)
                return str(content).encode('utf-8', 'ignore').decode('utf-8')
            except Exception as e:
                logger.error(f"LLM invocation for refined answer failed: {e}")
                return "I am sorry, but I was unable to generate a refined response. Please try again later."
    
    def _extract_comprehensive_response_from_xml(self, xml_response: str) -> str:
        """Extract and construct comprehensive response from XML."""
        try:
            import xml.etree.ElementTree as ET
            
            # Clean XML response
            xml_response = re.sub(r'```xml\s*|\s*```', '', xml_response)
            xml_response = re.sub(r'```\s*|\s*```', '', xml_response)
            
            # Parse XML
            root = ET.fromstring(xml_response)
            
            # Extract components
            components = {}
            
            # Handle nested structure
            for child in root:
                if child.tag == 'detailed_analysis':
                    analysis_parts = {}
                    for analysis_child in child:
                        if analysis_child.text and analysis_child.text.strip():
                            analysis_parts[analysis_child.tag] = analysis_child.text.strip()
                    components['detailed_analysis'] = analysis_parts
                elif child.text and child.text.strip():
                    components[child.tag] = child.text.strip()
            
            # Construct comprehensive response
            response_parts = []
            
            # Executive summary
            if 'executive_summary' in components:
                response_parts.append(components['executive_summary'])
            
            # Detailed analysis
            if 'detailed_analysis' in components:
                analysis = components['detailed_analysis']
                
                if 'primary_findings' in analysis:
                    response_parts.append(f"\n**Primary Findings:**\n{analysis['primary_findings']}")
                
                if 'supporting_information' in analysis:
                    response_parts.append(f"\n**Supporting Information:**\n{analysis['supporting_information']}")
                
                if 'clinical_interpretation' in analysis:
                    response_parts.append(f"\n**Clinical Interpretation:**\n{analysis['clinical_interpretation']}")
            
            # Synthesis conclusion
            if 'synthesis_conclusion' in components:
                response_parts.append(f"\n**Conclusion:**\n{components['synthesis_conclusion']}")
            
            # Information gaps
            if 'information_gaps' in components:
                response_parts.append(f"\n**Additional Information Needed:**\n{components['information_gaps']}")
            
            # Medical disclaimer
            if 'medical_disclaimer' in components:
                response_parts.append(f"\n\n{components['medical_disclaimer']}")
            else:
                response_parts.append("\n\nThis comprehensive analysis is for informational purposes only and does not constitute medical advice. Clinical correlation and professional medical consultation are recommended.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Comprehensive XML parsing error: {e}")
            # Fallback regex extraction
            return self._extract_comprehensive_text_fallback(xml_response)
    
    def _extract_comprehensive_text_fallback(self, xml_response: str) -> str:
        """Fallback extraction for comprehensive response."""
        try:
            # Extract key sections with regex
            summary_match = re.search(r'<executive_summary>(.*?)</executive_summary>', xml_response, re.DOTALL)
            conclusion_match = re.search(r'<synthesis_conclusion>(.*?)</synthesis_conclusion>', xml_response, re.DOTALL)
            disclaimer_match = re.search(r'<medical_disclaimer>(.*?)</medical_disclaimer>', xml_response, re.DOTALL)
            
            parts = []
            
            if summary_match:
                parts.append(summary_match.group(1).strip())
            
            if conclusion_match:
                parts.append(f"\n**Conclusion:**\n{conclusion_match.group(1).strip()}")
            
            if disclaimer_match:
                parts.append(f"\n\n{disclaimer_match.group(1).strip()}")
            else:
                parts.append("\n\nThis information is for informational purposes only and does not constitute medical advice.")
            
            return "\n".join(parts) if parts else re.sub(r'<[^>]+>', '', xml_response).strip()
            
        except Exception as e:
            logger.error(f"Fallback comprehensive parsing failed: {e}")
            return re.sub(r'<[^>]+>', '', xml_response).strip() or "Unable to process the medical information properly."
