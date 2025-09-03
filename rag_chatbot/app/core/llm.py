import json
import re
import asyncio
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Any, Dict, Optional, Type, TypeVar, List
from pydantic import BaseModel, ValidationError
import logging
from . import config
import requests

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class OmegaLLM:
    def __init__(self):
        self.endpoint = config.LLM_ENDPOINT
        self.api_key = config.MODELS_API_KEY
        self.model = "omega"

    async def ainvoke(self, prompt: str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: requests.post(self.endpoint, headers=headers, json=data, timeout=1200))
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0 and "message" in result["choices"][0] and "content" in result["choices"][0]["message"]:
            return type('obj', (object,), {'content': result["choices"][0]["message"]["content"]})()
        else:
            raise ValueError("Unexpected response format from LLM API")

class JSONExtractionSystem:
    def __init__(self, llm, max_retries: int = 5):
        self.llm = llm
        self.max_retries = max_retries
    
    async def get_validated_response(
        self, 
        prompt: str, 
        model_class: Type[T],
        context: Optional[str] = None
    ) -> T:
        try:
            return await self._strategy_structured_output(prompt, model_class)
        except Exception as e:
            logger.warning(f"Strategy 1 failed: {e}")
        
        try:
            return await self._strategy_enhanced_prompt(prompt, model_class, context)
        except Exception as e:
            logger.warning(f"Strategy 2 failed: {e}")
        
        try:
            return await self._strategy_regex_fix(prompt, model_class)
        except Exception as e:
            logger.warning(f"Strategy 3 failed: {e}")
        
        try:
            return await self._strategy_llm_json_fixer(prompt, model_class)
        except Exception as e:
            logger.warning(f"Strategy 4 failed: {e}")
        
        try:
            return await self._strategy_partial_extraction(prompt, model_class)
        except Exception as e:
            logger.warning(f"Strategy 5 failed: {e}")
        
        return self._create_default_object(model_class)
    
    async def _strategy_structured_output(self, prompt: str, model_class: Type[T]) -> T:
        schema = model_class.model_json_schema()
        structured_prompt = f"""
{prompt}

You MUST respond with valid JSON that matches this exact schema:
{json.dumps(schema, indent=2)}

Important rules:
- Output ONLY valid JSON, no markdown, no explanations
- All required fields must be present
- Use exact field names from the schema
- Ensure proper data types (strings in quotes, numbers without quotes)

JSON Output:
"""
        
        response = await self.llm.ainvoke(structured_prompt)
        json_str = self._extract_json_from_response(response.content)
        parsed = json.loads(json_str)
        return model_class.model_validate(parsed)
    
    async def _strategy_enhanced_prompt(self, prompt: str, model_class: Type[T], context: Optional[str]) -> T:
        sample = self._create_sample_json(model_class)
        
        enhanced_prompt = f"""
{prompt}

CRITICAL: You must output ONLY valid JSON. Here's an example of the correct format:
{json.dumps(sample, indent=2)}

Rules:
1. Start your response with {{ and end with }}
2. No markdown code blocks (no ```)
3. No explanatory text before or after
4. All strings must be in double quotes
5. No trailing commas
6. No comments in the JSON

Context for accuracy: {context or 'N/A'}

Valid JSON Output:
"""
        
        response = await self.llm.ainvoke(enhanced_prompt)
        json_str = self._extract_json_from_response(response.content)
        parsed = json.loads(json_str)
        return model_class.model_validate(parsed)
    
    async def _strategy_regex_fix(self, prompt: str, model_class: Type[T]) -> T:
        response = await self.llm.ainvoke(prompt)
        content = response.content
        fixed_json = self._apply_json_fixes(content)
        parsed = json.loads(fixed_json)
        return model_class.model_validate(parsed)
    
    def _apply_json_fixes(self, content: str) -> str:
        content = re.sub(r'```json\s*|\s*```', '', content)
        content = re.sub(r'```\s*|\s*```', '', content)
        json_match = re.search(r'(\{[^{}]*\{.*\}[^{}]*\}|\{.*\}|\[.*\])', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        
        fixes = [
            (r',\s*}', '}'), (r',\s*]', ']'), (r"'([^']*)'", r'"\1"'),
            (r'//.*?$', '', re.MULTILINE), (r'/\*.*?\*/', '', re.DOTALL),
            (r'(\w+):', r'"\1":'), (r':\s*None', ': null'), (r':\s*True', ': true'),
            (r':\s*False', ': false'), (r'^\ufeff', ''), (r'\u200b', ''), (r'\\{2,}"', r'\"'),
        ]
        
        for pattern, replacement, *flags in fixes:
            flag = flags[0] if flags else 0
            content = re.sub(pattern, replacement, content, flags=flag)
        
        content = content.strip()
        if not content.startswith('{') and not content.startswith('['):
            content = '{' + content
        if not content.endswith('}') and not content.endswith(']'):
            content = content + '}'
        
        return content
    
    async def _strategy_llm_json_fixer(self, original_prompt: str, model_class: Type[T]) -> T:
        response = await self.llm.ainvoke(original_prompt)
        broken_json = response.content
        schema = model_class.model_json_schema()
        
        fix_prompt = f"""
The following text contains JSON data but it might be malformed or have issues.
Please fix it and return ONLY valid JSON that matches this schema:

Schema:
{json.dumps(schema, indent=2)}

Broken JSON to fix:
{broken_json}

Rules for fixing:
1. Preserve all data from the original
2. Fix any syntax errors (quotes, commas, brackets)
3. Ensure all required fields are present (use reasonable defaults if missing)
4. Remove any non-JSON text
5. Output ONLY the fixed JSON, nothing else

Fixed JSON:
"""
        
        fixed_response = await self.llm.ainvoke(fix_prompt)
        json_str = self._extract_json_from_response(fixed_response.content)
        parsed = json.loads(json_str)
        return model_class.model_validate(parsed)
    
    async def _strategy_partial_extraction(self, prompt: str, model_class: Type[T]) -> T:
        response = await self.llm.ainvoke(prompt)
        content = response.content
        extracted_data = {}
        schema = model_class.model_json_schema()
        
        if 'properties' in schema:
            for field_name, field_info in schema['properties'].items():
                patterns = [
                    rf'"{field_name}"\s*:\s*"([^"]*)"', rf'"{field_name}"\s*:\s*(\d+\.?\d*)',
                    rf'"{field_name}"\s*:\s*(true|false)', rf'"{field_name}"\s*:\s*(\[[^\]]*\])',
                    rf'"{field_name}"\s*:\s*(\{{[^}}]*\}})', rf'{field_name}\s*:\s*"([^"]*)"',
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        value = match.group(1)
                        try:
                            if field_info.get('type') == 'integer':
                                extracted_data[field_name] = int(float(value))
                            elif field_info.get('type') == 'number':
                                extracted_data[field_name] = float(value)
                            elif field_info.get('type') == 'boolean':
                                extracted_data[field_name] = value.lower() == 'true'
                            elif field_info.get('type') == 'array':
                                extracted_data[field_name] = json.loads(value)
                            elif field_info.get('type') == 'object':
                                extracted_data[field_name] = json.loads(value)
                            else:
                                extracted_data[field_name] = value
                        except:
                            extracted_data[field_name] = value
                        break
        
        default_obj = self._create_default_object(model_class)
        default_dict = default_obj.model_dump()
        default_dict.update(extracted_data)
        return model_class.model_validate(default_dict)
    
    def _extract_json_from_response(self, content: str) -> str:
        patterns = [
            r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```',
            r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})', r'(\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\])',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return self._apply_json_fixes(content)
    
    def _create_sample_json(self, model_class: Type[T]) -> Dict[str, Any]:
        schema = model_class.model_json_schema()
        sample = {}
        
        if 'properties' in schema:
            for field_name, field_info in schema['properties'].items():
                field_type = field_info.get('type', 'string')
                if field_type == 'string': sample[field_name] = "example_value"
                elif field_type == 'integer': sample[field_name] = 0
                elif field_type == 'number': sample[field_name] = 0.0
                elif field_type == 'boolean': sample[field_name] = True
                elif field_type == 'array': sample[field_name] = []
                elif field_type == 'object': sample[field_name] = {}
                else: sample[field_name] = None
        
        return sample
    
    def _create_default_object(self, model_class: Type[T]) -> T:
        schema = model_class.model_json_schema()
        defaults = {}
        
        if 'properties' in schema:
            for field_name, field_info in schema['properties'].items():
                if 'default' in field_info:
                    defaults[field_name] = field_info['default']
                else:
                    field_type = field_info.get('type', 'string')
                    if field_type == 'string': defaults[field_name] = ""
                    elif field_type == 'integer': defaults[field_name] = 0
                    elif field_type == 'number': defaults[field_name] = 0.0
                    elif field_type == 'boolean': defaults[field_name] = False
                    elif field_type == 'array': defaults[field_name] = []
                    elif field_type == 'object': defaults[field_name] = {}
                    else: defaults[field_name] = None
        
        return model_class.model_validate(defaults)

class XMLExtractionSystem:
    """Advanced XML-based extraction system for structured data from LLM responses."""
    
    def __init__(self, llm, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries
    
    def _create_xml_schema_prompt(self, model_class: Type[T]) -> str:
        """Generate XML schema description from Pydantic model."""
        schema = model_class.model_json_schema()
        xml_structure = self._json_schema_to_xml_structure(schema)
        
        return f"""
        <response>
            {xml_structure}
        </response>
        
        **XML Rules:**
        1. ALL data must be enclosed in XML tags
        2. Use exact field names from schema as XML tag names
        3. No attributes - use nested tags for complex data
        4. Empty values should use self-closing tags: <field/>
        5. Arrays use repeated tags: <item>value1</item><item>value2</item>
        6. Boolean values: <field>true</field> or <field>false</field>
        7. Numbers without quotes: <field>42</field> or <field>3.14</field>
        """
    
    def _json_schema_to_xml_structure(self, schema: Dict[str, Any]) -> str:
        """Convert JSON schema to XML structure example."""
        if 'properties' not in schema:
            return "<data>content_here</data>"
        
        xml_parts = []
        for field_name, field_info in schema['properties'].items():
            field_type = field_info.get('type', 'string')
            
            if field_type == 'array':
                items_type = field_info.get('items', {}).get('type', 'string')
                if items_type == 'object':
                    xml_parts.append(f"<{field_name}><item><subfield>value</subfield></item></{field_name}>")
                else:
                    xml_parts.append(f"<{field_name}><item>value1</item><item>value2</item></{field_name}>")
            elif field_type == 'object':
                xml_parts.append(f"<{field_name}><subfield>value</subfield></{field_name}>")
            elif field_type == 'boolean':
                xml_parts.append(f"<{field_name}>true</{field_name}>")
            elif field_type in ['integer', 'number']:
                xml_parts.append(f"<{field_name}>42</{field_name}>")
            else:
                xml_parts.append(f"<{field_name}>example_value</{field_name}>")
        
        return '\n            '.join(xml_parts)
    
    def _extract_xml_from_response(self, content: str) -> str:
        """Extract XML content from LLM response."""
        # Remove markdown code blocks
        content = re.sub(r'```xml\s*|\s*```', '', content)
        content = re.sub(r'```\s*|\s*```', '', content)
        
        # Try to find XML within response tags
        response_match = re.search(r'<response>(.*?)</response>', content, re.DOTALL)
        if response_match:
            return f"<response>{response_match.group(1)}</response>"
        
        # Try to find any complete XML structure
        xml_match = re.search(r'<[^<>]+>.*</[^<>]+>', content, re.DOTALL)
        if xml_match:
            return xml_match.group(0)
        
        # If no XML found, wrap content in generic tags
        cleaned_content = content.strip()
        if not cleaned_content.startswith('<'):
            return f"<response>{cleaned_content}</response>"
        
        return cleaned_content
    
    def _xml_to_json(self, xml_content: str) -> Dict[str, Any]:
        """Convert XML to JSON dictionary."""
        try:
            root = ET.fromstring(xml_content)
            return self._element_to_dict(root)
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
            # Try to fix common XML issues
            fixed_xml = self._fix_xml_structure(xml_content)
            try:
                root = ET.fromstring(fixed_xml)
                return self._element_to_dict(root)
            except ET.ParseError:
                logger.error("Failed to parse XML even after fixes")
                return self._extract_data_with_regex(xml_content)
    
    def _fix_xml_structure(self, xml_content: str) -> str:
        """Fix common XML formatting issues."""
        # Remove invalid characters
        xml_content = re.sub(r'[^\x09\x0A\x0D\x20-\x7E]', '', xml_content)
        
        # Fix unclosed tags
        xml_content = re.sub(r'<([^/>]+)>\s*$', r'<\1></\1>', xml_content, flags=re.MULTILINE)
        
        # Fix boolean values
        xml_content = re.sub(r'<([^>]+)>True</([^>]+)>', r'<\1>true</\2>', xml_content)
        xml_content = re.sub(r'<([^>]+)>False</([^>]+)>', r'<\1>false</\2>', xml_content)
        
        # Ensure proper root tag
        if not xml_content.strip().startswith('<'):
            xml_content = f"<root>{xml_content}</root>"
        
        return xml_content
    
    def _element_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert XML Element to dictionary."""
        result = {}
        
        # Handle element text
        if element.text and element.text.strip():
            text = element.text.strip()
            # Convert to appropriate type
            if text.lower() in ['true', 'false']:
                result['_text'] = text.lower() == 'true'
            elif text.isdigit():
                result['_text'] = int(text)
            elif re.match(r'^\d*\.\d+$', text):
                result['_text'] = float(text)
            else:
                result['_text'] = text
        
        # Handle child elements
        for child in element:
            child_data = self._element_to_dict(child)
            
            if child.tag in result:
                # Handle arrays (repeated tags)
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                if child_data.get('_text') is not None:
                    result[child.tag].append(child_data['_text'])
                else:
                    result[child.tag].append(child_data)
            else:
                if child_data.get('_text') is not None and len(child_data) == 1:
                    result[child.tag] = child_data['_text']
                else:
                    result[child.tag] = child_data if child_data else ""
        
        return result
    
    def _extract_data_with_regex(self, xml_content: str) -> Dict[str, Any]:
        """Fallback: extract data using regex when XML parsing fails."""
        result = {}
        
        # Extract simple tag-value pairs
        simple_tags = re.findall(r'<([^/>]+)>([^<]+)</\1>', xml_content)
        for tag, value in simple_tags:
            # Convert to appropriate type
            if value.lower() in ['true', 'false']:
                result[tag] = value.lower() == 'true'
            elif value.isdigit():
                result[tag] = int(value)
            elif re.match(r'^\d*\.\d+$', value):
                result[tag] = float(value)
            else:
                result[tag] = value
        
        # Extract array-like structures
        array_pattern = r'<(\w+)>((?:<item>[^<]*</item>)+)</\1>'
        array_matches = re.findall(array_pattern, xml_content)
        for array_name, items_xml in array_matches:
            items = re.findall(r'<item>([^<]*)</item>', items_xml)
            result[array_name] = items
        
        return result
    
    async def get_structured_xml_response(
        self, 
        prompt: str, 
        model_class: Type[T],
        context: Optional[str] = None
    ) -> T:
        """Main method to get structured response via XML."""
        
        # Create XML-enhanced prompt
        xml_schema = self._create_xml_schema_prompt(model_class)
        
        enhanced_prompt = f"""
        {context or ''}
        
        {prompt}
        
        **CRITICAL INSTRUCTIONS:**
        You MUST respond in VALID XML format ONLY. No explanations, no markdown, no text outside XML tags.
        
        {xml_schema}
        
        Your response must be COMPLETE and VALID XML following the exact structure above.
        """
        
        for attempt in range(self.max_retries):
            try:
                # Get LLM response
                response = await self.llm.ainvoke(enhanced_prompt)
                xml_content = self._extract_xml_from_response(response.content)
                
                # Convert XML to JSON
                json_data = self._xml_to_json(xml_content)
                
                # Handle nested response wrapper
                if 'response' in json_data and isinstance(json_data['response'], dict):
                    json_data = json_data['response']
                
                # Validate with Pydantic
                validated_obj = model_class.model_validate(json_data)
                logger.info(f"Successfully created {model_class.__name__} from XML on attempt {attempt + 1}")
                return validated_obj
                
            except ValidationError as e:
                logger.warning(f"Validation failed on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    # Enhance prompt with validation error info
                    enhanced_prompt += f"\n\nPrevious attempt failed validation: {str(e)}\nPlease fix these issues in your XML response."
                else:
                    logger.error("All validation attempts failed, creating default object")
                    return self._create_default_object(model_class)
                    
            except Exception as e:
                logger.error(f"XML processing failed on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return self._create_default_object(model_class)
        
        return self._create_default_object(model_class)
    
    def _create_default_object(self, model_class: Type[T]) -> T:
        """Create default object when all parsing attempts fail."""
        schema = model_class.model_json_schema()
        defaults = {}
        
        if 'properties' in schema:
            for field_name, field_info in schema['properties'].items():
                field_type = field_info.get('type', 'string')
                if 'default' in field_info:
                    defaults[field_name] = field_info['default']
                elif field_type == 'string':
                    defaults[field_name] = ""
                elif field_type == 'integer':
                    defaults[field_name] = 0
                elif field_type == 'number':
                    defaults[field_name] = 0.0
                elif field_type == 'boolean':
                    defaults[field_name] = False
                elif field_type == 'array':
                    defaults[field_name] = []
                elif field_type == 'object':
                    defaults[field_name] = {}
                else:
                    defaults[field_name] = None
        
        return model_class.model_validate(defaults)

CustomLLM = OmegaLLM
