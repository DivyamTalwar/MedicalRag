import re
from typing import Dict, List

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

    def flatten_entities(self, entities: Dict) -> List[str]:
        return [item for sublist in entities.values() for item in sublist if item]
