#!/usr/bin/env python3
"""
MEDICAL KNOWLEDGE GRAPH SYSTEM
Building Semantic Medical Intelligence
"""

import networkx as nx
from typing import List, Dict, Tuple, Set, Any
import json
from dataclasses import dataclass

@dataclass
class MedicalRelationship:
    """Represents a medical relationship in the knowledge graph"""
    source: str
    target: str
    relationship_type: str
    weight: float
    evidence: List[str] = None

class MedicalKnowledgeGraph:
    """
    LEGENDARY KNOWLEDGE GRAPH OVERLAY
    Enhances retrieval with medical intelligence
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_comprehensive_graph()
    
    def _build_comprehensive_graph(self):
        """Build the complete medical knowledge graph"""
        
        # SYMPTOM → DISEASE RELATIONSHIPS
        symptom_disease_relationships = [
            ("chest_pain", "myocardial_infarction", 0.9),
            ("chest_pain", "angina", 0.85),
            ("chest_pain", "pericarditis", 0.6),
            ("shortness_of_breath", "heart_failure", 0.8),
            ("shortness_of_breath", "pneumonia", 0.75),
            ("shortness_of_breath", "pulmonary_embolism", 0.7),
            ("fever", "infection", 0.9),
            ("fever", "sepsis", 0.6),
            ("cough", "bronchitis", 0.8),
            ("cough", "pneumonia", 0.85),
            ("headache", "hypertension", 0.5),
            ("headache", "migraine", 0.9),
            ("fatigue", "anemia", 0.7),
            ("fatigue", "hypothyroidism", 0.65),
            ("palpitations", "arrhythmia", 0.9),
            ("palpitations", "anxiety", 0.6),
            ("edema", "heart_failure", 0.8),
            ("edema", "kidney_disease", 0.75),
            ("jaundice", "liver_disease", 0.9),
            ("jaundice", "hepatitis", 0.85)
        ]
        
        for symptom, disease, weight in symptom_disease_relationships:
            self.graph.add_edge(symptom, disease, 
                              relationship="indicates", 
                              weight=weight,
                              bidirectional=False)
        
        # DISEASE → TREATMENT RELATIONSHIPS
        disease_treatment_relationships = [
            ("myocardial_infarction", "aspirin", 1.0),
            ("myocardial_infarction", "clopidogrel", 0.95),
            ("myocardial_infarction", "heparin", 0.9),
            ("myocardial_infarction", "beta_blocker", 0.85),
            ("hypertension", "lisinopril", 0.9),
            ("hypertension", "amlodipine", 0.85),
            ("hypertension", "metoprolol", 0.8),
            ("diabetes_type2", "metformin", 0.95),
            ("diabetes_type2", "insulin", 0.7),
            ("diabetes_type2", "glipizide", 0.75),
            ("heart_failure", "furosemide", 0.9),
            ("heart_failure", "spironolactone", 0.8),
            ("pneumonia", "antibiotics", 0.95),
            ("pneumonia", "azithromycin", 0.9),
            ("asthma", "albuterol", 0.95),
            ("asthma", "inhaled_corticosteroids", 0.9),
            ("hyperlipidemia", "statin", 0.95),
            ("hyperlipidemia", "atorvastatin", 0.9),
            ("depression", "ssri", 0.85),
            ("anxiety", "benzodiazepine", 0.7)
        ]
        
        for disease, treatment, weight in disease_treatment_relationships:
            self.graph.add_edge(disease, treatment,
                              relationship="treated_by",
                              weight=weight,
                              bidirectional=False)
        
        # TEST → RESULT INTERPRETATION RELATIONSHIPS
        test_result_relationships = [
            ("troponin_elevated", "cardiac_damage", 0.95),
            ("troponin_elevated", "myocardial_infarction", 0.9),
            ("bnp_elevated", "heart_failure", 0.85),
            ("d_dimer_elevated", "pulmonary_embolism", 0.8),
            ("wbc_elevated", "infection", 0.85),
            ("wbc_elevated", "leukemia", 0.4),
            ("hemoglobin_low", "anemia", 0.95),
            ("glucose_elevated", "diabetes", 0.9),
            ("creatinine_elevated", "kidney_disease", 0.9),
            ("ast_elevated", "liver_disease", 0.8),
            ("alt_elevated", "hepatitis", 0.85),
            ("tsh_elevated", "hypothyroidism", 0.9),
            ("tsh_low", "hyperthyroidism", 0.9),
            ("ldl_elevated", "hyperlipidemia", 0.85),
            ("psa_elevated", "prostate_cancer", 0.6)
        ]
        
        for test, result, weight in test_result_relationships:
            self.graph.add_edge(test, result,
                              relationship="confirms",
                              weight=weight,
                              bidirectional=False)
        
        # MEDICATION → SIDE EFFECT RELATIONSHIPS
        medication_side_effects = [
            ("aspirin", "gi_bleeding", 0.3),
            ("warfarin", "bleeding", 0.4),
            ("statins", "myalgia", 0.2),
            ("ace_inhibitors", "cough", 0.15),
            ("beta_blockers", "fatigue", 0.25),
            ("diuretics", "hypokalemia", 0.3),
            ("metformin", "gi_upset", 0.3),
            ("insulin", "hypoglycemia", 0.4),
            ("antibiotics", "diarrhea", 0.25),
            ("nsaids", "kidney_damage", 0.2)
        ]
        
        for med, side_effect, weight in medication_side_effects:
            self.graph.add_edge(med, side_effect,
                              relationship="causes",
                              weight=weight,
                              bidirectional=False)
        
        # DISEASE → COMPLICATION RELATIONSHIPS
        disease_complications = [
            ("diabetes", "neuropathy", 0.4),
            ("diabetes", "retinopathy", 0.35),
            ("diabetes", "nephropathy", 0.3),
            ("hypertension", "stroke", 0.5),
            ("hypertension", "heart_failure", 0.4),
            ("myocardial_infarction", "arrhythmia", 0.6),
            ("heart_failure", "pulmonary_edema", 0.5),
            ("cirrhosis", "ascites", 0.7),
            ("cirrhosis", "variceal_bleeding", 0.6)
        ]
        
        for disease, complication, weight in disease_complications:
            self.graph.add_edge(disease, complication,
                              relationship="leads_to",
                              weight=weight,
                              bidirectional=False)
        
        # CONTRAINDICATION RELATIONSHIPS
        contraindications = [
            ("aspirin", "bleeding_disorder", 1.0),
            ("metformin", "kidney_failure", 1.0),
            ("beta_blockers", "asthma", 0.9),
            ("ace_inhibitors", "pregnancy", 1.0),
            ("warfarin", "pregnancy", 1.0),
            ("statins", "pregnancy", 0.95)
        ]
        
        for med, condition, weight in contraindications:
            self.graph.add_edge(med, condition,
                              relationship="contraindicated_in",
                              weight=weight,
                              bidirectional=False)
    
    def expand_query(self, query_entities: List[str], max_hops: int = 2) -> Set[str]:
        """
        Expand query using knowledge graph relationships
        Follow edges to find related concepts
        """
        expanded_entities = set(query_entities)
        
        for entity in query_entities:
            if entity in self.graph:
                # Get neighbors within max_hops
                for neighbor in nx.single_source_shortest_path(self.graph, entity, cutoff=max_hops):
                    expanded_entities.add(neighbor)
                
                # Also get predecessors (reverse relationships)
                for predecessor in self.graph.predecessors(entity):
                    expanded_entities.add(predecessor)
        
        return expanded_entities
    
    def validate_medical_logic(self, entities: List[str]) -> Dict[str, Any]:
        """
        Validate medical logic in retrieved chunks
        Check for consistency and contradictions
        """
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'confidence': 1.0
        }
        
        # Check for contraindications
        medications = [e for e in entities if e in self.graph and 
                      any(self.graph[e][n]['relationship'] == 'contraindicated_in' 
                          for n in self.graph[e])]
        
        conditions = [e for e in entities if e in self.graph]
        
        for med in medications:
            for condition in conditions:
                if self.graph.has_edge(med, condition):
                    edge_data = self.graph[med][condition]
                    if edge_data.get('relationship') == 'contraindicated_in':
                        validation_result['errors'].append(
                            f"CONTRAINDICATION: {med} is contraindicated in {condition}"
                        )
                        validation_result['is_valid'] = False
                        validation_result['confidence'] *= 0.5
        
        # Check for unlikely combinations
        symptoms = [e for e in entities if any(
            self.graph.has_edge(e, n) and self.graph[e][n]['relationship'] == 'indicates'
            for n in self.graph[e] if e in self.graph
        )]
        
        diseases = [e for e in entities if any(
            self.graph.has_edge(n, e) and self.graph[n][e]['relationship'] == 'indicates'
            for n in self.graph.predecessors(e) if e in self.graph
        )]
        
        # Validate symptom-disease consistency
        for symptom in symptoms:
            if symptom in self.graph:
                indicated_diseases = [n for n in self.graph[symptom] 
                                    if self.graph[symptom][n]['relationship'] == 'indicates']
                
                if diseases and not any(d in indicated_diseases for d in diseases):
                    validation_result['warnings'].append(
                        f"WARNING: {symptom} typically doesn't indicate {diseases}"
                    )
                    validation_result['confidence'] *= 0.8
        
        return validation_result
    
    def multi_hop_reasoning(self, start_entity: str, target_entity: str) -> List[List[str]]:
        """
        Find all paths between two medical entities
        Useful for complex medical reasoning
        """
        if start_entity not in self.graph or target_entity not in self.graph:
            return []
        
        try:
            # Find all simple paths (no cycles)
            all_paths = list(nx.all_simple_paths(
                self.graph, start_entity, target_entity, cutoff=4
            ))
            
            # Sort by path length (shorter is usually more direct)
            all_paths.sort(key=len)
            
            return all_paths[:5]  # Return top 5 paths
        except nx.NetworkXNoPath:
            return []
    
    def get_treatment_options(self, disease: str) -> List[Tuple[str, float]]:
        """Get all treatment options for a disease with confidence scores"""
        if disease not in self.graph:
            return []
        
        treatments = []
        for neighbor in self.graph[disease]:
            edge_data = self.graph[disease][neighbor]
            if edge_data.get('relationship') == 'treated_by':
                treatments.append((neighbor, edge_data.get('weight', 0.5)))
        
        # Sort by weight (confidence)
        treatments.sort(key=lambda x: x[1], reverse=True)
        return treatments
    
    def get_differential_diagnosis(self, symptoms: List[str]) -> List[Tuple[str, float]]:
        """Get differential diagnosis based on symptoms"""
        disease_scores = {}
        
        for symptom in symptoms:
            if symptom in self.graph:
                for disease in self.graph[symptom]:
                    edge_data = self.graph[symptom][disease]
                    if edge_data.get('relationship') == 'indicates':
                        weight = edge_data.get('weight', 0.5)
                        if disease in disease_scores:
                            disease_scores[disease] = max(disease_scores[disease], weight)
                        else:
                            disease_scores[disease] = weight
        
        # Sort by score
        differential = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        return differential
    
    def check_drug_interactions(self, medications: List[str]) -> List[Dict[str, Any]]:
        """Check for potential drug interactions"""
        interactions = []
        
        # For this example, we check if medications share side effects
        # In production, this would use a drug interaction database
        for i, med1 in enumerate(medications):
            for med2 in medications[i+1:]:
                shared_effects = set()
                
                if med1 in self.graph and med2 in self.graph:
                    effects1 = {n for n in self.graph[med1] 
                               if self.graph[med1][n].get('relationship') == 'causes'}
                    effects2 = {n for n in self.graph[med2] 
                               if self.graph[med2][n].get('relationship') == 'causes'}
                    
                    shared_effects = effects1.intersection(effects2)
                    
                    if shared_effects:
                        interactions.append({
                            'drug1': med1,
                            'drug2': med2,
                            'shared_effects': list(shared_effects),
                            'severity': 'moderate'
                        })
        
        return interactions
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': {
                'symptoms': len([n for n in self.graph.nodes() if '_' not in n]),
                'diseases': len([n for n in self.graph.nodes() if 'disease' in n or 'itis' in n]),
                'medications': len([n for n in self.graph.nodes() if any(
                    self.graph[n][neighbor].get('relationship') == 'causes'
                    for neighbor in self.graph[n]
                ) if n in self.graph])
            },
            'relationship_types': list(set(
                data['relationship'] for _, _, data in self.graph.edges(data=True)
            )),
            'average_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        }


def demonstrate_knowledge_graph():
    """Demonstration of the medical knowledge graph"""
    
    print("\n" + "="*80)
    print("MEDICAL KNOWLEDGE GRAPH DEMONSTRATION")
    print("="*80)
    
    # Create knowledge graph
    kg = MedicalKnowledgeGraph()
    
    # Get statistics
    stats = kg.get_graph_statistics()
    print(f"\nGraph Statistics:")
    print(f"  Total Nodes: {stats['total_nodes']}")
    print(f"  Total Edges: {stats['total_edges']}")
    print(f"  Relationship Types: {', '.join(stats['relationship_types'])}")
    
    # Test query expansion
    print("\n1. QUERY EXPANSION:")
    query_entities = ["chest_pain", "troponin_elevated"]
    expanded = kg.expand_query(query_entities, max_hops=2)
    print(f"  Original: {query_entities}")
    print(f"  Expanded: {list(expanded)[:10]}...")
    
    # Test differential diagnosis
    print("\n2. DIFFERENTIAL DIAGNOSIS:")
    symptoms = ["chest_pain", "shortness_of_breath", "palpitations"]
    differential = kg.get_differential_diagnosis(symptoms)
    print(f"  Symptoms: {symptoms}")
    print(f"  Possible Diagnoses:")
    for disease, score in differential[:5]:
        print(f"    - {disease}: {score:.2f}")
    
    # Test treatment options
    print("\n3. TREATMENT OPTIONS:")
    disease = "myocardial_infarction"
    treatments = kg.get_treatment_options(disease)
    print(f"  Disease: {disease}")
    print(f"  Treatments:")
    for treatment, confidence in treatments:
        print(f"    - {treatment}: {confidence:.2f}")
    
    # Test multi-hop reasoning
    print("\n4. MULTI-HOP REASONING:")
    paths = kg.multi_hop_reasoning("chest_pain", "aspirin")
    print(f"  Finding path from 'chest_pain' to 'aspirin':")
    for i, path in enumerate(paths[:3], 1):
        print(f"    Path {i}: {' → '.join(path)}")
    
    # Test medical validation
    print("\n5. MEDICAL LOGIC VALIDATION:")
    entities = ["aspirin", "bleeding_disorder", "myocardial_infarction"]
    validation = kg.validate_medical_logic(entities)
    print(f"  Entities: {entities}")
    print(f"  Valid: {validation['is_valid']}")
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH READY - MEDICAL INTELLIGENCE ACTIVATED!")
    print("="*80)


if __name__ == "__main__":
    demonstrate_knowledge_graph()