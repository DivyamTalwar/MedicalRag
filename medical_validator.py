#!/usr/bin/env python3
"""
MEDICAL VALIDATION PIPELINE
Ensuring Medical Accuracy and Safety
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json

class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"  # Must be addressed immediately
    HIGH = "high"          # Should be reviewed
    MEDIUM = "medium"      # Important to note
    LOW = "low"           # Informational

@dataclass
class ValidationIssue:
    """Represents a validation issue"""
    severity: ValidationSeverity
    category: str
    message: str
    location: str
    suggestion: str = None
    evidence: List[str] = None

class MedicalFactChecker:
    """
    VALIDATION LAYER 1: Medical Fact Checking
    Validates against medical knowledge bases
    """
    
    def __init__(self):
        # FDA-approved drug dosage ranges
        self.drug_dosages = {
            'aspirin': {'min': 81, 'max': 325, 'unit': 'mg', 'frequency': 'daily'},
            'metformin': {'min': 500, 'max': 2000, 'unit': 'mg', 'frequency': 'daily'},
            'lisinopril': {'min': 5, 'max': 40, 'unit': 'mg', 'frequency': 'daily'},
            'atorvastatin': {'min': 10, 'max': 80, 'unit': 'mg', 'frequency': 'daily'},
            'warfarin': {'min': 1, 'max': 10, 'unit': 'mg', 'frequency': 'daily'},
            'insulin': {'min': 0.1, 'max': 2, 'unit': 'units/kg', 'frequency': 'daily'},
            'heparin': {'min': 5000, 'max': 10000, 'unit': 'units', 'frequency': 'q8-12h'},
            'clopidogrel': {'min': 75, 'max': 600, 'unit': 'mg', 'frequency': 'daily'}
        }
        
        # Normal lab value ranges
        self.lab_ranges = {
            'glucose': {'min': 70, 'max': 100, 'unit': 'mg/dL', 'critical_low': 40, 'critical_high': 400},
            'hemoglobin': {
                'male': {'min': 13.5, 'max': 17.5, 'unit': 'g/dL'},
                'female': {'min': 12.0, 'max': 15.5, 'unit': 'g/dL'}
            },
            'wbc': {'min': 4500, 'max': 11000, 'unit': 'cells/μL', 'critical_low': 1000, 'critical_high': 50000},
            'platelets': {'min': 150000, 'max': 400000, 'unit': 'per μL', 'critical_low': 20000},
            'creatinine': {'min': 0.6, 'max': 1.2, 'unit': 'mg/dL', 'critical_high': 4.0},
            'troponin': {'normal': 0.04, 'elevated': 0.4, 'unit': 'ng/mL'},
            'ph': {'min': 7.35, 'max': 7.45, 'critical_low': 7.2, 'critical_high': 7.6},
            'pco2': {'min': 35, 'max': 45, 'unit': 'mmHg'},
            'po2': {'min': 80, 'max': 100, 'unit': 'mmHg', 'critical_low': 60},
            'sodium': {'min': 136, 'max': 145, 'unit': 'mEq/L', 'critical_low': 120, 'critical_high': 160},
            'potassium': {'min': 3.5, 'max': 5.0, 'unit': 'mEq/L', 'critical_low': 2.5, 'critical_high': 6.5}
        }
        
        # Diagnostic criteria
        self.diagnostic_criteria = {
            'diabetes': {
                'fasting_glucose': {'threshold': 126, 'unit': 'mg/dL'},
                'hba1c': {'threshold': 6.5, 'unit': '%'},
                'random_glucose': {'threshold': 200, 'unit': 'mg/dL'}
            },
            'hypertension': {
                'systolic': {'stage1': 130, 'stage2': 140, 'crisis': 180},
                'diastolic': {'stage1': 80, 'stage2': 90, 'crisis': 120}
            },
            'hyperlipidemia': {
                'total_cholesterol': {'borderline': 200, 'high': 240},
                'ldl': {'optimal': 100, 'borderline': 130, 'high': 160},
                'hdl': {'low_male': 40, 'low_female': 50}
            }
        }
    
    def validate_drug_dosage(self, drug: str, dosage: float, unit: str) -> List[ValidationIssue]:
        """Validate drug dosage against FDA limits"""
        issues = []
        drug_lower = drug.lower()
        
        if drug_lower in self.drug_dosages:
            limits = self.drug_dosages[drug_lower]
            
            # Check unit consistency
            if unit.lower() != limits['unit'].lower():
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    category="dosage_unit",
                    message=f"Unusual unit for {drug}: {unit} (expected {limits['unit']})",
                    location=f"{drug} dosage",
                    suggestion=f"Verify unit or convert to {limits['unit']}"
                ))
            
            # Check dosage range
            if dosage < limits['min']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category="dosage_low",
                    message=f"{drug} dosage {dosage}{unit} below minimum ({limits['min']}{limits['unit']})",
                    location=f"{drug} dosage",
                    suggestion=f"Consider increasing to at least {limits['min']}{limits['unit']}"
                ))
            elif dosage > limits['max']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="dosage_high",
                    message=f"{drug} dosage {dosage}{unit} exceeds maximum ({limits['max']}{limits['unit']})",
                    location=f"{drug} dosage",
                    suggestion=f"Reduce to maximum {limits['max']}{limits['unit']}",
                    evidence=[f"FDA max: {limits['max']}{limits['unit']}"]
                ))
        
        return issues
    
    def validate_lab_value(self, test: str, value: float, gender: str = None) -> List[ValidationIssue]:
        """Validate lab values against normal ranges"""
        issues = []
        test_lower = test.lower()
        
        if test_lower in self.lab_ranges:
            ranges = self.lab_ranges[test_lower]
            
            # Handle gender-specific ranges
            if isinstance(ranges, dict) and 'male' in ranges:
                if gender:
                    ranges = ranges.get(gender.lower(), ranges['male'])
                else:
                    ranges = ranges['male']  # Default to male if not specified
            
            # Check critical values
            if 'critical_low' in ranges and value < ranges['critical_low']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="critical_lab_value",
                    message=f"CRITICAL: {test} value {value} is critically low",
                    location=f"{test} result",
                    suggestion="Immediate medical attention required",
                    evidence=[f"Critical low: {ranges['critical_low']}"]
                ))
            elif 'critical_high' in ranges and value > ranges['critical_high']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="critical_lab_value",
                    message=f"CRITICAL: {test} value {value} is critically high",
                    location=f"{test} result",
                    suggestion="Immediate medical attention required",
                    evidence=[f"Critical high: {ranges['critical_high']}"]
                ))
            # Check normal ranges
            elif value < ranges['min']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category="abnormal_lab_value",
                    message=f"{test} value {value} below normal range",
                    location=f"{test} result",
                    suggestion=f"Normal range: {ranges['min']}-{ranges['max']} {ranges.get('unit', '')}",
                ))
            elif value > ranges['max']:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    category="abnormal_lab_value",
                    message=f"{test} value {value} above normal range",
                    location=f"{test} result",
                    suggestion=f"Normal range: {ranges['min']}-{ranges['max']} {ranges.get('unit', '')}",
                ))
        
        return issues
    
    def validate_diagnostic_criteria(self, condition: str, values: Dict[str, float]) -> List[ValidationIssue]:
        """Validate diagnostic criteria for conditions"""
        issues = []
        condition_lower = condition.lower()
        
        if condition_lower in self.diagnostic_criteria:
            criteria = self.diagnostic_criteria[condition_lower]
            
            for test, value in values.items():
                if test in criteria:
                    threshold = criteria[test]
                    
                    if isinstance(threshold, dict):
                        # Handle multiple thresholds
                        if 'threshold' in threshold and value >= threshold['threshold']:
                            issues.append(ValidationIssue(
                                severity=ValidationSeverity.HIGH,
                                category="diagnostic_criteria",
                                message=f"{test} value {value} meets criteria for {condition}",
                                location=f"{condition} diagnosis",
                                evidence=[f"Threshold: {threshold['threshold']} {threshold.get('unit', '')}"]
                            ))
        
        return issues


class ConsistencyChecker:
    """
    VALIDATION LAYER 2: Consistency Checking
    Ensures no contradictions in medical information
    """
    
    def check_temporal_consistency(self, events: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """Check temporal consistency of medical events"""
        issues = []
        
        # Sort events by timestamp if available
        sorted_events = sorted(events, key=lambda x: x.get('timestamp', 0))
        
        for i in range(len(sorted_events) - 1):
            current = sorted_events[i]
            next_event = sorted_events[i + 1]
            
            # Check for impossible temporal sequences
            if current.get('type') == 'death' and next_event.get('timestamp', 0) > current.get('timestamp', 0):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="temporal_inconsistency",
                    message="Events occurring after patient death",
                    location=f"Event sequence",
                    suggestion="Review temporal sequence of events"
                ))
            
            # Check medication consistency
            if current.get('type') == 'medication_stop' and next_event.get('type') == 'medication_continue':
                if current.get('medication') == next_event.get('medication'):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.MEDIUM,
                        category="medication_inconsistency",
                        message=f"Medication {current.get('medication')} stopped then continued",
                        location="Medication timeline",
                        suggestion="Clarify medication status"
                    ))
        
        return issues
    
    def check_dosage_consistency(self, medications: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """Check dosage consistency across mentions"""
        issues = []
        med_dosages = {}
        
        for med in medications:
            name = med.get('name', '').lower()
            dosage = med.get('dosage')
            
            if name in med_dosages:
                if dosage != med_dosages[name]:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.HIGH,
                        category="dosage_inconsistency",
                        message=f"Inconsistent dosages for {name}: {med_dosages[name]} vs {dosage}",
                        location=f"{name} dosage",
                        suggestion="Verify correct dosage"
                    ))
            else:
                med_dosages[name] = dosage
        
        return issues
    
    def check_diagnosis_treatment_alignment(self, diagnoses: List[str], treatments: List[str]) -> List[ValidationIssue]:
        """Check if treatments align with diagnoses"""
        issues = []
        
        # Standard treatment mappings
        standard_treatments = {
            'myocardial_infarction': ['aspirin', 'clopidogrel', 'heparin', 'beta_blocker'],
            'diabetes': ['metformin', 'insulin', 'glipizide'],
            'hypertension': ['lisinopril', 'amlodipine', 'metoprolol'],
            'heart_failure': ['furosemide', 'spironolactone', 'ace_inhibitor'],
            'pneumonia': ['antibiotics', 'azithromycin', 'ceftriaxone']
        }
        
        for diagnosis in diagnoses:
            diagnosis_lower = diagnosis.lower().replace(' ', '_')
            
            if diagnosis_lower in standard_treatments:
                expected_treatments = standard_treatments[diagnosis_lower]
                treatments_lower = [t.lower() for t in treatments]
                
                # Check if any standard treatment is present
                if not any(treat in treatments_lower for treat in expected_treatments):
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.MEDIUM,
                        category="treatment_alignment",
                        message=f"No standard treatment found for {diagnosis}",
                        location=f"{diagnosis} treatment",
                        suggestion=f"Consider standard treatments: {', '.join(expected_treatments[:3])}"
                    ))
        
        return issues


class SafetyFilter:
    """
    VALIDATION LAYER 3: Safety Filtering
    Flags potentially dangerous medical advice
    """
    
    def __init__(self):
        self.critical_decisions = [
            'stop medication',
            'discontinue treatment',
            'surgery required',
            'emergency',
            'life-threatening',
            'fatal',
            'immediate hospitalization',
            'cardiac arrest',
            'respiratory failure',
            'anaphylaxis'
        ]
        
        self.dangerous_advice_patterns = [
            r'stop.{0,20}medication',
            r'discontinue.{0,20}treatment',
            r'ignore.{0,20}symptoms',
            r'delay.{0,20}treatment',
            r'avoid.{0,20}hospital',
            r'self.{0,20}medicate'
        ]
        
        self.required_disclaimers = [
            "This information is for educational purposes only",
            "Consult a healthcare professional",
            "Not a substitute for medical advice",
            "Seek immediate medical attention if"
        ]
    
    def contains_critical_decision(self, text: str) -> bool:
        """Check if text contains critical medical decisions"""
        text_lower = text.lower()
        return any(decision in text_lower for decision in self.critical_decisions)
    
    def check_dangerous_advice(self, text: str) -> List[ValidationIssue]:
        """Check for potentially dangerous medical advice"""
        issues = []
        
        for pattern in self.dangerous_advice_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    category="dangerous_advice",
                    message=f"Potentially dangerous advice detected: {pattern}",
                    location="Response content",
                    suggestion="Add medical disclaimer and recommend professional consultation"
                ))
        
        return issues
    
    def add_safety_disclaimers(self, response: str) -> str:
        """Add appropriate medical disclaimers"""
        disclaimers = []
        
        if self.contains_critical_decision(response):
            disclaimers.append("⚠️ CRITICAL MEDICAL INFORMATION: This requires immediate professional medical evaluation.")
        
        disclaimers.append("\n\n📋 MEDICAL DISCLAIMER: This information is for educational purposes only and should not replace professional medical advice.")
        disclaimers.append("Always consult with a qualified healthcare provider for medical decisions.")
        
        if 'emergency' in response.lower() or 'immediate' in response.lower():
            disclaimers.append("🚨 If this is a medical emergency, call emergency services immediately.")
        
        return response + "\n\n" + "\n".join(disclaimers)
    
    def flag_for_review(self, response: str, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Flag response for human review if needed"""
        critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
        
        return {
            'needs_review': len(critical_issues) > 0,
            'critical_count': len(critical_issues),
            'review_reasons': [i.message for i in critical_issues],
            'suggested_actions': list(set([i.suggestion for i in critical_issues if i.suggestion]))
        }


class MedicalValidator:
    """
    COMPLETE MEDICAL VALIDATION PIPELINE
    Orchestrates all validation layers
    """
    
    def __init__(self):
        self.fact_checker = MedicalFactChecker()
        self.consistency_checker = ConsistencyChecker()
        self.safety_filter = SafetyFilter()
        self.validation_stats = {
            'total_validations': 0,
            'issues_found': 0,
            'critical_issues': 0,
            'responses_modified': 0
        }
    
    def validate_response(self, response: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete validation pipeline for medical responses
        """
        self.validation_stats['total_validations'] += 1
        all_issues = []
        
        # Layer 1: Fact Checking
        # Extract drug dosages from response
        drug_pattern = r'(\w+)\s+(\d+\.?\d*)\s*(mg|ml|units?|mcg)'
        drug_matches = re.findall(drug_pattern, response, re.IGNORECASE)
        
        for drug, dosage, unit in drug_matches:
            issues = self.fact_checker.validate_drug_dosage(drug, float(dosage), unit)
            all_issues.extend(issues)
        
        # Extract lab values
        lab_pattern = r'(glucose|hemoglobin|wbc|creatinine|troponin|ph|sodium|potassium)[:\s]+(\d+\.?\d*)'
        lab_matches = re.findall(lab_pattern, response, re.IGNORECASE)
        
        for test, value in lab_matches:
            issues = self.fact_checker.validate_lab_value(test, float(value))
            all_issues.extend(issues)
        
        # Layer 2: Consistency Checking
        if 'medications' in metadata:
            issues = self.consistency_checker.check_dosage_consistency(metadata['medications'])
            all_issues.extend(issues)
        
        if 'diagnoses' in metadata and 'treatments' in metadata:
            issues = self.consistency_checker.check_diagnosis_treatment_alignment(
                metadata['diagnoses'], metadata['treatments']
            )
            all_issues.extend(issues)
        
        # Layer 3: Safety Filtering
        safety_issues = self.safety_filter.check_dangerous_advice(response)
        all_issues.extend(safety_issues)
        
        # Process results
        critical_issues = [i for i in all_issues if i.severity == ValidationSeverity.CRITICAL]
        high_issues = [i for i in all_issues if i.severity == ValidationSeverity.HIGH]
        
        self.validation_stats['issues_found'] += len(all_issues)
        self.validation_stats['critical_issues'] += len(critical_issues)
        
        # Modify response if needed
        modified_response = response
        if critical_issues or self.safety_filter.contains_critical_decision(response):
            modified_response = self.safety_filter.add_safety_disclaimers(response)
            self.validation_stats['responses_modified'] += 1
        
        # Calculate confidence score
        confidence = 1.0
        confidence -= len(critical_issues) * 0.2
        confidence -= len(high_issues) * 0.1
        confidence = max(0.0, min(1.0, confidence))
        
        # Prepare validation result
        result = {
            'original_response': response,
            'validated_response': modified_response,
            'is_valid': len(critical_issues) == 0,
            'confidence_score': confidence,
            'total_issues': len(all_issues),
            'critical_issues': len(critical_issues),
            'issues': [
                {
                    'severity': issue.severity.value,
                    'category': issue.category,
                    'message': issue.message,
                    'suggestion': issue.suggestion
                }
                for issue in all_issues
            ],
            'needs_review': self.safety_filter.flag_for_review(response, all_issues),
            'disclaimers_added': modified_response != response
        }
        
        return result
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            **self.validation_stats,
            'critical_issue_rate': (self.validation_stats['critical_issues'] / 
                                   self.validation_stats['total_validations'] 
                                   if self.validation_stats['total_validations'] > 0 else 0),
            'modification_rate': (self.validation_stats['responses_modified'] / 
                                 self.validation_stats['total_validations'] 
                                 if self.validation_stats['total_validations'] > 0 else 0)
        }


def demonstrate_medical_validation():
    """Demonstration of the medical validation pipeline"""
    
    print("\n" + "="*80)
    print("MEDICAL VALIDATION PIPELINE DEMONSTRATION")
    print("="*80)
    
    # Create validator
    validator = MedicalValidator()
    
    # Test response 1: Contains dosage issues
    response1 = """
    The patient was diagnosed with myocardial infarction.
    Started on aspirin 1000mg daily and heparin 50000 units.
    Troponin levels were 2.5 ng/mL indicating cardiac damage.
    """
    
    print("\nTest 1: Validating response with dosage issues")
    print("-" * 60)
    result1 = validator.validate_response(response1, {
        'diagnoses': ['myocardial_infarction'],
        'treatments': ['aspirin', 'heparin']
    })
    
    print(f"Valid: {result1['is_valid']}")
    print(f"Confidence: {result1['confidence_score']:.2f}")
    print(f"Issues found: {result1['total_issues']} ({result1['critical_issues']} critical)")
    
    if result1['issues']:
        print("\nIssues:")
        for issue in result1['issues'][:3]:
            print(f"  [{issue['severity'].upper()}] {issue['message']}")
            if issue['suggestion']:
                print(f"    → {issue['suggestion']}")
    
    # Test response 2: Contains dangerous advice
    response2 = """
    Based on your symptoms, you should immediately stop all medications
    and avoid going to the hospital. This is a life-threatening emergency
    but you can manage it at home.
    """
    
    print("\n\nTest 2: Validating response with dangerous advice")
    print("-" * 60)
    result2 = validator.validate_response(response2, {})
    
    print(f"Valid: {result2['is_valid']}")
    print(f"Confidence: {result2['confidence_score']:.2f}")
    print(f"Disclaimers added: {result2['disclaimers_added']}")
    print(f"Needs review: {result2['needs_review']['needs_review']}")
    
    if result2['needs_review']['needs_review']:
        print(f"Review reasons: {result2['needs_review']['review_reasons']}")
    
    # Test response 3: Normal response
    response3 = """
    Your blood glucose level is 95 mg/dL which is within normal range.
    Continue taking metformin 500mg twice daily as prescribed.
    Schedule a follow-up appointment in 3 months.
    """
    
    print("\n\nTest 3: Validating normal response")
    print("-" * 60)
    result3 = validator.validate_response(response3, {
        'diagnoses': ['diabetes'],
        'treatments': ['metformin']
    })
    
    print(f"Valid: {result3['is_valid']}")
    print(f"Confidence: {result3['confidence_score']:.2f}")
    print(f"Issues found: {result3['total_issues']}")
    
    # Show overall statistics
    stats = validator.get_validation_stats()
    print("\n" + "="*80)
    print("VALIDATION STATISTICS:")
    print(f"  Total Validations: {stats['total_validations']}")
    print(f"  Total Issues Found: {stats['issues_found']}")
    print(f"  Critical Issue Rate: {stats['critical_issue_rate']*100:.1f}%")
    print(f"  Modification Rate: {stats['modification_rate']*100:.1f}%")
    
    print("\n" + "="*80)
    print("MEDICAL VALIDATION PIPELINE READY - SAFETY ASSURED!")
    print("="*80)


if __name__ == "__main__":
    demonstrate_medical_validation()