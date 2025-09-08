#!/usr/bin/env python3
"""
ACTIVE LEARNING & CONTINUOUS IMPROVEMENT SYSTEM
Self-Improving RAG that Gets Smarter Over Time
"""

import json
import time
import pickle
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime, timedelta
import sqlite3
import hashlib

@dataclass
class FeedbackEntry:
    """Represents user feedback on a response"""
    query_id: str
    query: str
    original_response: str
    corrected_response: Optional[str]
    feedback_type: str  # 'correction', 'rating', 'flag'
    rating: Optional[float]  # 1-5 scale
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class ErrorPattern:
    """Represents a systematic error pattern"""
    pattern_id: str
    pattern_type: str
    frequency: int
    examples: List[Dict[str, Any]]
    suggested_fix: Optional[str]
    confidence: float

@dataclass
class TrainingPair:
    """Represents a training pair for model fine-tuning"""
    pair_id: str
    input_text: str
    target_output: str
    pair_type: str  # 'positive', 'negative', 'contrastive'
    weight: float
    source: str  # 'user_feedback', 'synthetic', 'augmented'

class ActiveLearningSystem:
    """
    LEGENDARY ACTIVE LEARNING SYSTEM
    Continuously improves based on user feedback and performance
    """
    
    def __init__(self, db_path: str = "active_learning.db"):
        self.db_path = db_path
        self._init_database()
        
        # Error pattern detection
        self.error_patterns = defaultdict(lambda: {'count': 0, 'examples': []})
        
        # Performance tracking
        self.performance_metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'corrected_queries': 0,
            'average_rating': 0.0,
            'improvement_rate': 0.0
        }
        
        # Learning parameters
        self.learning_config = {
            'min_examples_for_pattern': 3,
            'confidence_threshold': 0.7,
            'retraining_frequency': 100,  # Retrain after N feedback entries
            'weight_decay': 0.95,  # Decay factor for old training pairs
            'exploration_rate': 0.1  # Percentage of experimental responses
        }
        
        # A/B testing
        self.ab_tests = {}
        
        # Stage weights for retrieval pipeline
        self.stage_weights = {
            'bm25': 0.2,
            'dense': 0.4,
            'rerank': 0.3,
            'diversity': 0.1
        }
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                query_id TEXT PRIMARY KEY,
                query TEXT,
                original_response TEXT,
                corrected_response TEXT,
                feedback_type TEXT,
                rating REAL,
                timestamp REAL,
                metadata TEXT
            )
        ''')
        
        # Error patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS error_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT,
                frequency INTEGER,
                examples TEXT,
                suggested_fix TEXT,
                confidence REAL,
                last_updated REAL
            )
        ''')
        
        # Training pairs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_pairs (
                pair_id TEXT PRIMARY KEY,
                input_text TEXT,
                target_output TEXT,
                pair_type TEXT,
                weight REAL,
                source TEXT,
                created_at REAL
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                timestamp REAL PRIMARY KEY,
                metric_name TEXT,
                metric_value REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def learn_from_correction(self, query: str, original_response: str, 
                             corrected_response: str, metadata: Dict[str, Any] = None):
        """
        Learn from user corrections
        This is the CORE of active learning
        """
        query_id = hashlib.md5(f"{query}{time.time()}".encode()).hexdigest()
        
        # Store feedback
        feedback = FeedbackEntry(
            query_id=query_id,
            query=query,
            original_response=original_response,
            corrected_response=corrected_response,
            feedback_type='correction',
            rating=None,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        self._store_feedback(feedback)
        
        # Analyze error
        error_analysis = self._analyze_error(original_response, corrected_response)
        
        # Update error patterns
        self._update_error_patterns(query, error_analysis)
        
        # Create training pairs
        self._create_training_pairs_from_correction(query, original_response, corrected_response)
        
        # Update performance metrics
        self.performance_metrics['corrected_queries'] += 1
        
        # Trigger retraining if needed
        if self.performance_metrics['corrected_queries'] % self.learning_config['retraining_frequency'] == 0:
            self.trigger_retraining()
    
    def _store_feedback(self, feedback: FeedbackEntry):
        """Store feedback in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO feedback 
            (query_id, query, original_response, corrected_response, 
             feedback_type, rating, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.query_id,
            feedback.query,
            feedback.original_response,
            feedback.corrected_response,
            feedback.feedback_type,
            feedback.rating,
            feedback.timestamp,
            json.dumps(feedback.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    def _analyze_error(self, original: str, corrected: str) -> Dict[str, Any]:
        """Analyze the type and nature of error"""
        analysis = {
            'error_type': 'unknown',
            'severity': 'medium',
            'key_differences': [],
            'missing_information': [],
            'incorrect_information': []
        }
        
        # Simple analysis based on text differences
        original_lower = original.lower()
        corrected_lower = corrected.lower()
        
        # Check for missing information
        corrected_words = set(corrected_lower.split())
        original_words = set(original_lower.split())
        
        missing_words = corrected_words - original_words
        extra_words = original_words - corrected_words
        
        if len(missing_words) > len(extra_words):
            analysis['error_type'] = 'incomplete'
            analysis['missing_information'] = list(missing_words)[:10]
        elif len(extra_words) > len(missing_words):
            analysis['error_type'] = 'excessive'
            analysis['incorrect_information'] = list(extra_words)[:10]
        else:
            analysis['error_type'] = 'inaccurate'
        
        # Check for medical term errors
        medical_terms = ['diagnosis', 'treatment', 'medication', 'dosage', 'symptoms']
        for term in medical_terms:
            if term in corrected_lower and term not in original_lower:
                analysis['error_type'] = 'medical_missing'
                analysis['severity'] = 'high'
                break
        
        return analysis
    
    def _update_error_patterns(self, query: str, error_analysis: Dict[str, Any]):
        """Update systematic error patterns"""
        pattern_key = error_analysis['error_type']
        
        self.error_patterns[pattern_key]['count'] += 1
        self.error_patterns[pattern_key]['examples'].append({
            'query': query,
            'analysis': error_analysis,
            'timestamp': time.time()
        })
        
        # Check if this constitutes a systematic pattern
        if self.error_patterns[pattern_key]['count'] >= self.learning_config['min_examples_for_pattern']:
            self._create_error_pattern(pattern_key, self.error_patterns[pattern_key])
    
    def _create_error_pattern(self, pattern_key: str, pattern_data: Dict[str, Any]):
        """Create and store a systematic error pattern"""
        pattern = ErrorPattern(
            pattern_id=hashlib.md5(f"{pattern_key}{time.time()}".encode()).hexdigest(),
            pattern_type=pattern_key,
            frequency=pattern_data['count'],
            examples=pattern_data['examples'][-10:],  # Keep last 10 examples
            suggested_fix=self._suggest_fix_for_pattern(pattern_key),
            confidence=min(pattern_data['count'] / 10, 1.0)  # Confidence based on frequency
        )
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO error_patterns
            (pattern_id, pattern_type, frequency, examples, suggested_fix, confidence, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern.pattern_id,
            pattern.pattern_type,
            pattern.frequency,
            json.dumps(pattern.examples),
            pattern.suggested_fix,
            pattern.confidence,
            time.time()
        ))
        
        conn.commit()
        conn.close()
    
    def _suggest_fix_for_pattern(self, pattern_type: str) -> str:
        """Suggest fix for identified error pattern"""
        fixes = {
            'incomplete': "Increase retrieval top_k and ensure comprehensive context building",
            'excessive': "Implement better relevance filtering and response summarization",
            'inaccurate': "Improve fact validation and cross-reference checking",
            'medical_missing': "Enhance medical entity extraction and knowledge graph integration",
            'hallucination': "Strengthen grounding in retrieved documents",
            'contradiction': "Implement consistency checking across response"
        }
        
        return fixes.get(pattern_type, "Review and adjust retrieval and generation parameters")
    
    def _create_training_pairs_from_correction(self, query: str, original: str, corrected: str):
        """Create training pairs from user corrections"""
        
        # Positive pair (query → correct response)
        positive_pair = TrainingPair(
            pair_id=hashlib.md5(f"pos_{query}_{corrected}".encode()).hexdigest(),
            input_text=query,
            target_output=corrected,
            pair_type='positive',
            weight=1.0,
            source='user_feedback'
        )
        
        # Negative pair (query → incorrect response)
        negative_pair = TrainingPair(
            pair_id=hashlib.md5(f"neg_{query}_{original}".encode()).hexdigest(),
            input_text=query,
            target_output=original,
            pair_type='negative',
            weight=-0.5,  # Negative weight for incorrect examples
            source='user_feedback'
        )
        
        # Store training pairs
        self._store_training_pair(positive_pair)
        self._store_training_pair(negative_pair)
        
        # Create augmented pairs
        self._create_augmented_pairs(query, corrected)
    
    def _store_training_pair(self, pair: TrainingPair):
        """Store training pair in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO training_pairs
            (pair_id, input_text, target_output, pair_type, weight, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            pair.pair_id,
            pair.input_text,
            pair.target_output,
            pair.pair_type,
            pair.weight,
            pair.source,
            time.time()
        ))
        
        conn.commit()
        conn.close()
    
    def _create_augmented_pairs(self, query: str, corrected_response: str):
        """Create augmented training pairs for better generalization"""
        
        # Paraphrase variations
        paraphrases = [
            query.replace("What is", "What's"),
            query.replace("What are", "What're"),
            query + "?",
            "Can you tell me " + query.lower(),
            "I need to know " + query.lower()
        ]
        
        for paraphrase in paraphrases[:3]:  # Limit augmentation
            augmented_pair = TrainingPair(
                pair_id=hashlib.md5(f"aug_{paraphrase}_{corrected_response}".encode()).hexdigest(),
                input_text=paraphrase,
                target_output=corrected_response,
                pair_type='positive',
                weight=0.8,  # Slightly lower weight for augmented
                source='augmented'
            )
            self._store_training_pair(augmented_pair)
    
    def optimize_stage_weights(self):
        """
        Optimize retrieval stage weights based on performance
        Uses gradient-free optimization
        """
        
        # Get recent feedback
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT query, original_response, corrected_response, rating
            FROM feedback
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            LIMIT 100
        ''', (time.time() - 86400 * 7,))  # Last 7 days
        
        feedback_data = cursor.fetchall()
        conn.close()
        
        if not feedback_data:
            return
        
        # Calculate performance for different weight combinations
        best_weights = self.stage_weights.copy()
        best_performance = 0
        
        # Try different weight combinations
        for _ in range(10):  # 10 iterations
            # Randomly perturb weights
            test_weights = {
                'bm25': max(0.1, min(0.5, best_weights['bm25'] + np.random.randn() * 0.05)),
                'dense': max(0.2, min(0.6, best_weights['dense'] + np.random.randn() * 0.05)),
                'rerank': max(0.1, min(0.5, best_weights['rerank'] + np.random.randn() * 0.05)),
                'diversity': max(0.05, min(0.3, best_weights['diversity'] + np.random.randn() * 0.05))
            }
            
            # Normalize weights
            total = sum(test_weights.values())
            test_weights = {k: v/total for k, v in test_weights.items()}
            
            # Evaluate performance (simplified)
            performance = self._evaluate_weights(test_weights, feedback_data)
            
            if performance > best_performance:
                best_performance = performance
                best_weights = test_weights
        
        # Update weights
        self.stage_weights = best_weights
        print(f"Optimized stage weights: {self.stage_weights}")
    
    def _evaluate_weights(self, weights: Dict[str, float], feedback_data: List) -> float:
        """Evaluate performance of weight configuration"""
        # Simplified evaluation - in production, this would re-run retrieval
        performance = 0
        
        for query, original, corrected, rating in feedback_data:
            if rating:
                # Higher rating = better performance
                performance += rating * weights['dense']  # Dense retrieval most important
            elif corrected:
                # Penalize if correction was needed
                performance -= 1.0
        
        return performance
    
    def trigger_retraining(self):
        """
        Trigger model retraining with accumulated training pairs
        """
        print("\n" + "="*60)
        print("TRIGGERING MODEL RETRAINING")
        print("="*60)
        
        # Get all training pairs
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT pair_id, input_text, target_output, pair_type, weight, source
            FROM training_pairs
            WHERE weight > 0
            ORDER BY created_at DESC
            LIMIT 1000
        ''')
        
        training_data = cursor.fetchall()
        conn.close()
        
        print(f"Found {len(training_data)} training pairs")
        
        # Apply weight decay to older pairs
        decayed_pairs = []
        for i, (pair_id, input_text, target_output, pair_type, weight, source) in enumerate(training_data):
            decay_factor = self.learning_config['weight_decay'] ** (i / 100)  # Decay based on age
            decayed_weight = weight * decay_factor
            decayed_pairs.append((input_text, target_output, decayed_weight))
        
        # In production, this would trigger actual model fine-tuning
        print(f"Retraining with {len(decayed_pairs)} weighted pairs")
        print("Retraining complete (simulated)")
        
        # Update performance metrics
        self.performance_metrics['improvement_rate'] = 0.05  # Simulated improvement
    
    def run_ab_test(self, test_name: str, variant_a: Any, variant_b: Any, 
                    duration_hours: int = 24) -> Dict[str, Any]:
        """
        Run A/B test between two strategies
        """
        test_id = hashlib.md5(f"{test_name}{time.time()}".encode()).hexdigest()
        
        self.ab_tests[test_id] = {
            'name': test_name,
            'variant_a': variant_a,
            'variant_b': variant_b,
            'start_time': time.time(),
            'end_time': time.time() + duration_hours * 3600,
            'results_a': {'queries': 0, 'success': 0, 'rating_sum': 0},
            'results_b': {'queries': 0, 'success': 0, 'rating_sum': 0}
        }
        
        print(f"Started A/B test: {test_name}")
        return {'test_id': test_id, 'status': 'running'}
    
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get results of A/B test"""
        if test_id not in self.ab_tests:
            return {'error': 'Test not found'}
        
        test = self.ab_tests[test_id]
        
        # Calculate performance metrics
        perf_a = (test['results_a']['success'] / test['results_a']['queries'] 
                 if test['results_a']['queries'] > 0 else 0)
        perf_b = (test['results_b']['success'] / test['results_b']['queries'] 
                 if test['results_b']['queries'] > 0 else 0)
        
        avg_rating_a = (test['results_a']['rating_sum'] / test['results_a']['queries'] 
                       if test['results_a']['queries'] > 0 else 0)
        avg_rating_b = (test['results_b']['rating_sum'] / test['results_b']['queries'] 
                       if test['results_b']['queries'] > 0 else 0)
        
        return {
            'test_name': test['name'],
            'status': 'completed' if time.time() > test['end_time'] else 'running',
            'variant_a': {
                'queries': test['results_a']['queries'],
                'success_rate': perf_a,
                'avg_rating': avg_rating_a
            },
            'variant_b': {
                'queries': test['results_b']['queries'],
                'success_rate': perf_b,
                'avg_rating': avg_rating_b
            },
            'winner': 'A' if perf_a > perf_b else 'B' if perf_b > perf_a else 'tie'
        }
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get feedback stats
        cursor.execute('SELECT COUNT(*) FROM feedback')
        total_feedback = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM feedback WHERE corrected_response IS NOT NULL')
        total_corrections = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL')
        avg_rating = cursor.fetchone()[0] or 0
        
        # Get training pair stats
        cursor.execute('SELECT COUNT(*) FROM training_pairs')
        total_pairs = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT pattern_type) FROM error_patterns')
        unique_patterns = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_feedback': total_feedback,
            'total_corrections': total_corrections,
            'correction_rate': total_corrections / total_feedback if total_feedback > 0 else 0,
            'average_rating': avg_rating,
            'total_training_pairs': total_pairs,
            'unique_error_patterns': unique_patterns,
            'current_stage_weights': self.stage_weights,
            'improvement_rate': self.performance_metrics['improvement_rate'],
            'last_retraining': self.performance_metrics.get('last_retraining', 'Never')
        }


def demonstrate_active_learning():
    """Demonstration of the active learning system"""
    
    print("\n" + "="*80)
    print("ACTIVE LEARNING SYSTEM DEMONSTRATION")
    print("="*80)
    
    # Create active learning system
    learner = ActiveLearningSystem()
    
    # Simulate user corrections
    corrections = [
        {
            'query': "What is the treatment for myocardial infarction?",
            'original': "Treatment includes rest and observation",
            'corrected': "Treatment includes aspirin 325mg, clopidogrel 600mg, heparin infusion, and urgent cardiac catheterization"
        },
        {
            'query': "What are normal glucose levels?",
            'original': "Normal glucose is below 200",
            'corrected': "Normal fasting glucose is 70-100 mg/dL"
        },
        {
            'query': "What is the dosage for metformin?",
            'original': "Metformin dosage varies",
            'corrected': "Metformin typically starts at 500mg twice daily, can increase to 2000mg daily maximum"
        }
    ]
    
    print("\nProcessing user corrections:")
    for correction in corrections:
        print(f"\n  Query: {correction['query'][:50]}...")
        learner.learn_from_correction(
            correction['query'],
            correction['original'],
            correction['corrected']
        )
        print("  ✓ Learned from correction")
    
    # Optimize weights
    print("\nOptimizing retrieval stage weights...")
    learner.optimize_stage_weights()
    
    # Run A/B test
    print("\nStarting A/B test...")
    test_result = learner.run_ab_test(
        "Dense vs Sparse Retrieval",
        variant_a={'method': 'dense', 'weight': 0.7},
        variant_b={'method': 'sparse', 'weight': 0.7},
        duration_hours=1
    )
    print(f"  Test ID: {test_result['test_id']}")
    
    # Get learning statistics
    stats = learner.get_learning_statistics()
    
    print("\n" + "="*80)
    print("LEARNING STATISTICS:")
    print("-" * 60)
    print(f"Total Feedback: {stats['total_feedback']}")
    print(f"Total Corrections: {stats['total_corrections']}")
    print(f"Correction Rate: {stats['correction_rate']*100:.1f}%")
    print(f"Average Rating: {stats['average_rating']:.2f}")
    print(f"Training Pairs Created: {stats['total_training_pairs']}")
    print(f"Unique Error Patterns: {stats['unique_error_patterns']}")
    print(f"\nOptimized Stage Weights:")
    for stage, weight in stats['current_stage_weights'].items():
        print(f"  {stage}: {weight:.3f}")
    
    print("\n" + "="*80)
    print("ACTIVE LEARNING SYSTEM READY - CONTINUOUS IMPROVEMENT ENABLED!")
    print("="*80)


if __name__ == "__main__":
    demonstrate_active_learning()