#!/usr/bin/env python3

import numpy as np
from typing import List, Dict, Tuple, Any, Set
import json
import os
import time
from dataclasses import dataclass
import requests
import hashlib
from collections import defaultdict
import math
from scipy.sparse import csr_matrix
import heapq

@dataclass
class SPLADEVector:
    term_weights: Dict[str, float]
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    
    @property
    def sparse_vector(self) -> Dict[str, float]:
        return self.term_weights
    
    def get_top_terms(self, k: int = 10) -> List[Tuple[str, float]]:
        return sorted(self.term_weights.items(), key=lambda x: x[1], reverse=True)[:k]

class SPLADESystem:
    def __init__(self):
        pass
        
    def encode_splade(self, text: str, doc_id: str) -> SPLADEVector:
        return SPLADEVector({}, doc_id, text, {})
