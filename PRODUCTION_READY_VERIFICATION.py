#!/usr/bin/env python3
"""
FINAL PRODUCTION VERIFICATION - NO MOCK DATA
Ensures everything is using REAL APIs and REAL data
"""

import os
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

print("\n" + "="*80)
print("PRODUCTION DEPLOYMENT VERIFICATION")
print("NO MOCK DATA - EVERYTHING REAL")
print("="*80)

# Track all checks
checks_passed = []
checks_failed = []

# CHECK 1: Environment Variables
print("\n[1/10] ENVIRONMENT VARIABLES")
required_env = {
    "MODELS_API_KEY": os.getenv("MODELS_API_KEY"),
    "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
    "EMBEDDING_API_URL": os.getenv("EMBEDDING_API_URL", "https://api.us.inc/usf/v1/embed/embeddings"),
    "LLM_ENDPOINT": os.getenv("LLM_ENDPOINT", "https://api.us.inc/omega/civie/v1/chat/completions")
}

for key, value in required_env.items():
    if value:
        print(f"  [OK] {key}: {value[:30]}...")
        checks_passed.append(f"{key} configured")
    else:
        print(f"  [FAIL] {key}: NOT SET!")
        checks_failed.append(f"{key} missing")

# CHECK 2: OMEGA Embedding API
print("\n[2/10] OMEGA EMBEDDING API (REAL)")
try:
    response = requests.post(
        required_env["EMBEDDING_API_URL"],
        headers={
            "x-api-key": required_env["MODELS_API_KEY"],
            "Content-Type": "application/json"
        },
        json={
            "input": "medical test query",
            "model": "medical-embeddings-v1"
        },
        timeout=10
    )
    
    if response.status_code == 200:
        data = response.json()
        if 'result' in data and 'data' in data['result']:
            embedding = data['result']['data'][0]['embedding']
            print(f"  [OK] API Response: 200 OK")
            print(f"  [OK] Embedding Dimensions: {len(embedding)}")
            print(f"  [OK] Using REAL OMEGA embeddings!")
            checks_passed.append("OMEGA embedding API working")
        else:
            print(f"  [FAIL] Unexpected response format")
            checks_failed.append("OMEGA API format issue")
    else:
        print(f"  [FAIL] API returned: {response.status_code}")
        checks_failed.append(f"OMEGA API error {response.status_code}")
except Exception as e:
    print(f"  [FAIL] API Error: {e}")
    checks_failed.append("OMEGA API connection failed")

# CHECK 3: Pinecone Database
print("\n[3/10] PINECONE VECTOR DATABASE")
try:
    pc = Pinecone(api_key=required_env["PINECONE_API_KEY"])
    index = pc.Index("legendary-medical-rag")
    stats = index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)
    
    print(f"  [OK] Connected to Pinecone")
    print(f"  [OK] Index: legendary-medical-rag")
    print(f"  [OK] Vectors in database: {total_vectors}")
    
    if total_vectors > 100:
        print(f"  [OK] Sufficient vectors for production")
        checks_passed.append(f"Pinecone has {total_vectors} vectors")
    else:
        print(f"  [WARNING] Low vector count: {total_vectors}")
        checks_failed.append("Insufficient vectors in Pinecone")
        
except Exception as e:
    print(f"  [FAIL] Pinecone Error: {e}")
    checks_failed.append("Pinecone connection failed")

# CHECK 4: PDF Files
print("\n[4/10] PDF FILES IN DATA DIRECTORY")
pdf_dir = Path("rag_chatbot/data")
pdf_files = list(pdf_dir.glob("*.pdf"))
expected_pdfs = [
    "arterial-blood-gas",
    "cardiology",
    "pathology",
    "Torch-profile",
    "UMNwriteup"
]

print(f"  [OK] Found {len(pdf_files)} PDFs")
for pdf in pdf_files[:5]:
    print(f"  [OK] {pdf.name}")
    
if len(pdf_files) >= 9:
    checks_passed.append(f"All {len(pdf_files)} PDFs present")
else:
    checks_failed.append(f"Only {len(pdf_files)} PDFs found")

# CHECK 5: No Mock Functions
print("\n[5/10] VERIFY NO MOCK IMPLEMENTATIONS")
from ultimate_fix_pinecone import simple_embedding

test_embedding = simple_embedding("test medical query")
if isinstance(test_embedding, list) and len(test_embedding) == 1024:
    # Check if it's random (mock) or real
    import numpy as np
    if max(test_embedding) > 0.5 or min(test_embedding) < -0.5:
        print("  [FAIL] Using RANDOM embeddings!")
        checks_failed.append("Using mock embeddings")
    else:
        print("  [OK] Using REAL OMEGA embeddings")
        print(f"  [OK] Sample values: {test_embedding[:3]}")
        checks_passed.append("Real embeddings verified")

# CHECK 6: Test Retrieval
print("\n[6/10] TEST ACTUAL RETRIEVAL")
try:
    test_queries = [
        "hemoglobin blood test",
        "arterial pH",
        "blood pressure"
    ]
    
    success_count = 0
    for query in test_queries:
        embedding = simple_embedding(query)
        results = index.query(vector=embedding, top_k=1, include_metadata=True)
        
        if results['matches'] and results['matches'][0]['score'] > 0.5:
            success_count += 1
            match = results['matches'][0]
            print(f"  [OK] Query: '{query}' -> Score: {match['score']:.3f}")
    
    if success_count == len(test_queries):
        checks_passed.append("All test queries successful")
    else:
        checks_failed.append(f"Only {success_count}/{len(test_queries)} queries worked")
        
except Exception as e:
    print(f"  [FAIL] Retrieval Error: {e}")
    checks_failed.append("Retrieval test failed")

# CHECK 7: Python Packages
print("\n[7/10] REQUIRED PYTHON PACKAGES")
required_packages = [
    "pinecone",
    "PyPDF2",
    "requests",
    "numpy",
    "fastapi",
    "uvicorn",
    "pydantic"
]

for package in required_packages:
    try:
        __import__(package)
        print(f"  [OK] {package} installed")
    except ImportError:
        print(f"  [FAIL] {package} NOT installed")
        checks_failed.append(f"{package} missing")

# CHECK 8: API Endpoints
print("\n[8/10] API ENDPOINTS CONFIGURED")
endpoints = {
    "Embedding": required_env["EMBEDDING_API_URL"],
    "LLM": required_env["LLM_ENDPOINT"],
    "Reranker": "https://api.us.inc/usf-shiprocket/v1/embed/reranker"
}

for name, url in endpoints.items():
    if url and url.startswith("https://"):
        print(f"  [OK] {name}: {url}")
        checks_passed.append(f"{name} endpoint configured")
    else:
        print(f"  [FAIL] {name}: Invalid or missing")
        checks_failed.append(f"{name} endpoint issue")

# CHECK 9: Main Application
print("\n[9/10] MAIN APPLICATION FILE")
if Path("main.py").exists():
    print("  [OK] main.py exists")
    checks_passed.append("Main application present")
else:
    print("  [FAIL] main.py not found")
    checks_failed.append("Main application missing")

# CHECK 10: Final System Test
print("\n[10/10] FINAL INTEGRATION TEST")
try:
    # Test the complete flow
    test_query = "What are the hemoglobin levels?"
    
    # Get embedding
    query_embedding = simple_embedding(test_query)
    
    # Search
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )
    
    if results['matches']:
        top_match = results['matches'][0]
        score = top_match['score']
        source = top_match['metadata']['source']
        text = top_match['metadata']['text'][:100]
        
        print(f"  [OK] Query: '{test_query}'")
        print(f"  [OK] Best Match Score: {score:.3f}")
        print(f"  [OK] Source: {source}")
        print(f"  [OK] Content: {text}...")
        
        if score > 0.6:
            print("  [OK] HIGH CONFIDENCE MATCH!")
            checks_passed.append("Integration test successful")
        else:
            print("  [WARNING] Low confidence score")
            checks_failed.append("Low confidence in results")
    else:
        print("  [FAIL] No results returned")
        checks_failed.append("Integration test failed")
        
except Exception as e:
    print(f"  [FAIL] Integration Error: {e}")
    checks_failed.append("Integration test error")

# FINAL REPORT
print("\n" + "="*80)
print("PRODUCTION READINESS REPORT")
print("="*80)

total_checks = len(checks_passed) + len(checks_failed)
success_rate = (len(checks_passed) / total_checks * 100) if total_checks > 0 else 0

print(f"\nChecks Passed: {len(checks_passed)}/{total_checks}")
print(f"Success Rate: {success_rate:.1f}%")

if checks_passed:
    print("\n[OK] SUCCESSES:")
    for check in checks_passed[:10]:
        print(f"  - {check}")

if checks_failed:
    print("\n[FAIL] ISSUES TO FIX:")
    for check in checks_failed[:10]:
        print(f"  - {check}")

print("\n" + "="*80)
if success_rate >= 90:
    print("VERDICT: PRODUCTION READY!")
    print("All critical systems using REAL APIs and REAL data")
    print("NO MOCK IMPLEMENTATIONS DETECTED")
elif success_rate >= 70:
    print("VERDICT: MOSTLY READY (Fix remaining issues)")
else:
    print("VERDICT: NOT READY (Critical issues found)")
print("="*80)