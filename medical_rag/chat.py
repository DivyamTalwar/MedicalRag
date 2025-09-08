import requests
from typing import List, Dict, Any
from search import SearchService
from config import LLM_ENDPOINT, OMEGA_API_KEY, OMEGA_MODEL

class ChatService:
    def __init__(self):
        self.api_url = LLM_ENDPOINT
        self.api_key = OMEGA_API_KEY
        self.model = OMEGA_MODEL
        self.search_service = SearchService()
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        print(f"[OK] Chat service initialized: {self.model}")
    
    def generate_response(self, prompt: str, max_tokens: int = 2048) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a medical AI assistant. Provide accurate, helpful medical information based on the context provided. Always recommend consulting healthcare professionals for personalized advice."
                },
                {
                    "role": "user", 
                    "content": prompt.strip()
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.95
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"[ERROR] API returned {response.status_code}: {response.text}")
                raise Exception(f"LLM API failed: {response.status_code}")
            
            result = response.json()
            
            if "result" in result:
                result = result["result"]
            
            if "choices" not in result or not result["choices"]:
                print(f"[ERROR] No choices in response: {result}")
                raise Exception("No response generated")
            
            message = result["choices"][0]["message"]["content"]
            print(f"[OK] Generated response: {len(message)} characters")
            return message.strip()
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request failed: {e}")
            raise Exception(f"LLM request failed: {e}")
    
    def rag_query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        
        print(f"[RAG] Processing question: '{question}'")
        search_results = self.search_service.search(question, top_k=top_k)
        
        if not search_results:
            print("[RAG] No relevant documents found, generating general response")
            context = "No specific medical documents found."
        else:
            context_parts = []
            for i, result in enumerate(search_results, 1):
                context_parts.append(f"Document {i}: {result['text']}")
            
            context = "\n".join(context_parts)
            print(f"[RAG] Using context from {len(search_results)} documents")
        
        rag_prompt = f"""Based on the following medical information, please answer the question.

MEDICAL CONTEXT:
{context}

QUESTION: {question}

Please provide a clear, accurate answer based on the context provided. If the context doesn't contain sufficient information to answer the question, please indicate that and provide general medical guidance while recommending consultation with healthcare professionals."""
        
        response = self.generate_response(rag_prompt)
        
        return {
            "question": question,
            "answer": response,
            "sources": search_results,
            "context_length": len(context),
            "sources_used": len(search_results)
        }
    
    def simple_chat(self, message: str) -> str:
        return self.generate_response(message)