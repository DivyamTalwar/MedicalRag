import sys
import os
import json
import logging
import random
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rag_chatbot.app.services.agent.builder import build_medical_rag_agent
from langchain_core.messages import HumanMessage

def generate_questions_from_metadata(file_path: str, num_questions: int = 100) -> list[str]:
    """Generates a diverse set of questions based on the provided metadata file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        parent_chunks = data.get("parent_chunks", [])
    except Exception as e:
        logging.error(f"Failed to load or parse metadata from {file_path}: {e}")
        return []

    questions = [
        # General Questions
        "What is the purpose of the CIVIE CRM Portal?",
        "Describe the 5-step order journey through CIVIE.",
        "What are the key features of CIVIE's Radiology Information System?",
        "How does CIVIE's RIS Insight AI System help prevent patient leakage?",
        "What solutions does CIVIE offer for patient self-scheduling?",
        
        # Numerical Questions
        "What percentage of patients scheduled their appointments via the patient portal?",
        "By what percentage does contrast scheduling improve with CIVIE's system?",
        "What is the reported cost reduction after implementing CIVIE's solutions?",
        "How many orders were completed in the last 30 days according to CIVIE INSIGHTS?",
        "What is the payment collection rate shown in the CIVIE INSIGHTS report?",
        "What is the average time taken for a patient to self-schedule?",
        
        # Specific Detail Questions
        "Who is the Chief Financial Officer of CIVIE?",
        "What are the three main industry challenges CIVIE aims to solve?",
        "Name two ways incoming requisitions are typically received.",
        "What is the name of the CT Technologist quoted in the document?",
        "What is the title of the section that mentions 'Real-time TOS posting and reconciliation'?",
        "What was the last product added to the CIVIE timeline in 2024?",
        
        # Yes/No or Existence Questions
        "Does CIVIE's platform include a walk-in kiosk for scheduling?",
        "Is there a feature for managing special hours for exam resources?",
        "Does the CRM portal allow for the segmentation of physicians?",
        "Does CIVIE provide a solution for MQSA (Mammography Quality Standards Act) compliance?"
    ]
    
    # Add more questions based on chunk content to reach the desired number
    for chunk in parent_chunks:
        text = chunk.get("metadata", {}).get("text", "")
        if text and len(text) > 50: # Ensure there's enough content
            # Create a question from the first sentence
            first_sentence = text.split('.')[0]
            questions.append(f"What does the document say about '{first_sentence}'?")

    # Ensure we have enough unique questions
    unique_questions = list(set(questions))
    if len(unique_questions) < num_questions:
        for i in range(num_questions - len(unique_questions)):
            unique_questions.append(f"Provide a summary of CIVIE's features (query {i}).")
            
    return random.sample(unique_questions, num_questions)

def validate_run(question: str, result: dict) -> bool:
    """Validates the agent's output for a single run."""
    final_answer = result.get("generation_state", {}).get("final_answer", "")
    rich_citations = result.get("generation_state", {}).get("rich_citations", [])

    if not final_answer or "ERROR" in final_answer or "could not generate" in final_answer.lower():
        logging.warning(f"Validation FAILED for question '{question}': Invalid answer generated.")
        return False
        
    if not rich_citations:
        logging.warning(f"Validation FAILED for question '{question}': No citations were generated.")
        return False

    # Simple keyword check for relevance
    question_keywords = set(question.lower().split()) - {'what', 'is', 'are', 'the', 'a', 'an', 'of', 'for', 'does'}
    answer_lower = final_answer.lower()
    if not any(keyword in answer_lower for keyword in question_keywords):
        logging.warning(f"Validation FAILED for question '{question}': Answer does not seem relevant.")
        return False
            
    logging.info(f"Validation PASSED for question '{question}'.")
    return True

def run_full_agent_test_loop():
    """
    Runs a 100-iteration test of the full RAG agent.
    """
    logging.info("--- Starting Full Agent Test Loop ---")
    
    try:
        agent = build_medical_rag_agent()
        logging.info("Agent built successfully.")
    except Exception as e:
        logging.error(f"Failed to build the agent: {e}")
        return

    questions = generate_questions_from_metadata("civie_ris_metadata.json")
    if not questions:
        return

    perfect_runs = 0
    
    # Preliminary test with 3 diverse questions
    preliminary_questions = random.sample(questions, 3)
    logging.info("--- Starting Preliminary Test Phase (3 questions) ---")
    
    preliminary_success = True
    for i, question in enumerate(preliminary_questions, 1):
        logging.info(f"--- Running Preliminary Iteration {i}/3 ---")
        logging.info(f"Question: {question}")
        
        initial_state = {
            "query_state": {
                "original_query": question,
                "condensed_query": "",
                "chat_history": [],
                "medical_entities": {}
            },
            "search_state": {
                "dense_results": [],
                "sparse_results": [],
                "merged_candidates": [],
                "reranked_chunks": []
            },
            "context_state": {
                "parent_chunks": [],
                "assembled_context": "",
                "context_sufficiency": False,
                "medical_metadata": {}
            },
            "generation_state": {
                "final_answer": "",
                "rich_citations": [],
                "is_streaming": False,
            },
            "performance_state": {
                "node_timings": {},
                "total_duration": 0.0
            },
            "error_state": {
                "error_message": None,
                "failed_node": None
            }
        }
        
        try:
            result = agent.run(initial_state)
            final_answer = result.get("generation_state", {}).get("final_answer", "ERROR: No answer.")
            rich_citations = result.get("generation_state", {}).get("rich_citations", [])

            print(f"\n[PRELIMINARY ITERATION {i}] FINAL ANSWER:\n{final_answer}")
            
            if rich_citations:
                print(f"\n[PRELIMINARY ITERATION {i}] CITATIONS:")
                for c in rich_citations:
                    print(f"  - Source: {c.get('source_name', 'N/A')}, Page: {c.get('page_number', 'N/A')}")
            
            if not validate_run(question, result):
                preliminary_success = False
                break

            print("\n" + "="*50 + "\n")

        except Exception as e:
            logging.error(f"Error during preliminary execution in iteration {i}: {e}")
            preliminary_success = False
            break
            
    if not preliminary_success:
        logging.error("Preliminary test failed. Aborting full test.")
        return
        
    logging.info("--- Preliminary Test Passed. Starting Full Test Loop (100 questions) ---")
    
    for i, question in enumerate(questions, 1):
        logging.info(f"--- Running Full Iteration {i}/{len(questions)} ---")
        logging.info(f"Question: {question}")
        
        initial_state = {
            "query_state": {
                "original_query": question,
                "condensed_query": "",
                "chat_history": [],
                "medical_entities": {}
            },
            "search_state": {
                "dense_results": [],
                "sparse_results": [],
                "merged_candidates": [],
                "reranked_chunks": []
            },
            "context_state": {
                "parent_chunks": [],
                "assembled_context": "",
                "context_sufficiency": False,
                "medical_metadata": {}
            },
            "generation_state": {
                "final_answer": "",
                "rich_citations": [],
                "is_streaming": False,
            },
            "performance_state": {
                "node_timings": {},
                "total_duration": 0.0
            },
            "error_state": {
                "error_message": None,
                "failed_node": None
            }
        }
        
        try:
            result = agent.run(initial_state)
            final_answer = result.get("generation_state", {}).get("final_answer", "ERROR: No answer.")
            rich_citations = result.get("generation_state", {}).get("rich_citations", [])

            print(f"\n[ITERATION {i}] FINAL ANSWER:\n{final_answer}")
            
            if rich_citations:
                print(f"\n[ITERATION {i}] CITATIONS:")
                for c in rich_citations:
                    print(f"  - Source: {c.get('source_name', 'N/A')}, Page: {c.get('page_number', 'N/A')}")
            
            if validate_run(question, result):
                perfect_runs += 1

            print("\n" + "="*50 + "\n")

        except Exception as e:
            logging.error(f"Error during agent execution in iteration {i}: {e}")

    logging.info(f"--- Test Complete ---")
    logging.info(f"Final Score: {perfect_runs}/{len(questions)} perfect runs.")
    
    if perfect_runs < 99:
        logging.error("The agent did not meet the 99% accuracy threshold. Further refinement is needed.")
    else:
        logging.info("The agent has met the 99% accuracy threshold. The system is ready.")

if __name__ == "__main__":
    run_full_agent_test_loop()
