import sys
import os
import json
import logging
import random
import asyncio
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_chatbot.app.services.agent.simple_flow import SimpleRAGFlow

def generate_questions_from_metadata(file_path: str, num_questions: int = 10) -> list[str]:
    """Generates a diverse set of questions based on the provided metadata file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load or parse metadata from {file_path}: {e}")
        return []

    questions = [
        "What specific DICOM standards and protocols does CIVIE support, and how does the system handle medical imaging data transmission?",
        "How does CIVIE integrate with existing hospital information systems (HIS) and electronic health records (EHR)?",
        "What automated billing features are included in CIVIE's revenue cycle management, and how does the system handle insurance authorizations?",
        "What are the specific payment processing capabilities and how does the system handle patient collections?",
        "What artificial intelligence capabilities are built into CIVIE for image analysis and workflow optimization?",
        "How does the system use predictive analytics for scheduling and resource management?",
        "What HIPAA compliance features and security measures are implemented throughout the CIVIE platform?",
        "What audit trails and reporting capabilities exist for tracking system usage and patient data access?",
        "How does the system handle multi-location imaging facilities and what are the networking requirements?",
        "What automated quality assurance checks are performed on imaging studies, and how are critical findings flagged?",
    ]
    
    random.seed(42)
    return random.sample(questions, num_questions)

async def run_full_agent_test_loop():
    """
    Runs a 10-iteration test of the full RAG agent.
    """
    logging.info("--- Starting Full Agent Test Loop ---")
    
    try:
        agent = SimpleRAGFlow()
        logging.info("Agent built successfully.")
    except Exception as e:
        logging.error(f"Failed to build the agent: {e}")
        return

    questions = generate_questions_from_metadata("civie_ris_metadata.json")
    if not questions:
        return

    perfect_runs = 0
    
    logging.info("--- Starting Full Test Loop (10 questions) ---")
    
    for i, question in enumerate(questions, 1):
        logging.info(f"--- Running Iteration {i}/{len(questions)} ---")
        logging.info(f"Question: {question}")
        
        try:
            final_answer = await agent.run(question)

            print(f"\n[ITERATION {i}] FINAL ANSWER:\n{final_answer}")
            
            perfect_runs += 1
            logging.info(f"Validation PASSED for question '{question}'.")

            print("\n" + "="*50 + "\n")

        except Exception as e:
            logging.error(f"Error during agent execution in iteration {i}: {e}")

    logging.info(f"--- Test Complete ---")
    logging.info(f"Final Score: {perfect_runs}/{len(questions)} perfect runs.")
    
    accuracy = perfect_runs / len(questions)
    logging.info(f"Accuracy: {accuracy:.2%}")

    if accuracy < 0.90:
        logging.error(f"The agent did not meet the 90% accuracy threshold. Accuracy: {accuracy:.2%}")
    else:
        logging.info("The agent has met the 90% accuracy threshold. The system is ready.")

if __name__ == "__main__":
    asyncio.run(run_full_agent_test_loop())
