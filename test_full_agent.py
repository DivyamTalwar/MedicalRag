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
        "What is the primary purpose of the CIVIE CRM Portal and what are its main features?",
        "Explain the 5-step order journey as described in the CIVIE documentation.",
        "What are the key functionalities of CIVIE's Radiology Information System (RIS)?",
        "How does the RIS Insight AI System help in preventing patient leakage?",
        "What different methods does CIVIE provide for patient self-scheduling?",
        "Describe the capabilities of CIVIE's in-house call center application.",
        "What percentage of patients use the self-scheduling link?",
        "By what percentage does CIVIE's contrast scheduling improve resource optimization?",
        "What is the cost reduction percentage mentioned by Dr. Krishna Das from Sol Radiology?",
        "Who holds the position of Chief Financial Officer at CIVIE?",
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
            
            if len(final_answer) > 20 and "i don't know" not in final_answer.lower() and "could not find" not in final_answer.lower():
                perfect_runs += 1
                logging.info(f"Validation PASSED for question '{question}'.")
            else:
                logging.warning(f"Validation FAILED for question '{question}': Invalid final answer generated.")

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
