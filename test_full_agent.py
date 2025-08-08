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
        "What are the specific features of CIVIE's CRM portal that help marketers track engagement and referral growth?",
        "Provide a detailed explanation of each step in CIVIE's 5-step order journey, including the technologies used at each stage.",
        "What are the key differences between the 'Standard' and 'Table' parsing modes in LlamaParse, and how do they impact the quality of extracted data?",
        "How does the RIS Insight AI System use follow-up appointments to prevent patient leakage, and what is the average patient leakage rate across the industry?",
        "What are the technical specifications of the walk-in kiosk for patient self-scheduling, and how does it reduce operational costs?",
        "Describe the architecture of CIVIE's in-house call center application, including its scripting capabilities and integration with the scheduling system.",
        "What is the statistical correlation between the 68% of patients who click the self-scheduling link and the 25% reduction in call center volume?",
        "Explain the data-driven algorithms used by CIVIE's intelligent scheduling system to optimize resource utilization, and quantify the 35% improvement in contrast scheduling.",
        "What is the full context of Dr. Krishna Das's testimonial, and how does the 30% cost reduction figure relate to the other metrics mentioned?",
        "Who are the key members of CIVIE's executive team, and what are their roles and responsibilities?",
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
