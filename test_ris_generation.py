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

from rag_chatbot.app.models.data_models import Document
from rag_chatbot.app.services.agent.nodes import GenerateAnswerNode
from rag_chatbot.app.services.query_engine.generation import AnswerGenerator
from rag_chatbot.app.core.llm import CustomLLM

def load_test_data(file_path: str) -> list[Document]:
    """Loads parent chunks from the specified JSON file and converts them to Document objects."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        parent_chunks = data.get("parent_chunks", [])
        
        return [
            Document(
                id=chunk.get("chunk_id"),
                text=chunk.get("metadata", {}).get("text", ""),
                metadata=chunk.get("metadata", {})
            )
            for chunk in parent_chunks
        ]
    except Exception as e:
        logging.error(f"Failed to load or parse test data from {file_path}: {e}")
        return []

def generate_questions(context_documents: list[Document]) -> list[str]:
    """Generates a diverse set of questions based on the provided context."""
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
    
    # Add more questions to reach 100
    additional_questions = [f"Summarize the benefits of CIVIE's solution mentioned in chunk {i}" for i in range(len(context_documents))]
    questions.extend(additional_questions)
    
    # Ensure we have 100 unique questions
    unique_questions = list(set(questions))
    if len(unique_questions) < 100:
        # Add generic questions if we still don't have enough
        for i in range(100 - len(unique_questions)):
            unique_questions.append(f"What details are provided in the document about CIVIE's offerings? (query {i})")
            
    return random.sample(unique_questions, 100)

def validate_response(question: str, answer: str, citations: list, context: list[Document]) -> bool:
    """Validates the generated answer and citations."""
    if not answer or "ERROR" in answer or "could not generate" in answer.lower():
        logging.warning(f"Validation FAILED for question '{question}': Invalid answer generated.")
        return False
        
    if not citations:
        logging.warning(f"Validation FAILED for question '{question}': No citations were generated.")
        return False

    # Simple keyword check for relevance
    question_keywords = set(question.lower().split()) - {'what', 'is', 'are', 'the', 'a', 'an'}
    answer_lower = answer.lower()
    if not any(keyword in answer_lower for keyword in question_keywords):
        logging.warning(f"Validation FAILED for question '{question}': Answer does not seem relevant (missing keywords).")
        return False

    # Check if cited sources are valid
    for citation in citations:
        source_name = citation.get('source_name')
        page_no = citation.get('page_number')
        
        # In this test, all sources should be from the same file
        if not source_name or "CIVIE-RIS.pdf" not in source_name:
            logging.warning(f"Validation FAILED for question '{question}': Invalid source name '{source_name}'.")
            return False
            
    logging.info(f"Validation PASSED for question '{question}'.")
    return True

def run_generation_test_loop():
    """
    Runs a 100-iteration test of the GenerateAnswerNode.
    """
    logging.info("--- Starting Advanced Generation Test Loop ---")
    
    context_documents = load_test_data("civie_ris_metadata.json")
    if not context_documents:
        return

    try:
        llm = CustomLLM()
        answer_generator = AnswerGenerator(llm)
        generation_node = GenerateAnswerNode(answer_generator)
        logging.info("Components initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize components: {e}")
        return

    questions = generate_questions(context_documents)
    perfect_runs = 0
    
    for i, question in enumerate(questions, 1):
        logging.info(f"--- Running Iteration {i}/100 ---")
        logging.info(f"Question: {question}")
        
        test_state = {
            "query_state": {"condensed_query": question},
            "context_state": {"parent_chunks": context_documents},
            "generation_state": {"is_streaming": False}
        }
        
        try:
            result_state = generation_node.run(test_state)
            final_answer = result_state.get("generation_state", {}).get("final_answer", "ERROR: No answer.")
            rich_citations = result_state.get("generation_state", {}).get("rich_citations", [])

            print(f"\n[ITERATION {i}] FINAL ANSWER:\n{final_answer}")
            
            if rich_citations:
                print(f"\n[ITERATION {i}] CITATIONS:")
                for c in rich_citations:
                    print(f"  - Source: {c.get('source_name', 'N/A')}, Page: {c.get('page_number', 'N/A')}")
            
            if validate_response(question, final_answer, rich_citations, context_documents):
                perfect_runs += 1

            print("\n" + "="*50 + "\n")

        except Exception as e:
            logging.error(f"Error during generation in iteration {i}: {e}")

    logging.info(f"--- Test Complete ---")
    logging.info(f"Final Score: {perfect_runs}/100 perfect runs.")
    
    if perfect_runs < 99:
        logging.error("The agent did not meet the 99% accuracy threshold. Further refinement is needed.")
    else:
        logging.info("The agent has met the 99% accuracy threshold. The system is ready.")

if __name__ == "__main__":
    run_generation_test_loop()
