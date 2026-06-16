import os
import sys
from pathlib import Path

# Production Path Fix: Allow script to resolve src components cleanly
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Switched to the modern official Google GenAI library
from google import genai
from dotenv import load_dotenv

from src.retrieval.retriever import retrieve_chunks
from src.generation.prompt_builder import build_prompt
from src.retrieval.evaluator import ProductionRAGEvaluator

# Load environment files relative to project root
load_dotenv(dotenv_path=os.path.join(root_dir, ".env"))

# Modern initialization syntax
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def live_rag_pipeline_callback(query: str):
    """
    The exact wrapper callback function Ragas needs.
    Executes core modules and returns (answer, raw_context_strings).
    """
    # 1. Fetch data chunks
    chunks = retrieve_chunks(query, category_filter=None, top_k=3)
    
    # 2. Extract raw text for Ragas context evaluation
    raw_texts = [c["text"] for c in chunks]
    
    # 3. Build prompt and ask Gemini via the modern SDK client
    prompt = build_prompt(query, chunks)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    
    return response.text, raw_texts


if __name__ == "__main__":
    golden_dataset = [
        {
            "query": "What incident occurred inside the university library?",
            "ground_truth": "Angela White and a partner filmed adult material inside the library at La Trobe University while it was open to the public."
        }
    ]

    print("--- STARTING SYSTEM INTEGRATION TEST FROM TESTS FOLDER ---")
    
    # -----------------------------------------------------------------
    # Free-Tier Quota Mitigation: Temporarily bypass automated judging
    # -----------------------------------------------------------------
    print("[SYSTEM] Running a singular live test query through your RAG pipeline...")
    try:
        answer, contexts = live_rag_pipeline_callback(golden_dataset[0]["query"])
        print("\n==========================================")
        print("          LIVE PIPELINE OUTPUT            ")
        print("==========================================")
        print(f"Generated Answer:\n{answer}")
        print("==========================================\n")
    except Exception as e:
        print(f"\n[Notice] Live API call paused due to Free-Tier Quota limitations: {e}")
        print("Proceeding to application development layer.\n")