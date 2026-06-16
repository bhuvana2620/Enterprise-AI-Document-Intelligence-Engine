"""
Live RAG pipeline evaluation using a custom LLM-as-judge evaluator
(src/retrieval/evaluator.py — ProductionRAGEvaluator).

This is an INTEGRATION test, not a unit test:
- Requires a real GEMINI_API_KEY
- Requires a populated Pinecone index with matching content
- Makes live network calls to Gemini for both generation and judging
- Is rate-limited by design (sleeps between calls) to respect free-tier RPM

It does NOT run in CI by default. Run manually with:

    RUN_EVAL=true python3 tests/test_rag_evaluation.py

Replace the golden_dataset below with queries that match documents
actually present in your Pinecone namespace before running.
"""

import os
import sys
from pathlib import Path

ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))


def live_rag_pipeline_callback(query: str):
    """
    Callba the evaluator calls per test case.
    Returns (generated_answer, retrieved_context_strings).
    """
    from google import genai
    from src.retrieval.retriever import retrieve_chunks
    from src.generation.prompt_builder import build_prompt

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    chunks = retrieve_chunks(query, category_filter=None, top_k=3)
    raw_texts = [c["text"] for c in chunks]

    prompt = build_prompt(query, chunks)
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        contents=prompt,
    )

    return response.text, raw_texts


def run_evaluation():
    from src.retrieval.evaluator import ProductionRAGEvaluator

    golden_dataset = [
        {
            "query": "What is the main topic covered in the uploaded documents?",
            "ground_truth": "Replace with a query and expected answer matching your actual indexed documents.",
        },
    ]

    print("--- RAG PIPELINE EVALUATION (LLM-as-judge) ---")

    evaluator = ProductionRAGEvaluator()
    results = evaluator.evaluate_pipeline(golden_dataset, live_rag_pipeline_callback)

    print("\n==========================================")
    print("              EVALUATION RESULTS           ")
    print("==========================================")
    for metric, score in results.items():
        print(f"{metric}: {score:.3f}")
    print("==========================================\n")

    return results


if __name__ == "__main__":
    if os.getenv("RUN_EVAL", "false").strip().lower() != "true":
        print(
            "Skipping live evaluation (set RUN_EVAL=true to run).\n"
            "This test makes real Gemini API calls and requires a populated "
            "Pinecone index — it is not a unit test."
        )
        sys.exit(0)

    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not set. Cannot run live evaluation.")
        sys.exit(1)

    run_evaluation()
