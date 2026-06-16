# src/generation/llm_generator.py
"""
Legacy Gemini-only generator.

This file is not used by the FastAPI backend.
The production backend uses src/generation/llm_client.py.
Keep this only for old local experiments.
"""
import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError

from src.retrieval.retriever import retrieve_chunks
from src.generation.prompt_builder import build_prompt

# ---------------------------------------------------
# Load environment variables
# ---------------------------------------------------
load_dotenv()

# ---------------------------------------------------
# Initialize the Modern Gemini Client
# ---------------------------------------------------
# The new SDK automatically picks up the GEMINI_API_KEY environment variable!
client = genai.Client()

# Recommended ultra-fast model
MODEL_ID = "gemini-3.1-flash-lite"


# ---------------------------------------------------
# Generate Final Answer
# ---------------------------------------------------
def generate_answer(query):
    """Full RAG pipeline with robust error handling for prepaid accounts."""

    # Step 1: Retrieve relevant chunks
    retrieved_chunks = retrieve_chunks(query)

    # Step 2: Build prompt
    prompt = build_prompt(query, retrieved_chunks)

    # Step 3: Generate response with handling for 429 quota/billing blockages
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_ID, contents=prompt
            )
            return response.text

        except APIError as e:
            # Check if it's a 429 error (Resource Exhausted / Limit = 0)
            if e.code == 429:
                if "limit: 0" in str(e).lower() or "billing" in str(e).lower():
                    return (
                        "❌ [Billing Block] Google has locked your Free Tier because this API Key is tied to a \n"
                        "   Paid Project with a $0.00 balance. \n\n"
                        "👉 FIX: Go back to Google AI Studio, create a BRAND NEW API Key inside a \n"
                        "   'New Project' completely separate from your 'My Billing Account' profile."
                    )

                # Standard rate limit hit, backoff and retry
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(
                        f"⚠️ Rate limit hit. Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
                else:
                    return "Error: Rate limits exceeded. Please try again in a moment."
            else:
                return f"Gemini API Error ({e.code}): {e.message}"

        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"


# ---------------------------------------------------
# Main Test Runner
# ---------------------------------------------------
if __name__ == "__main__":

    query = "What happened at the university?"

    print("\nGenerating AI Answer...\n")

    answer = generate_answer(query)

    print("===== FINAL ANSWER =====\n")
    print(answer)