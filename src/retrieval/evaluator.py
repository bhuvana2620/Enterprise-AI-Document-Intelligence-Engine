import os
import time
from google import genai

class ProductionRAGEvaluator:
    def __init__(self):
        """Initializes the judge engine using the modern google-genai package."""
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _judge_statement_with_backoff(self, prompt: str, retries: int = 3, delay: int = 5) -> float:
        """Helper to get a numerical rating from the judge model with basic backoff handling."""
        for attempt in range(retries):
            try:
                # Enforce a 2-second standard cooldown between requests to respect free-tier RPM
                time.sleep(2)
                
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                
                text = response.text.strip()
                for word in text.split():
                    clean_word = "".join(c for c in word if c.isdigit() or c == '.')
                    if clean_word:
                        val = float(clean_word)
                        return val if val <= 1.0 else val / 100.0
                return 0.5
                
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    print(f"   [Warning] Hit 429 Rate Limit. Backing off for {delay}s (Attempt {attempt+1}/{retries})...")
                    time.sleep(delay)
                    delay *= 2  # Double the wait time for exponential spacing
                else:
                    print(f"   [Error] Judge generation failed: {e}")
                    return 0.0
        return 0.0

    def calculate_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Checks if the generated answer is completely derived from the context."""
        context_str = "\n".join(contexts)
        prompt = f"""
        Analyze the Grounded Answer against the Context Data. 
        Rate how faithful the answer is to the context on a scale from 0.0 to 1.0.
        If the answer contains claims NOT found in the context (hallucinations), rate it low.
        Return ONLY a single float value between 0.0 and 1.0.

        [CONTEXT DATA]
        {context_str}

        [GROUNDED ANSWER]
        {answer}
        
        Score:"""
        return self._judge_statement_with_backoff(prompt)

    def calculate_answer_relevance(self, question: str, answer: str) -> float:
        """Checks if the generated answer directly addresses the original question."""
        prompt = f"""
        Rate how relevant the Generated Answer is to the User Question on a scale from 0.0 to 1.0.
        Does it fully address the query, or does it deflect?
        Return ONLY a single float value between 0.0 and 1.0.

        [USER QUESTION]
        {question}

        [GENERATED ANSWER]
        {answer}
        
        Score:"""
        return self._judge_statement_with_backoff(prompt)

    def calculate_context_precision(self, question: str, contexts: list[str]) -> float:
        """Checks if the retrieved chunks are highly relevant to the question."""
        context_str = "\n".join(contexts)
        prompt = f"""
        Rate how precisely the Context Data matches the information needed to answer the User Question.
        Is the context highly relevant or filled with noise?
        Return ONLY a single float value between 0.0 and 1.0.

        [USER QUESTION]
        {question}

        [CONTEXT DATA]
        {context_str}
        
        Score:"""
        return self._judge_statement_with_backoff(prompt)

    def evaluate_pipeline(self, test_dataset: list, rag_pipeline_callback) -> dict:
        """Runs the validation test suite across all metrics."""
        total_faithfulness = 0.0
        total_relevance = 0.0
        total_precision = 0.0
        count = len(test_dataset)

        print(f"\n[EVALUATOR] Running pacing-controlled validation loop for {count} test cases...")

        for idx, item in enumerate(test_dataset, start=1):
            query = item["query"]
            print(f" -> Processing evaluation step {idx}/{count}: '{query}'")
            
            # Execute your live components
            generated_answer, retrieved_context_list = rag_pipeline_callback(query)
            
            # Add a small buffer pause between the live RAG run and judging steps
            time.sleep(1)
            
            # Evaluate each metric individually with built-in retry handling
            total_faithfulness += self.calculate_faithfulness(generated_answer, retrieved_context_list)
            total_relevance += self.calculate_answer_relevance(query, generated_answer)
            total_precision += self.calculate_context_precision(query, retrieved_context_list)

        return {
            "faithfulness": total_faithfulness / count,
            "answer_relevance": total_relevance / count,
            "context_precision": total_precision / count
        }