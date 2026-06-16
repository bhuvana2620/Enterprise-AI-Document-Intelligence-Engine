# src/generation/prompt_builder.py

def build_prompt(query, retrieved_chunks):
    """
    Build a RAG prompt using retrieved chunk dictionaries, 
    injecting explicit source tracking and citation guidelines.
    """
    
    # STEP 6: Construct an indexed reference block out of our dictionary metadata fields
    formatted_context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        block = f"--- SOURCE REFERENCE [{idx}]: {chunk['source']} ---\n{chunk['text']}"
        formatted_context_blocks.append(block)
        
    context = "\n\n".join(formatted_context_blocks)

    # STEP 7: Re-engineer the prompt text template to strictly enforce hallucination guardrails
    prompt = f"""You are a precise, security-oriented corporate QA intelligence model. 
Your goal is to answer the user question using ONLY the provided contexts listed below.

==================================================
CONTEXT DATA
==================================================
{context}
==================================================

USER QUESTION: {query}

CRITICAL EXECUTION DIRECTIVES FOR GENERATION:
1. Grounding: Rely entirely on facts explicitly mentioned inside the CONTEXT DATA above. Do not bring in external knowledge, logic assumptions, or unsourced facts.
2. CITATIONS: For every factual claim, conclusion, or piece of evidence you write, append the exact corresponding numeric citation wrapper inline directly where it occurs (e.g., "The team verified completion on Tuesday [1]"). Do not clump references at the very end of your response.
3. HALLUCINATION SAFE GUARDRAIL: If the provided context lacks sufficient information to clearly answer the question, or if you cannot trace the answer directly to a source block, respond EXACTLY with:
"I could not find the answer in the provided documents."
Do not attempt to write a partial or speculative answer.

Grounded Answer:"""

    return prompt