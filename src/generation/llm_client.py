# src/generation/llm_client.py

import os
import time
import random
from pathlib import Path
from typing import Optional, Callable

try:
    from dotenv import load_dotenv

    ROOT_DIR = Path(__file__).resolve().parents[2]
    load_dotenv(ROOT_DIR / ".env")
except Exception:
    pass


# ---------------------------------------------------
# Error Detection Helpers
# ---------------------------------------------------
def is_quota_error(error_text: str) -> bool:
    """
    Detect quota/rate-limit errors across Gemini, Grok, and OpenAI-compatible APIs.
    """
    lowered = error_text.lower()

    return (
        "429" in lowered
        or "resource_exhausted" in lowered
        or "quota" in lowered
        or "rate limit" in lowered
        or "rate_limit" in lowered
        or "too many requests" in lowered
        or "insufficient_quota" in lowered
    )


def is_retryable_error(error_text: str) -> bool:
    """
    Detect temporary provider errors worth retrying.
    """
    lowered = error_text.lower()

    return (
        is_quota_error(error_text)
        or "500" in lowered
        or "502" in lowered
        or "503" in lowered
        or "504" in lowered
        or "timeout" in lowered
        or "temporarily unavailable" in lowered
        or "connection" in lowered
        or "overloaded" in lowered
    )


def clean_error(error: Exception, max_chars: int = 300) -> str:
    """
    Keep provider errors short and log-safe.
    """
    return str(error).replace("\n", " ")[:max_chars]


def retry_delay(attempt: int) -> float:
    """
    Exponential backoff with jitter.
    """
    return min(30.0, 2 ** attempt) + random.uniform(0, 1.5)


# ---------------------------------------------------
# Gemini Model Config
# ---------------------------------------------------
def get_gemini_models() -> list[str]:
    """
    Reads Gemini model order from env.

    Primary:
      GEMINI_MODEL=gemini-3.1-flash-lite

    Fallbacks:
      GEMINI_FALLBACK_MODELS=gemini-2.5-flash-lite,gemini-2.0-flash-lite
    """
    primary = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()

    fallback_raw = os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.5-flash-lite,gemini-2.0-flash-lite"
    )

    models = [primary]

    for model in fallback_raw.split(","):
        model = model.strip()
        if model:
            models.append(model)

    unique_models = []
    seen = set()

    for model in models:
        if model not in seen:
            unique_models.append(model)
            seen.add(model)

    return unique_models


# ---------------------------------------------------
# Mock Fallback
# ---------------------------------------------------
def generate_with_mock(prompt: str, reason: Optional[str] = None) -> str:
    """
    Last-resort fallback.

    This guarantees the backend still returns a response even if all live LLMs fail.
    """
    show_reason = os.getenv("SHOW_LLM_FALLBACK_REASON", "true").lower() == "true"

    message = (
        "I could not reach the live LLM provider right now, but the retrieval "
        "pipeline is working. Below is a preview of the retrieved context and prompt "
        "that would have been sent to the model."
    )

    if reason and show_reason:
        message += f"\n\nFallback reason: {reason}"

    prompt_preview = prompt[:1500]

    return (
        f"{message}\n\n"
        "Retrieved prompt/context preview:\n"
        f"{prompt_preview}"
    )


# ---------------------------------------------------
# Retry Wrapper
# ---------------------------------------------------
def call_with_retry(
    provider_name: str,
    call_fn: Callable[[], str],
    max_retries: Optional[int] = None
) -> str:
    """
    Retry temporary LLM failures before falling back.
    """
    retries = max_retries or int(os.getenv("LLM_MAX_RETRIES", "3"))

    last_error = None

    for attempt in range(retries):
        try:
            return call_fn()

        except Exception as e:
            last_error = e
            error_text = str(e)

            if attempt < retries - 1 and is_retryable_error(error_text):
                wait_seconds = retry_delay(attempt)

                print(
                    f"[LLM] {provider_name} temporary error. "
                    f"Retrying in {wait_seconds:.1f}s "
                    f"({attempt + 2}/{retries}). Error: {clean_error(e)}",
                    flush=True
                )

                time.sleep(wait_seconds)
                continue

            raise e

    raise last_error


# ---------------------------------------------------
# Gemini Provider
# ---------------------------------------------------
def generate_with_gemini(prompt: str) -> str:
    """
    Generate answer using Google Gemini.

    Tries GEMINI_MODEL first, then GEMINI_FALLBACK_MODELS.
    """
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")

    client = genai.Client(api_key=api_key)

    errors = []

    for model_name in get_gemini_models():
        try:
            print(f"[LLM] Trying Gemini model: {model_name}", flush=True)

            def call():
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )

                return (
                    response.text
                    or "I could not find the answer in the provided documents."
                )

            return call_with_retry(
                provider_name=f"gemini:{model_name}",
                call_fn=call
            )

        except Exception as e:
            errors.append(f"{model_name}: {clean_error(e)}")
            print(
                f"[LLM] Gemini model failed: {model_name}. Error: {clean_error(e)}",
                flush=True
            )

    raise RuntimeError("All Gemini models failed. " + " | ".join(errors))


# ---------------------------------------------------
# Grok Provider
# ---------------------------------------------------
def generate_with_grok(prompt: str) -> str:
    """
    Generate answer using xAI Grok through the OpenAI-compatible API.
    """
    from openai import OpenAI

    api_key = os.getenv("XAI_API_KEY")

    if not api_key:
        raise ValueError("Missing XAI_API_KEY.")

    base_url = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
    model_name = os.getenv("XAI_MODEL", "grok-4.3")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    print(f"[LLM] Trying Grok model: {model_name}", flush=True)

    def call():
        response = client.responses.create(
            model=model_name,
            input=prompt,
            temperature=0.0
        )

        if hasattr(response, "output_text") and response.output_text:
            return response.output_text

        try:
            return response.output[0].content[0].text
        except Exception:
            return "I could not find the answer in the provided documents."

    return call_with_retry(
        provider_name=f"grok:{model_name}",
        call_fn=call
    )


# ---------------------------------------------------
# Main Router
# ---------------------------------------------------
def generate_answer(prompt: str) -> str:
    """
    Main LLM provider router.

    Supported:
      LLM_PROVIDER=gemini
      LLM_PROVIDER=grok
      LLM_PROVIDER=mock
      LLM_PROVIDER=auto

    Recommended:
      LLM_PROVIDER=auto
    """
    mock_mode = os.getenv("MOCK_LLM_MODE", "false").lower() == "true"

    if mock_mode:
        return generate_with_mock(prompt, reason="MOCK_LLM_MODE=true")

    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

    if provider == "mock":
        return generate_with_mock(prompt)

    if provider == "gemini":
        try:
            return generate_with_gemini(prompt)
        except Exception as e:
            if is_quota_error(str(e)):
                return generate_with_mock(
                    prompt,
                    reason=f"Gemini quota/rate-limit error: {clean_error(e)}"
                )

            raise e

    if provider == "grok":
        try:
            return generate_with_grok(prompt)
        except Exception as e:
            if is_quota_error(str(e)):
                return generate_with_mock(
                    prompt,
                    reason=f"Grok quota/rate-limit error: {clean_error(e)}"
                )

            raise e

    if provider == "auto":
        errors = []

        # 1. Try Gemini first.
        try:
            return generate_with_gemini(prompt)
        except Exception as e:
            errors.append(f"gemini: {clean_error(e)}")
            print(
                f"[LLM] Gemini failed in auto mode: {clean_error(e)}",
                flush=True
            )

        # 2. Try Grok second.
        try:
            return generate_with_grok(prompt)
        except Exception as e:
            errors.append(f"grok: {clean_error(e)}")
            print(
                f"[LLM] Grok failed in auto mode: {clean_error(e)}",
                flush=True
            )

        # 3. Last-resort mock response.
        return generate_with_mock(
            prompt,
            reason="All configured LLM providers failed. " + " | ".join(errors)
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER='{provider}'. "
        "Use gemini, grok, mock, or auto."
    )