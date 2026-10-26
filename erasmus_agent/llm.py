"""One doorway to the LLM: every part of the agent calls chat() or chat_json() and nothing else."""
import json
import os
import time

from openai import OpenAI, RateLimitError

from .config import DEFAULT_MODEL, GROQ_BASE_URL, MODELS

RATE_LIMIT_ATTEMPTS = 5   # tries when Groq answers "too many requests"
RATE_LIMIT_WAIT = 30      # seconds to wait between those tries


def get_client() -> OpenAI:
    """Return a Groq client, or explain clearly if the key is missing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
    return OpenAI(base_url=GROQ_BASE_URL, api_key=api_key)


def chat(messages: list[dict], model: str = DEFAULT_MODEL, tools: list[dict] | None = None, **kwargs):
    """Send messages to a GPT-OSS model. Temperature 0 and a fixed seed keep runs repeatable."""
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {', '.join(MODELS)}")
    if tools:
        kwargs["tools"] = tools
    return get_client().chat.completions.create(
        model=MODELS[model], messages=messages, temperature=0, seed=42, **kwargs
    )


def chat_json(messages: list[dict], schema_name: str, schema: dict, model: str = DEFAULT_MODEL,
              reasoning_effort: str = "low") -> tuple[str, int]:
    """Ask for a reply shaped by a JSON schema. Returns (reply text, tokens used).

    The reply is returned as text, not parsed: the caller validates it and decides what
    to do if it is wrong. Waits and retries only when Groq's rate limit is hit.
    """
    response_format = {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}}
    for attempt in range(1, RATE_LIMIT_ATTEMPTS + 1):
        try:
            response = chat(messages, model=model, response_format=response_format,
                            reasoning_effort=reasoning_effort)
            break
        except RateLimitError:
            if attempt == RATE_LIMIT_ATTEMPTS:
                raise
            print(f"    rate limit reached, waiting {RATE_LIMIT_WAIT}s (attempt {attempt}/{RATE_LIMIT_ATTEMPTS})")
            time.sleep(RATE_LIMIT_WAIT)
    tokens = response.usage.total_tokens if response.usage else 0
    return response.choices[0].message.content or "", tokens


def ping(model: str) -> tuple[str, float]:
    """Ask for a one-word reply and time it."""
    start = time.perf_counter()
    response = chat([{"role": "user", "content": "Reply with exactly one word: pong"}], model=model)
    return (response.choices[0].message.content or "").strip(), time.perf_counter() - start


if __name__ == "__main__":
    for name, model_id in MODELS.items():
        try:
            text, seconds = ping(name)
            print(f"OK    {name:5} {model_id:22} {seconds:5.1f}s  -> {text[:40]!r}")
        except Exception as error:
            print(f"FAIL  {name:5} {model_id:22} {type(error).__name__}: {error}")
