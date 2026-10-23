"""One doorway to the LLM: every part of the agent calls chat() and nothing else."""
import os
import time

from openai import OpenAI

from .config import DEFAULT_MODEL, GROQ_BASE_URL, MODELS


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
            