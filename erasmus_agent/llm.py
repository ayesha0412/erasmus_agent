"""One doorway to the LLM: every part of the agent calls chat() or chat_json() and nothing else."""
import json
import os
import time

from openai import OpenAI, RateLimitError

from .config import DEFAULT_MODEL, GROQ_BASE_URL, LOCAL_BASE_URL, LOCAL_MODELS, MODELS
from .logbook import log_event

RATE_LIMIT_ATTEMPTS = 5   # tries when Groq answers "too many requests"
RATE_LIMIT_WAIT = 30      # seconds to wait between those tries


def get_client(model: str = DEFAULT_MODEL) -> OpenAI:
    """Return a client for the model's provider: Ollama for local models, otherwise Groq."""
    if model in LOCAL_MODELS:
        return OpenAI(base_url=LOCAL_BASE_URL, api_key="ollama")   # Ollama ignores the key
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
    if model in LOCAL_MODELS:
        kwargs.pop("reasoning_effort", None)   # a Groq option the local model does not use
    return get_client(model).chat.completions.create(
        model=MODELS[model], messages=messages, temperature=0, seed=42, **kwargs
    )


FALLBACK = {"main": "fast", "fast": "local"}  # if a model's DAILY budget is used up, switch instead of waiting


def is_daily_limit(error: RateLimitError) -> bool:
    """Groq says 'tokens per day' or 'requests per day' when the daily budget is gone; waiting won't help."""
    return "per day" in str(error)


def chat_with_retry(messages: list[dict], model: str, purpose: str = "chat", **kwargs):
    """chat(), but safe against rate limits. Returns (response, model actually used).

    Per-minute limit: wait and retry. Daily limit: switch to the fallback model at once.
    Every call and every limit event is written to the run log.
    """
    requested = model
    for attempt in range(1, RATE_LIMIT_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            response = chat(messages, model=model, **kwargs)
        except RateLimitError as error:
            daily = is_daily_limit(error)
            switch = daily and model in FALLBACK
            action = f"switch to {FALLBACK[model]}" if switch else ("give up" if daily else "wait")
            log_event("rate_limit", purpose=purpose, model=MODELS[model], limit="daily" if daily else "per_minute",
                      action=action)
            if switch:
                print(f"    daily token limit reached for '{model}', switching to '{FALLBACK[model]}'")
                model = FALLBACK[model]
                continue
            if attempt == RATE_LIMIT_ATTEMPTS or daily:
                raise
            print(f"    rate limit reached, waiting {RATE_LIMIT_WAIT}s (attempt {attempt}/{RATE_LIMIT_ATTEMPTS})")
            time.sleep(RATE_LIMIT_WAIT)
            continue
        usage = response.usage
        log_event("llm_call", purpose=purpose, model=MODELS[model], fallback=model != requested,
                  prompt_tokens=getattr(usage, "prompt_tokens", 0), completion_tokens=getattr(usage, "completion_tokens", 0),
                  tokens=getattr(usage, "total_tokens", 0), latency_ms=int((time.perf_counter() - started) * 1000))
        return response, model
    raise RuntimeError("no model available")


def chat_json(messages: list[dict], schema_name: str, schema: dict, model: str = DEFAULT_MODEL,
              reasoning_effort: str = "low", purpose: str = "json") -> tuple[str, int]:
    """Ask for a reply shaped by a JSON schema. Returns (reply text, tokens used).

    The reply is returned as text, not parsed: the caller validates it and decides what
    to do if it is wrong.
    """
    response_format = {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema}}
    response, _ = chat_with_retry(messages, model, purpose=purpose, response_format=response_format,
                                  reasoning_effort=reasoning_effort)
    tokens = response.usage.total_tokens if response.usage else 0
    return response.choices[0].message.content or "", tokens


def chat_tools(messages: list[dict], tools: list[dict], model: str = DEFAULT_MODEL,
               reasoning_effort: str = "medium", purpose: str = "decide") -> tuple[str | None, str, int, str]:
    """Ask the model to choose ONE tool. Returns (tool name or None, arguments as JSON text, tokens, model used)."""
    response, used = chat_with_retry(messages, model, purpose=purpose, tools=tools, tool_choice="required",
                                     reasoning_effort=reasoning_effort)
    tokens = response.usage.total_tokens if response.usage else 0
    message = response.choices[0].message
    if not message.tool_calls:
        return None, message.content or "", tokens, used
    call = message.tool_calls[0]
    return call.function.name, call.function.arguments or "{}", tokens, used


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
