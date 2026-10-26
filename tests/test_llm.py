"""Tests for the rate-limit handling in llm.py (milestone M12, needed early): no real API calls."""

import types

import httpx2
import pytest
from openai import RateLimitError

from erasmus_agent import llm


def limit_error(message: str) -> RateLimitError:
    request = httpx2.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return RateLimitError(message, response=httpx2.Response(429, request=request), body=None)


def fake_response() -> object:
    message = types.SimpleNamespace(content="{}", tool_calls=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)],
                                 usage=types.SimpleNamespace(total_tokens=10))


def test_daily_limit_switches_to_the_fallback_model_without_waiting(monkeypatch):
    used = []

    def chat(messages, model, **kwargs):
        used.append(model)
        if model == "main":
            raise limit_error("Rate limit reached ... on tokens per day (TPD): Limit 200000")
        return fake_response()
    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: pytest.fail("must not wait for a daily limit"))
    _, model = llm.chat_with_retry([], "main")
    assert used == ["main", "fast"] and model == "fast"


def test_per_minute_limit_waits_and_retries_the_same_model(monkeypatch):
    calls = []

    def chat(messages, model, **kwargs):
        calls.append(model)
        if len(calls) == 1:
            raise limit_error("Rate limit reached ... on tokens per minute (TPM): Limit 8000")
        return fake_response()
    monkeypatch.setattr(llm, "chat", chat)
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    _, model = llm.chat_with_retry([], "main")
    assert calls == ["main", "main"] and model == "main"


def test_daily_limit_on_the_last_model_is_raised(monkeypatch):
    def chat(messages, model, **kwargs):
        raise limit_error("Rate limit reached ... on tokens per day (TPD)")
    monkeypatch.setattr(llm, "chat", chat)
    with pytest.raises(RateLimitError):
        llm.chat_with_retry([], "fast")
