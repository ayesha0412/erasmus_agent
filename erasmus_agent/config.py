"""Loads settings from .env and lists the Groq models the agent can use."""
import os

from dotenv import load_dotenv

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

MODELS = {
    "main": "openai/gpt-oss-120b",  # stronger tool calling and reasoning
    "fast": "openai/gpt-oss-20b",   # fallback: faster, separate rate limit
}

DEFAULT_MODEL = os.getenv("LLM_MODEL", "main")