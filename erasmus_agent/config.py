"""Loads settings from .env and lists the models the agent can use (Groq in the cloud, Ollama locally)."""
import os

from dotenv import load_dotenv

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Ollama speaks the same OpenAI-style API; OLLAMA_HOST is Ollama's own setting (default port 11434).
LOCAL_BASE_URL = f"http://{os.getenv('OLLAMA_HOST', '127.0.0.1:11434')}/v1"

MODELS = {
    "main": "openai/gpt-oss-120b",                    # Groq: stronger tool calling and reasoning
    "fast": "openai/gpt-oss-20b",                     # Groq: faster, separate daily budget
    "local": os.getenv("LOCAL_MODEL", "qwen3:14b"),   # Ollama on this computer: no rate limits, data stays local
}
LOCAL_MODELS = {"local"}

DEFAULT_MODEL = os.getenv("LLM_MODEL", "main")      # the agent's next-action decisions
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "fast")  # page extraction: simpler task, separate daily budget
