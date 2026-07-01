import os

# Context Builder Settings
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "32000"))

# Characters per token approximation (fallback for token budget checks)
CHARS_PER_TOKEN = 4

# LLM Generation Settings
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))

# Fallback models in case of limits
GEMINI_FALLBACK_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-3.1-flash-lite",
    "gemini-3-flash"
]
