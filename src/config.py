"""Configuration: settings, constants, and rate limits.

Centralizes tunable values (model id, decoding parameters, per-session
turn cap, evaluation attributes) so they can be adjusted without touching
the inference or prompt logic.
"""

import os

MIN_SCORE: int = 1
MAX_SCORE: int = 7
SESSION_TURN_CAP: int = 12

# Groq's identifier for the GPT-OSS 120B model.
# Migrated from llama-3.3-70b-versatile on 2026-06-18 following Groq's
# deprecation notice; gpt-oss-120b is the recommended replacement and
# additionally supports Groq's Structured Outputs strict mode.
MODEL_ID: str = "openai/gpt-oss-120b"
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_MAX_TOKENS: int = 1200

# Max attempts per turn before surfacing an error to the user.
# With strict JSON schema mode, parse errors are structurally impossible, so
# retries are only kept for transport-level failures (timeouts, 5xx errors).
# 1 initial attempt + 1 defensive retry = 2.
DEFAULT_MAX_ATTEMPTS: int = 2

# Per-IP daily session cap (in-app enforcement layer).
# Prevents a single visitor or bot from exhausting the project-level Groq
# budget before other users get access.  Overridable via env var for easy
# adjustment without a code redeploy.
MAX_SESSIONS_PER_IP_PER_DAY: int = int(
    os.environ.get("MAX_SESSIONS_PER_IP_PER_DAY", "2")
)


DEFAULT_ATTRIBUTES: list[str] = [
    "competent",
    "likeable", 
    "considerate",
    "polite",
    "formal",
    "demanding",
]

# Color mapping for UI rendering (matches the PRISMA prism figure).
ATTRIBUTE_COLORS: dict[str, str] = {
    "competent": "#a855f7",   # purple
    "likeable": "#f97316",    # orange
    "considerate": "#22c55e", # green
    "polite": "#eab308",      # yellow
    "formal": "#3b82f6",      # blue
    "demanding": "#e11d48",   # red
}
