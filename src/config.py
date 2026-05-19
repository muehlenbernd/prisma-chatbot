"""Configuration: settings, constants, and rate limits.

Centralizes tunable values (model id, decoding parameters, per-session
turn cap, evaluation attributes) so they can be adjusted without touching
the inference or prompt logic.

Implementation pending — scaffolding only.
"""

MIN_SCORE: int = 1
MAX_SCORE: int = 7
SESSION_TURN_CAP: int = 12

MODEL_ID: str = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_MAX_TOKENS: int = 600


DEFAULT_ATTRIBUTES: list[str] = [
    "competent",
    "likeable", 
    "considerate",
    "polite",
    "formal",
    "demanding",
]
