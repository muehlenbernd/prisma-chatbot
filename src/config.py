"""Configuration: settings, constants, and rate limits.

Centralizes tunable values (model id, decoding parameters, per-session
turn cap, evaluation attributes) so they can be adjusted without touching
the inference or prompt logic.

Implementation pending — scaffolding only.
"""

MIN_SCORE: int = 1
MAX_SCORE: int = 7
SESSION_TURN_CAP: int = 12

DEFAULT_ATTRIBUTES: list[str] = [
    "competent",
    "likeable", 
    "considerate",
    "polite",
    "formal",
    "demanding",
]
