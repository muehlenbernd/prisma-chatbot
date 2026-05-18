"""Configuration: settings, constants, and rate limits.

Centralizes tunable values (model id, decoding parameters, per-session
turn cap, evaluation attributes) so they can be adjusted without touching
the inference or prompt logic.

Implementation pending — scaffolding only.
"""

DEFAULT_ATTRIBUTES: list[str] = [
    "competent",
    "likeable", 
    "considerate",
    "polite",
    "formal",
    "demanding",
]
