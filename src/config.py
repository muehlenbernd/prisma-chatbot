"""Configuration: settings, constants, and rate limits.

Centralizes tunable values (model id, decoding parameters, per-session
turn cap, evaluation attributes) so they can be adjusted without touching
the inference or prompt logic.

Implementation pending — scaffolding only.
"""

MIN_SCORE: int = 1
MAX_SCORE: int = 7
SESSION_TURN_CAP: int = 12

# Groq's identifier for Llama 3.3 70B Instruct.
MODEL_ID: str = "llama-3.3-70b-versatile"
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_MAX_TOKENS: int = 1200

# Max attempts per turn before surfacing an EvaluationParseError to the user.
# Llama 3.3 70B under json_object mode occasionally emits JSON that omits or
# mis-types the 'response' field; resampling at the same temperature usually
# fixes it, so we resample up to (DEFAULT_MAX_ATTEMPTS - 1) times before
# giving up.
DEFAULT_MAX_ATTEMPTS: int = 5


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
