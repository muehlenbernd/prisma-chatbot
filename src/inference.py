"""Groq Inference API client wrapper for Prisma.

Provides PrismaInferenceClient, a small wrapper around the groq SDK's
client that:
- Forces JSON output via response_format={"type": "json_object"}.
  This is required for reliable structured output with Llama 3.3 70B,
  which otherwise produces conversational text before/instead of JSON.
- Parses and validates the response via src.evaluation.
- Raises typed errors for API failures (InferenceError) and parse
  failures (EvaluationParseError, propagated from evaluation).

The wrapper is initialized once per session with a Groq API key; each
generate() call sends a full message history (system + conversation)
and returns a validated ParsedTurn.
"""

from __future__ import annotations

from typing import Sequence

from groq import APIError, Groq

from .config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, MODEL_ID
from .evaluation import ParsedTurn, parse_model_output

# Single chat message in OpenAI format. Kept loose for v1; can tighten to
# a TypedDict later if message shapes diversify.
ChatMessage = dict[str, str]


class InferenceError(Exception):
    """Raised when the inference API call fails or returns malformed data.

    Wraps network errors, authentication failures, rate-limit errors,
    and missing-field errors in the API response. Parse errors on the
    model's content are *not* wrapped here — they surface as
    EvaluationParseError so the app layer can distinguish them.
    """


class PrismaInferenceClient:
    """Wrapper around the Groq client configured for Prisma.

    Holds a single ``Groq`` instance and exposes a ``generate()`` method
    that takes a full message history and returns a validated
    ``ParsedTurn``.

    JSON output is forced unconditionally via the ``response_format``
    parameter. This is required for Llama 3.3 70B and harmless on models
    that already comply with prompt-level JSON instructions, so we apply
    it uniformly for consistency across model families.

    Args:
        token: Groq API key with inference permissions.
        model_id: Model to call. Defaults to ``MODEL_ID`` from config.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens per response.

    Raises:
        ValueError: If ``token`` is empty.
    """

    def __init__(
        self,
        token: str,
        model_id: str = MODEL_ID,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        if not token:
            raise ValueError("token must be a non-empty string")
        self._client = Groq(api_key=token)
        self._model_id = model_id
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model_id(self) -> str:
        """The model ID this client is configured to use."""
        return self._model_id

    def generate(self, messages: Sequence[ChatMessage]) -> ParsedTurn:
        """Send a chat completion request and return a parsed turn.

        Args:
            messages: Full chat history including the system message as the
                first entry. Each message is a dict with ``role`` and
                ``content`` keys (OpenAI format).

        Returns:
            A ``ParsedTurn`` with the response text and validated
            evaluation scores.

        Raises:
            ValueError: If ``messages`` is empty.
            InferenceError: If the API call itself fails (auth, rate limit,
                network, malformed response envelope).
            EvaluationParseError: If the model's content cannot be parsed
                or validated against the expected attribute schema.
        """
        if not messages:
            raise ValueError("messages must not be empty")

        try:
            completion = self._client.chat.completions.create(
                model=self._model_id,
                messages=list(messages),
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )
        except APIError as exc:
            raise InferenceError(
                f"Groq Inference API request failed: {exc}"
            ) from exc
        except Exception as exc:
            raise InferenceError(
                f"Unexpected error during inference call: {exc}"
            ) from exc

        try:
            raw = completion.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise InferenceError(
                f"Inference response missing expected fields: {exc}"
            ) from exc

        if not isinstance(raw, str) or not raw.strip():
            raise InferenceError(
                "Inference response content is empty or non-text"
            )

        return parse_model_output(raw)
