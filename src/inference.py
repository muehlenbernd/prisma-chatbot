"""Groq Inference API client wrapper for Prisma.

Provides PrismaInferenceClient, a small wrapper around the groq SDK's
client that:
- Forces structured JSON output via Groq's json_schema strict mode.
  This constrains decoding to a predefined schema, making schema-
  non-conformant output structurally impossible (unlike json_object mode,
  which only guarantees syntactic JSON validity).
- Parses and validates the response via src.evaluation as a belt-and-
  suspenders check.
- Raises typed errors for API failures (InferenceError) and parse
  failures (EvaluationParseError, propagated from evaluation).
- Surfaces a user-friendly CapacityError when the project's Groq-side
  daily rate limit is hit (HTTP 429 with a daily-limit body).

Project-level Groq rate limits for openai/gpt-oss-120b (set in the Groq
console — adjust there if traffic patterns change, not here):
    Requests per Minute : 1,000
    Requests per Day    : 10,000
    Tokens per Minute   : 250,000
    Tokens per Day      : 3,000,000
At ~1,800 tokens/turn this covers roughly 1,650 turns/day, bounding
worst-case spend to ~$1.80/day.

The wrapper is initialized once per session with a Groq API key; each
generate() call sends a full message history (system + conversation)
and returns a validated ParsedTurn.
"""

from __future__ import annotations

from typing import Any, Sequence

from groq import APIError, APIStatusError, Groq

from .config import (
    DEFAULT_ATTRIBUTES,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    MODEL_ID,
)
from .evaluation import EvaluationParseError, ParsedTurn, parse_model_output

# Single chat message in OpenAI format.
ChatMessage = dict[str, str]

# HTTP status code Groq returns when a project-level rate limit is hit.
_GROQ_RATE_LIMIT_STATUS = 429


class InferenceError(Exception):
    """Raised when the inference API call fails or returns malformed data.

    Wraps network errors, authentication failures, non-capacity rate-limit
    errors, and missing-field errors in the API response. Parse errors on the
    model's content are *not* wrapped here — they surface as
    EvaluationParseError so the app layer can distinguish them.
    """


class CapacityError(Exception):
    """Raised when the project's Groq daily rate limit (429) is hit.

    Distinct from InferenceError so the app layer can show a specific
    'demo at capacity' message rather than a generic error.
    """


def _build_response_schema(attributes: list[str]) -> dict[str, Any]:
    """Build the Groq strict-mode json_schema response_format dict.

    The schema mirrors what parse_model_output validates: a root object
    with a string 'response' field and an 'evaluation' object mapping each
    attribute name to an integer.  Every object has additionalProperties
    false and every property is listed in required, as required by strict mode.

    Args:
        attributes: Ordered list of evaluation attribute names, sourced from
            DEFAULT_ATTRIBUTES in config.

    Returns:
        A response_format dict ready to pass to the Groq completions API.
    """
    evaluation_properties: dict[str, Any] = {
        attr: {"type": "integer"} for attr in attributes
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "prisma_evaluation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "response": {"type": "string"},
                    "evaluation": {
                        "type": "object",
                        "properties": evaluation_properties,
                        "required": list(attributes),
                        "additionalProperties": False,
                    },
                },
                "required": ["response", "evaluation"],
                "additionalProperties": False,
            },
        },
    }


class PrismaInferenceClient:
    """Wrapper around the Groq client configured for Prisma.

    Holds a single ``Groq`` instance and exposes a ``generate()`` method
    that takes a full message history and returns a validated ``ParsedTurn``.

    JSON output is constrained via Groq's strict json_schema mode, which
    prevents schema-non-conformant output at the generation level.
    parse_model_output is retained as a belt-and-suspenders validator.

    Retries are limited to one defensive retry on transport-level
    InferenceErrors (timeouts, 5xx errors).  EvaluationParseError is not
    retried — with strict mode active, parse failures indicate something
    unexpected and surfacing them immediately is the right call.

    Args:
        token: Groq API key with inference permissions.
        model_id: Model to call. Defaults to ``MODEL_ID`` from config.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens per response.
        max_attempts: Max calls per turn before surfacing an error. ``1``
            disables retry. Applies to transport-level failures only.

    Raises:
        ValueError: If ``token`` is empty or ``max_attempts < 1``.
    """

    def __init__(
        self,
        token: str,
        model_id: str = MODEL_ID,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        if not token:
            raise ValueError("token must be a non-empty string")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._client = Groq(api_key=token)
        self._model_id = model_id
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_attempts = max_attempts
        # Build and cache the response_format dict once; it only depends on
        # the (fixed) attribute list and never changes at runtime.
        self._response_format = _build_response_schema(DEFAULT_ATTRIBUTES)

    @property
    def model_id(self) -> str:
        """The model ID this client is configured to use."""
        return self._model_id

    def generate(self, messages: Sequence[ChatMessage]) -> ParsedTurn:
        """Send a chat completion request and return a parsed turn.

        Retries up to ``self._max_attempts`` times on transport-level
        ``InferenceError`` (timeouts, 5xx).  ``EvaluationParseError`` and
        ``CapacityError`` are not retried and propagate immediately.

        Args:
            messages: Full chat history including the system message as the
                first entry. Each message is a dict with ``role`` and
                ``content`` keys (OpenAI format).

        Returns:
            A ``ParsedTurn`` with the response text and validated
            evaluation scores.

        Raises:
            ValueError: If ``messages`` is empty.
            CapacityError: If Groq returns a 429 indicating the project's
                daily rate limit has been reached.
            InferenceError: If all ``max_attempts`` calls fail due to
                transport errors (auth, 5xx, network, malformed envelope).
            EvaluationParseError: If the response fails schema validation
                despite strict mode. Propagates immediately without retry.
        """
        if not messages:
            raise ValueError("messages must not be empty")

        last_exc: InferenceError | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._call_once(messages)
            except CapacityError:
                # Daily cap hit — surface immediately, no retry.
                raise
            except EvaluationParseError:
                # Strict mode makes this unexpected; surface immediately.
                raise
            except InferenceError as exc:
                last_exc = exc
                print(
                    f"[retry] attempt {attempt}/{self._max_attempts} "
                    f"transport error: {exc}"
                )

        assert last_exc is not None
        raise last_exc

    def _call_once(self, messages: Sequence[ChatMessage]) -> ParsedTurn:
        """One round-trip: API call, envelope check, parse. No retry."""
        try:
            completion = self._client.chat.completions.create(
                model=self._model_id,
                messages=list(messages),
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                response_format=self._response_format,
            )
        except APIStatusError as exc:
            if exc.status_code == _GROQ_RATE_LIMIT_STATUS:
                raise CapacityError(
                    "Groq project daily rate limit reached"
                ) from exc
            raise InferenceError(
                f"Groq API error (HTTP {exc.status_code}): {exc}"
            ) from exc
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
