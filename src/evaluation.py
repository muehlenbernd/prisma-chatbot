"""Score parsing, validation, and display formatting for Prisma evaluations.

This module owns the evaluation side of the dual-role output:
- Parsing the model's raw JSON response into a structured turn
- Validating that all expected attributes are present with in-range integer scores
- Formatting scores as 'intensifier attribute (score/MAX)' for display

The intensifier scale runs from 1 (not at all) to 7 (extremely), pairing a
verbal label with the numeric score. Example: 'barely polite (2/7)'.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .config import DEFAULT_ATTRIBUTES, MAX_SCORE, MIN_SCORE


INTENSIFIER_SCALE: dict[int, str] = {
    1: "not at all",
    2: "barely",
    3: "somewhat",
    4: "moderately",
    5: "quite",
    6: "very",
    7: "extremely",
}


class EvaluationParseError(Exception):
    """Raised when the model's evaluation output cannot be parsed or validated."""


@dataclass(frozen=True)
class ParsedTurn:
    """A single parsed turn from the model.

    Attributes:
        response: Prisma's conversational reply.
        evaluation: Mapping from attribute name to integer score in
            ``[MIN_SCORE, MAX_SCORE]``.
    """

    response: str
    evaluation: dict[str, int]


def parse_model_output(
    raw_output: str,
    expected_attributes: list[str] | None = None,
) -> ParsedTurn:
    """Parse the model's raw JSON output into a validated ``ParsedTurn``.

    Tolerates markdown code fences around the JSON object, since some models
    wrap structured output in ``` blocks despite instructions otherwise.

    Args:
        raw_output: The raw text returned by the model.
        expected_attributes: Attribute names that must be present in the
            evaluation block. Defaults to ``DEFAULT_ATTRIBUTES``. Extra
            attributes returned by the model are silently ignored.

    Returns:
        A validated ``ParsedTurn`` containing the response and evaluation.

    Raises:
        EvaluationParseError: If JSON cannot be parsed, required fields are
            missing or malformed, or any score is out of range.
    """
    if expected_attributes is None:
        expected_attributes = list(DEFAULT_ATTRIBUTES)

    cleaned = _strip_json_fences(raw_output)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise EvaluationParseError(
            f"Could not parse JSON from model output: {exc}"
        ) from exc

    response = payload.get("response")
    if not isinstance(response, str):
        raise EvaluationParseError(
            "Missing or non-string 'response' field in model output."
        )

    evaluation_raw = payload.get("evaluation")
    if not isinstance(evaluation_raw, dict):
        raise EvaluationParseError(
            "Missing or non-object 'evaluation' field in model output."
        )

    evaluation = _validate_evaluation(evaluation_raw, expected_attributes)
    return ParsedTurn(response=response, evaluation=evaluation)


def format_score(attribute: str, score: int) -> str:
    """Format a single attribute score as 'intensifier attribute (score/MAX)'.

    Args:
        attribute: The attribute name (e.g. ``'polite'``).
        score: Integer score from ``MIN_SCORE`` to ``MAX_SCORE``.

    Returns:
        Formatted display string, e.g. ``'barely polite (2/7)'``.

    Raises:
        ValueError: If ``score`` is outside the valid intensifier range.
    """
    if score not in INTENSIFIER_SCALE:
        raise ValueError(
            f"Score {score} not in intensifier scale "
            f"(valid: {sorted(INTENSIFIER_SCALE)})"
        )
    return f"{INTENSIFIER_SCALE[score]} {attribute} ({score}/{MAX_SCORE})"


def format_evaluation(
    evaluation: dict[str, int],
    attributes: list[str] | None = None,
) -> list[str]:
    """Format an evaluation dict as an ordered list of display strings.

    Args:
        evaluation: Mapping from attribute name to score.
        attributes: Render order for the attributes. Defaults to
            ``DEFAULT_ATTRIBUTES``.

    Returns:
        Formatted strings in the given order.

    Raises:
        KeyError: If ``evaluation`` is missing one of the requested attributes.
    """
    if attributes is None:
        attributes = list(DEFAULT_ATTRIBUTES)
    return [format_score(attr, evaluation[attr]) for attr in attributes]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _strip_json_fences(text: str) -> str:
    """Remove markdown code fences from around a JSON payload, if present."""
    text = text.strip()
    match = _FENCE_PATTERN.match(text)
    if match:
        return match.group(1).strip()
    return text


def _validate_evaluation(
    raw_evaluation: dict[str, Any],
    expected_attributes: list[str],
) -> dict[str, int]:
    """Check presence, type, and range for each expected attribute score."""
    validated: dict[str, int] = {}
    for attr in expected_attributes:
        if attr not in raw_evaluation:
            raise EvaluationParseError(
                f"Missing attribute '{attr}' in evaluation."
            )
        score = raw_evaluation[attr]
        # bool is a subclass of int in Python; reject it explicitly.
        if isinstance(score, bool) or not isinstance(score, int):
            raise EvaluationParseError(
                f"Score for '{attr}' is not an integer: {score!r}"
            )
        if score < MIN_SCORE or score > MAX_SCORE:
            raise EvaluationParseError(
                f"Score for '{attr}' out of range "
                f"[{MIN_SCORE}-{MAX_SCORE}]: {score}"
            )
        validated[attr] = score
    return validated
