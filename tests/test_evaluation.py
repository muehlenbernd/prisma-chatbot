"""Unit tests for src.evaluation."""

from __future__ import annotations

import json

import pytest

from src.evaluation import (
    EvaluationParseError,
    ParsedTurn,
    format_evaluation,
    format_score,
    parse_model_output,
)


VALID_EVAL = {
    "competent": 5,
    "likeable": 4,
    "considerate": 6,
    "polite": 7,
    "formal": 3,
    "demanding": 2,
}


def _valid_payload(**overrides) -> str:
    """Build a JSON string representing a well-formed model output."""
    payload = {"response": "Hi there!", "evaluation": dict(VALID_EVAL)}
    payload.update(overrides)
    return json.dumps(payload)


# ---- format_score ----

def test_format_score_basic():
    assert format_score("polite", 2) == "barely polite (2/7)"
    assert format_score("competent", 5) == "quite competent (5/7)"
    assert format_score("demanding", 7) == "extremely demanding (7/7)"


def test_format_score_all_levels_render():
    for score in range(1, 8):
        result = format_score("attr", score)
        assert f"({score}/7)" in result


@pytest.mark.parametrize("bad_score", [0, 8, -1, 100])
def test_format_score_rejects_out_of_range(bad_score: int):
    with pytest.raises(ValueError):
        format_score("polite", bad_score)


# ---- format_evaluation ----

def test_format_evaluation_returns_ordered_list():
    result = format_evaluation(VALID_EVAL)
    assert len(result) == 6
    assert result[0].endswith("(5/7)")  # competent first by default


def test_format_evaluation_custom_order():
    result = format_evaluation(VALID_EVAL, attributes=["polite", "formal"])
    assert len(result) == 2
    assert "polite" in result[0]
    assert "formal" in result[1]


# ---- parse_model_output: happy paths ----

def test_parse_model_output_returns_parsed_turn():
    turn = parse_model_output(_valid_payload())
    assert isinstance(turn, ParsedTurn)
    assert turn.response == "Hi there!"
    assert turn.evaluation == VALID_EVAL


def test_parse_model_output_strips_json_fences():
    raw = f"```json\n{_valid_payload()}\n```"
    turn = parse_model_output(raw)
    assert turn.evaluation == VALID_EVAL


def test_parse_model_output_strips_plain_fences():
    raw = f"```\n{_valid_payload()}\n```"
    turn = parse_model_output(raw)
    assert turn.evaluation == VALID_EVAL


def test_parse_model_output_ignores_extra_attributes():
    extra = dict(VALID_EVAL)
    extra["unexpected"] = 5
    raw = json.dumps({"response": "hi", "evaluation": extra})
    turn = parse_model_output(raw)
    assert "unexpected" not in turn.evaluation


# ---- parse_model_output: error paths ----

def test_parse_model_output_rejects_invalid_json():
    with pytest.raises(EvaluationParseError, match="parse JSON"):
        parse_model_output("not even close to json")


def test_parse_model_output_rejects_missing_response():
    raw = json.dumps({"evaluation": VALID_EVAL})
    with pytest.raises(EvaluationParseError, match="response"):
        parse_model_output(raw)


def test_parse_model_output_rejects_missing_evaluation():
    raw = json.dumps({"response": "hi"})
    with pytest.raises(EvaluationParseError, match="evaluation"):
        parse_model_output(raw)


def test_parse_model_output_rejects_missing_attribute():
    incomplete = dict(VALID_EVAL)
    del incomplete["polite"]
    raw = json.dumps({"response": "hi", "evaluation": incomplete})
    with pytest.raises(EvaluationParseError, match="polite"):
        parse_model_output(raw)


def test_parse_model_output_rejects_out_of_range_score():
    bad = dict(VALID_EVAL)
    bad["polite"] = 9
    raw = json.dumps({"response": "hi", "evaluation": bad})
    with pytest.raises(EvaluationParseError, match="out of range"):
        parse_model_output(raw)


def test_parse_model_output_rejects_bool_score():
    bad = dict(VALID_EVAL)
    bad["polite"] = True  # noqa: not an int despite isinstance(True, int)
    raw = json.dumps({"response": "hi", "evaluation": bad})
    with pytest.raises(EvaluationParseError, match="not an integer"):
        parse_model_output(raw)


def test_parse_model_output_rejects_float_score():
    bad = dict(VALID_EVAL)
    bad["polite"] = 5.5
    raw = json.dumps({"response": "hi", "evaluation": bad})
    with pytest.raises(EvaluationParseError, match="not an integer"):
        parse_model_output(raw)