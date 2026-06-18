"""Unit tests for src.inference."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.config import DEFAULT_ATTRIBUTES, DEFAULT_MAX_ATTEMPTS
from src.evaluation import EvaluationParseError, ParsedTurn
from src.inference import (
    CapacityError,
    InferenceError,
    PrismaInferenceClient,
    _build_response_schema,
)


VALID_PAYLOAD = json.dumps({
    "response": "Hi there!",
    "evaluation": {
        "competent": 5,
        "likeable": 5,
        "considerate": 5,
        "polite": 5,
        "formal": 5,
        "demanding": 3,
    },
})


def _mock_completion(content: str) -> MagicMock:
    """Build a MagicMock mimicking the Groq chat completion return shape."""
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = content
    return completion


def _mock_status_error(status_code: int) -> MagicMock:
    """Build a MagicMock mimicking a groq.APIStatusError."""
    from groq import APIStatusError
    response = MagicMock()
    response.status_code = status_code
    return APIStatusError(
        message=f"HTTP {status_code}",
        response=response,
        body={},
    )


# ---- _build_response_schema ----

def test_build_response_schema_top_level_shape():
    schema = _build_response_schema(DEFAULT_ATTRIBUTES)
    assert schema["type"] == "json_schema"
    js = schema["json_schema"]
    assert js["name"] == "prisma_evaluation"
    assert js["strict"] is True


def test_build_response_schema_root_object_constraints():
    schema = _build_response_schema(DEFAULT_ATTRIBUTES)
    root = schema["json_schema"]["schema"]
    assert root["type"] == "object"
    assert root["additionalProperties"] is False
    assert "response" in root["required"]
    assert "evaluation" in root["required"]


def test_build_response_schema_includes_all_attributes():
    schema = _build_response_schema(DEFAULT_ATTRIBUTES)
    evaluation = schema["json_schema"]["schema"]["properties"]["evaluation"]
    assert evaluation["additionalProperties"] is False
    for attr in DEFAULT_ATTRIBUTES:
        assert attr in evaluation["properties"]
        assert attr in evaluation["required"]


def test_build_response_schema_attribute_types_are_integer():
    schema = _build_response_schema(DEFAULT_ATTRIBUTES)
    props = schema["json_schema"]["schema"]["properties"]["evaluation"]["properties"]
    for attr in DEFAULT_ATTRIBUTES:
        assert props[attr]["type"] == "integer"


def test_build_response_schema_custom_attributes():
    custom = ["a", "b", "c"]
    schema = _build_response_schema(custom)
    evaluation = schema["json_schema"]["schema"]["properties"]["evaluation"]
    assert set(evaluation["required"]) == set(custom)


# ---- Construction ----

def test_rejects_empty_token():
    with pytest.raises(ValueError, match="token"):
        PrismaInferenceClient(token="")


def test_exposes_model_id():
    client = PrismaInferenceClient(token="groq_test", model_id="some/model")
    assert client.model_id == "some/model"


def test_construction_rejects_zero_max_attempts():
    with pytest.raises(ValueError, match="max_attempts"):
        PrismaInferenceClient(token="groq_test", max_attempts=0)


# ---- generate(): happy paths ----

def test_generate_returns_parsed_turn():
    client = PrismaInferenceClient(token="groq_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.return_value = _mock_completion(
            VALID_PAYLOAD
        )
        result = client.generate([{"role": "user", "content": "hi"}])
    assert isinstance(result, ParsedTurn)
    assert result.response == "Hi there!"
    assert result.evaluation["competent"] == 5


def test_generate_uses_json_schema_strict_response_format():
    """The wrapper must pass a json_schema strict response_format."""
    client = PrismaInferenceClient(token="groq_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.return_value = _mock_completion(
            VALID_PAYLOAD
        )
        client.generate([{"role": "user", "content": "hi"}])
    call = mock_inner.chat.completions.create.call_args
    rf = call.kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "prisma_evaluation"


def test_generate_passes_messages_and_model():
    client = PrismaInferenceClient(token="groq_test", model_id="custom/model")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.return_value = _mock_completion(
            VALID_PAYLOAD
        )
        client.generate(messages)
    call = mock_inner.chat.completions.create.call_args
    assert call.kwargs["model"] == "custom/model"
    assert call.kwargs["messages"] == messages


# ---- generate(): error paths ----

def test_generate_rejects_empty_messages():
    client = PrismaInferenceClient(token="groq_test")
    with pytest.raises(ValueError, match="messages"):
        client.generate([])


def test_generate_wraps_unexpected_exception():
    client = PrismaInferenceClient(token="groq_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.side_effect = RuntimeError("boom")
        with pytest.raises(InferenceError, match="boom"):
            client.generate([{"role": "user", "content": "hi"}])


def test_generate_rejects_empty_content():
    client = PrismaInferenceClient(token="groq_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.return_value = _mock_completion("")
        with pytest.raises(InferenceError, match="empty"):
            client.generate([{"role": "user", "content": "hi"}])


def test_generate_rejects_missing_choices():
    client = PrismaInferenceClient(token="groq_test")
    with patch.object(client, "_client") as mock_inner:
        bad = MagicMock()
        bad.choices = []
        mock_inner.chat.completions.create.return_value = bad
        with pytest.raises(InferenceError, match="missing expected fields"):
            client.generate([{"role": "user", "content": "hi"}])


# ---- generate(): EvaluationParseError propagates immediately (no retry) ----

def test_generate_propagates_parse_error_immediately():
    """EvaluationParseError must propagate on the first attempt — no retry."""
    client = PrismaInferenceClient(token="groq_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.return_value = _mock_completion(
            "not json"
        )
        with pytest.raises(EvaluationParseError):
            client.generate([{"role": "user", "content": "hi"}])
        # Only one API call — parse errors are not retried.
        assert mock_inner.chat.completions.create.call_count == 1


# ---- generate(): retry on InferenceError (transport failures) ----

def test_generate_default_max_attempts_is_two():
    """Production default: 2 attempts (1 initial + 1 transport retry)."""
    assert DEFAULT_MAX_ATTEMPTS == 2


def test_generate_retries_transport_error_once_then_succeeds():
    """A single transport InferenceError is retried; success on second call."""
    client = PrismaInferenceClient(token="groq_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.side_effect = [
            RuntimeError("timeout"),
            _mock_completion(VALID_PAYLOAD),
        ]
        result = client.generate([{"role": "user", "content": "hi"}])
    assert isinstance(result, ParsedTurn)
    assert mock_inner.chat.completions.create.call_count == 2


def test_generate_propagates_inference_error_after_all_attempts():
    """If every transport attempt fails, InferenceError propagates."""
    client = PrismaInferenceClient(token="groq_test", max_attempts=2)
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.side_effect = RuntimeError("boom")
        with pytest.raises(InferenceError):
            client.generate([{"role": "user", "content": "hi"}])
        assert mock_inner.chat.completions.create.call_count == 2


def test_generate_respects_custom_max_attempts():
    """max_attempts=3 means at most three API calls before raising."""
    client = PrismaInferenceClient(token="groq_test", max_attempts=3)
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.side_effect = RuntimeError("boom")
        with pytest.raises(InferenceError):
            client.generate([{"role": "user", "content": "hi"}])
        assert mock_inner.chat.completions.create.call_count == 3


# ---- generate(): CapacityError (429) ----

def test_generate_raises_capacity_error_on_429():
    """A 429 from Groq must surface as CapacityError, not InferenceError."""
    client = PrismaInferenceClient(token="groq_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.side_effect = _mock_status_error(429)
        with pytest.raises(CapacityError):
            client.generate([{"role": "user", "content": "hi"}])


def test_generate_does_not_retry_capacity_error():
    """CapacityError (daily limit) must NOT be retried."""
    client = PrismaInferenceClient(token="groq_test", max_attempts=2)
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.side_effect = _mock_status_error(429)
        with pytest.raises(CapacityError):
            client.generate([{"role": "user", "content": "hi"}])
        assert mock_inner.chat.completions.create.call_count == 1


def test_generate_wraps_non_429_status_error_as_inference_error():
    """Non-429 HTTP errors are wrapped as InferenceError."""
    client = PrismaInferenceClient(token="groq_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.side_effect = _mock_status_error(500)
        with pytest.raises(InferenceError):
            client.generate([{"role": "user", "content": "hi"}])
