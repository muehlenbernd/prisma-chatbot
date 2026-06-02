"""Unit tests for src.inference."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.config import DEFAULT_MAX_ATTEMPTS
from src.evaluation import EvaluationParseError, ParsedTurn
from src.inference import InferenceError, PrismaInferenceClient


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


# ---- Construction ----

def test_rejects_empty_token():
    with pytest.raises(ValueError, match="token"):
        PrismaInferenceClient(token="")


def test_exposes_model_id():
    client = PrismaInferenceClient(token="groq_test", model_id="some/model")
    assert client.model_id == "some/model"


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


def test_generate_forces_json_response_format():
    """The wrapper must always pass response_format={'type': 'json_object'}."""
    client = PrismaInferenceClient(token="groq_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.return_value = _mock_completion(
            VALID_PAYLOAD
        )
        client.generate([{"role": "user", "content": "hi"}])
    call = mock_inner.chat.completions.create.call_args
    assert call.kwargs["response_format"] == {"type": "json_object"}


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


def test_generate_propagates_parse_errors_after_all_attempts():
    """If every attempt fails to parse, the error propagates and the API was
    called exactly max_attempts times (not more)."""
    client = PrismaInferenceClient(token="groq_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.return_value = _mock_completion(
            "not json"
        )
        with pytest.raises(EvaluationParseError):
            client.generate([{"role": "user", "content": "hi"}])
        assert (
            mock_inner.chat.completions.create.call_count
            == DEFAULT_MAX_ATTEMPTS
        )


# ---- generate(): retry on EvaluationParseError ----

def test_generate_default_max_attempts_is_five():
    """Production default: 5 attempts (1 initial + 4 retries)."""
    assert DEFAULT_MAX_ATTEMPTS == 5


def test_generate_retries_until_success():
    """If early samples fail and a later one succeeds, return that result and
    stop calling the API."""
    client = PrismaInferenceClient(token="groq_test")
    with patch.object(client, "_client") as mock_inner:
        # Fail first two, succeed on third; remaining sentinels must be unused.
        mock_inner.chat.completions.create.side_effect = [
            _mock_completion("not json"),
            _mock_completion("not json"),
            _mock_completion(VALID_PAYLOAD),
            _mock_completion("should-not-be-called"),
            _mock_completion("should-not-be-called"),
        ]
        result = client.generate([{"role": "user", "content": "hi"}])
    assert isinstance(result, ParsedTurn)
    assert result.response == "Hi there!"
    assert mock_inner.chat.completions.create.call_count == 3


def test_generate_respects_custom_max_attempts():
    """max_attempts=2 means at most two API calls before raising."""
    client = PrismaInferenceClient(token="groq_test", max_attempts=2)
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.return_value = _mock_completion(
            "not json"
        )
        with pytest.raises(EvaluationParseError):
            client.generate([{"role": "user", "content": "hi"}])
        assert mock_inner.chat.completions.create.call_count == 2


def test_construction_rejects_zero_max_attempts():
    with pytest.raises(ValueError, match="max_attempts"):
        PrismaInferenceClient(token="groq_test", max_attempts=0)


def test_generate_does_not_retry_on_inference_error():
    """InferenceError must NOT trigger a retry — only EvaluationParseError does."""
    client = PrismaInferenceClient(token="groq_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat.completions.create.side_effect = RuntimeError("boom")
        with pytest.raises(InferenceError, match="boom"):
            client.generate([{"role": "user", "content": "hi"}])
        assert mock_inner.chat.completions.create.call_count == 1
