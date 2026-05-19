"""Unit tests for src.inference."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

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
    """Build a MagicMock mimicking the HF chat_completion return shape."""
    completion = MagicMock()
    completion.choices = [MagicMock()]
    completion.choices[0].message.content = content
    return completion


# ---- Construction ----

def test_rejects_empty_token():
    with pytest.raises(ValueError, match="token"):
        PrismaInferenceClient(token="")


def test_exposes_model_id():
    client = PrismaInferenceClient(token="hf_test", model_id="some/model")
    assert client.model_id == "some/model"


# ---- generate(): happy paths ----

def test_generate_returns_parsed_turn():
    client = PrismaInferenceClient(token="hf_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat_completion.return_value = _mock_completion(VALID_PAYLOAD)
        result = client.generate([{"role": "user", "content": "hi"}])
    assert isinstance(result, ParsedTurn)
    assert result.response == "Hi there!"
    assert result.evaluation["competent"] == 5


def test_generate_forces_json_response_format():
    """The wrapper must always pass response_format={'type': 'json_object'}."""
    client = PrismaInferenceClient(token="hf_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat_completion.return_value = _mock_completion(VALID_PAYLOAD)
        client.generate([{"role": "user", "content": "hi"}])
    call = mock_inner.chat_completion.call_args
    assert call.kwargs["response_format"] == {"type": "json_object"}


def test_generate_passes_messages_and_model():
    client = PrismaInferenceClient(token="hf_test", model_id="custom/model")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat_completion.return_value = _mock_completion(VALID_PAYLOAD)
        client.generate(messages)
    call = mock_inner.chat_completion.call_args
    assert call.kwargs["model"] == "custom/model"
    assert call.kwargs["messages"] == messages


# ---- generate(): error paths ----

def test_generate_rejects_empty_messages():
    client = PrismaInferenceClient(token="hf_test")
    with pytest.raises(ValueError, match="messages"):
        client.generate([])


def test_generate_wraps_unexpected_exception():
    client = PrismaInferenceClient(token="hf_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat_completion.side_effect = RuntimeError("boom")
        with pytest.raises(InferenceError, match="boom"):
            client.generate([{"role": "user", "content": "hi"}])


def test_generate_rejects_empty_content():
    client = PrismaInferenceClient(token="hf_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat_completion.return_value = _mock_completion("")
        with pytest.raises(InferenceError, match="empty"):
            client.generate([{"role": "user", "content": "hi"}])


def test_generate_rejects_missing_choices():
    client = PrismaInferenceClient(token="hf_test")
    with patch.object(client, "_client") as mock_inner:
        bad = MagicMock()
        bad.choices = []
        mock_inner.chat_completion.return_value = bad
        with pytest.raises(InferenceError, match="missing expected fields"):
            client.generate([{"role": "user", "content": "hi"}])


def test_generate_propagates_parse_errors():
    """Parse failures bubble up as EvaluationParseError, not InferenceError."""
    client = PrismaInferenceClient(token="hf_test")
    with patch.object(client, "_client") as mock_inner:
        mock_inner.chat_completion.return_value = _mock_completion("not json")
        with pytest.raises(EvaluationParseError):
            client.generate([{"role": "user", "content": "hi"}])