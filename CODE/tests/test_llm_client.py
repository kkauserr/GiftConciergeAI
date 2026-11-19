"""LLMClient request/response handling against a mocked SDK client (no API calls)."""

from types import SimpleNamespace

import pytest

from llm.llm_client import LLMClient, LLMConfigError, LLMResponseError


class MockCompletions:
    def __init__(self, content, refusal=None):
        self.content, self.refusal, self.last = content, refusal, None

    def create(self, **params):
        self.last = params
        message = SimpleNamespace(content=self.content, refusal=self.refusal)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def make_client(content, refusal=None):
    completions = MockCompletions(content, refusal)
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return LLMClient(None, None, "my-deployment", client=sdk), completions


def test_complete_json_sends_strict_schema_and_parses():
    client, mock = make_client('{"a": 1}')
    schema = {"type": "object", "properties": {"a": {"type": "integer"}},
              "required": ["a"], "additionalProperties": False}
    assert client.complete_json("hi", schema, "thing", system="sys") == {"a": 1}
    assert mock.last["model"] == "my-deployment"
    assert mock.last["messages"][0] == {"role": "system", "content": "sys"}
    fmt = mock.last["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"] == {"name": "thing", "schema": schema, "strict": True}


def test_complete_returns_text():
    client, mock = make_client("hello")
    assert client.complete([{"role": "user", "content": "x"}]) == "hello"
    assert "response_format" not in mock.last


def test_invalid_json_and_refusal_raise():
    client, _ = make_client("not json")
    with pytest.raises(LLMResponseError):
        client.complete_json("x", {}, "s")
    client, _ = make_client(None, refusal="no")
    with pytest.raises(LLMResponseError, match="refused"):
        client.complete("x")


def test_from_env_reports_missing_vars(monkeypatch, tmp_path):
    for name in ("FOUNDRY_ENDPOINT", "FOUNDRY_API_KEY", "FOUNDRY_DEPLOYMENT"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LLMConfigError, match="FOUNDRY_API_KEY"):
        LLMClient.from_env(env_file=tmp_path / "missing.env")
