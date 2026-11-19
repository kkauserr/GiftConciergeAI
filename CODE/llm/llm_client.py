"""The single entry point for all LLM calls.

Provider: Microsoft Foundry through its Azure OpenAI-compatible endpoint, using
the `openai` package's AzureOpenAI client. To swap providers, implement another
class with the same two methods (complete, complete_json) and pass it in;
nothing else in CODE/llm talks to a provider directly.

Config comes from environment variables, loaded from the repo-root .env
(gitignored; see .env.example):

    FOUNDRY_ENDPOINT      https://<resource>.openai.azure.com/
    FOUNDRY_API_KEY       key
    FOUNDRY_DEPLOYMENT    deployment name (passed as `model`)
    FOUNDRY_API_VERSION   optional, default below; must support json_schema output
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_API_VERSION = "2024-10-21"
REQUIRED_ENV = ("FOUNDRY_ENDPOINT", "FOUNDRY_API_KEY", "FOUNDRY_DEPLOYMENT")


class LLMConfigError(RuntimeError):
    pass


class LLMResponseError(RuntimeError):
    pass


def _as_messages(prompt_or_messages, system=None):
    if isinstance(prompt_or_messages, str):
        messages = [{"role": "user", "content": prompt_or_messages}]
    else:
        messages = list(prompt_or_messages)
    if system:
        messages = [{"role": "system", "content": system}] + messages
    return messages


class LLMClient:
    def __init__(self, endpoint, api_key, deployment, api_version=DEFAULT_API_VERSION,
                 temperature=0.0, client=None):
        self.deployment = deployment
        self.temperature = temperature
        if client is None:
            from openai import AzureOpenAI
            client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key,
                                 api_version=api_version)
        self._client = client

    @classmethod
    def from_env(cls, env_file=REPO_ROOT / ".env", **kwargs):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file, override=False)
        except ImportError:  # dotenv is optional if vars are already exported
            pass
        missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise LLMConfigError(
                f"Missing {', '.join(missing)}. Copy .env.example to .env and fill it in.")
        return cls(
            endpoint=os.environ["FOUNDRY_ENDPOINT"],
            api_key=os.environ["FOUNDRY_API_KEY"],
            deployment=os.environ["FOUNDRY_DEPLOYMENT"],
            api_version=os.environ.get("FOUNDRY_API_VERSION") or DEFAULT_API_VERSION,
            **kwargs,
        )

    def _create(self, messages, **extra):
        params = {"model": self.deployment, "messages": messages, **extra}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        response = self._client.chat.completions.create(**params)
        choice = response.choices[0]
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise LLMResponseError(f"Model refused: {refusal}")
        return choice.message.content or ""

    def complete(self, prompt_or_messages, system=None):
        """Plain text completion. Accepts a prompt string or a messages list."""
        return self._create(_as_messages(prompt_or_messages, system))

    def complete_json(self, prompt_or_messages, schema, name, system=None):
        """Completion constrained to a JSON schema (strict structured output).

        `schema` must be strict-compatible: every property listed in
        `required`, `additionalProperties: false`, optional values as
        nullable types. Returns the parsed dict; callers validate semantics.
        """
        content = self._create(
            _as_messages(prompt_or_messages, system),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": True},
            },
        )
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Response was not valid JSON: {content[:200]!r}") from exc


class FakeLLMClient:
    """Offline stand-in with the same interface, for tests and the --offline demo.

    `responses` is a list consumed in order (dicts for complete_json, strings
    for complete), or a callable(kind, messages, schema_name) -> response.
    Every call is recorded in `self.calls`.
    """

    def __init__(self, responses=None):
        self._responses = responses if responses is not None else []
        self.calls = []

    def _next(self, kind, messages, schema_name=None):
        self.calls.append({"kind": kind, "messages": messages, "schema_name": schema_name})
        if callable(self._responses):
            return self._responses(kind, messages, schema_name)
        if not self._responses:
            raise LLMResponseError("FakeLLMClient has no scripted responses left")
        return self._responses.pop(0)

    def complete(self, prompt_or_messages, system=None):
        return self._next("text", _as_messages(prompt_or_messages, system))

    def complete_json(self, prompt_or_messages, schema, name, system=None):
        return self._next("json", _as_messages(prompt_or_messages, system), name)
