"""LLM client abstraction.

Both providers must yield a `Decision` from a `FeatureSummary`. Claude uses
tool-use to force JSON-schema conformance; OpenAI uses native
`response_format=json_schema`. Both routes drop the LLM JSON-parse failure
rate well below 5% (scaffolding design §6).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from semanticvibe.config import LLMProvider, get_settings
from semanticvibe.schemas.decision import Decision
from semanticvibe.schemas.feature_summary import FeatureSummary


@runtime_checkable
class LLMClient(Protocol):
    """Stage 2 interface. Implementations must be substitutable for A/B work."""

    provider_name: str

    def decide(self, summary: FeatureSummary, *, model: str | None = None) -> Decision: ...


class ClaudeClient:
    """Anthropic implementation. Uses tool-use for schema-locked JSON output
    and prompt caching for the system prompt + few-shots prefix.
    """

    provider_name = "claude"

    def __init__(self, api_key: str | None = None) -> None:
        # Lazy import — `anthropic` is a heavy install, don't pay for it on
        # users who only run the OpenAI path.
        from anthropic import Anthropic

        settings = get_settings()
        self._client = Anthropic(api_key=api_key or settings.anthropic_api_key)
        self._default_model = settings.model_for("claude")

    def decide(self, summary: FeatureSummary, *, model: str | None = None) -> Decision:
        raise NotImplementedError(
            "ClaudeClient.decide is wired in Week 3. The LLMClient protocol and "
            "construction path are stable; the body lands once Stage 2 work begins."
        )


class OpenAIClient:
    """OpenAI implementation. Uses `response_format=json_schema`."""

    provider_name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        from openai import OpenAI

        settings = get_settings()
        self._client = OpenAI(api_key=api_key or settings.openai_api_key)
        self._default_model = settings.model_for("openai")

    def decide(self, summary: FeatureSummary, *, model: str | None = None) -> Decision:
        raise NotImplementedError(
            "OpenAIClient.decide is wired in Week 3 alongside ClaudeClient — "
            "spec §3.3 contribution 1 (A/B comparison) needs both paths."
        )


_REGISTRY: dict[LLMProvider, type[LLMClient]] = {
    "claude": ClaudeClient,
    "openai": OpenAIClient,
}


def get_client(provider: LLMProvider | None = None) -> LLMClient:
    settings = get_settings()
    return _REGISTRY[provider or settings.llm_provider]()
