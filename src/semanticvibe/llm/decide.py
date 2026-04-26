"""Stage 2 entry point: FeatureSummary → Decision with retries on validation failure."""

from __future__ import annotations

import logging

from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from semanticvibe.config import LLMProvider, get_settings
from semanticvibe.llm.client import get_client
from semanticvibe.llm.heuristic import heuristic_decision
from semanticvibe.schemas.decision import Decision
from semanticvibe.schemas.feature_summary import FeatureSummary

log = logging.getLogger(__name__)


@retry(
    retry=retry_if_exception_type(ValidationError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def _decide_via_llm(
    summary: FeatureSummary,
    *,
    provider: LLMProvider | None,
    model: str | None,
) -> Decision:
    client = get_client(provider)
    return client.decide(summary, model=model)


def decide(
    summary: FeatureSummary,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
    fallback_to_heuristic: bool = True,
) -> Decision:
    """Drive the LLM and parse the response into a `Decision`.

    If `fallback_to_heuristic` is True (default) and the LLM call cannot
    proceed because no API key is configured, fall back to the deterministic
    heuristic generator. Validation errors and network errors still raise.
    """
    settings = get_settings()
    chosen = provider or settings.llm_provider
    has_key = (
        (chosen == "claude" and settings.anthropic_api_key)
        or (chosen == "openai" and settings.openai_api_key)
    )
    if not has_key:
        if fallback_to_heuristic:
            log.warning(
                "No %s API key — falling back to heuristic Decision generator.", chosen
            )
            return heuristic_decision(summary)
        raise RuntimeError(f"No API key configured for provider {chosen!r}")

    return _decide_via_llm(summary, provider=provider, model=model)
