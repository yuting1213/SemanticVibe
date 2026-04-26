"""Stage 2 entry point: FeatureSummary → Decision with retries on validation failure."""

from __future__ import annotations

from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from semanticvibe.config import LLMProvider
from semanticvibe.llm.client import get_client
from semanticvibe.schemas.decision import Decision
from semanticvibe.schemas.feature_summary import FeatureSummary


@retry(
    retry=retry_if_exception_type(ValidationError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def decide(
    summary: FeatureSummary,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
) -> Decision:
    """Drive the LLM and parse the response into a `Decision`.

    Retries on Pydantic ValidationError only — network errors are handled
    inside each provider client where vendor-specific retry semantics apply.
    """
    client = get_client(provider)
    return client.decide(summary, model=model)
