"""LLM client abstraction.

Both providers must yield a `Decision` from a `FeatureSummary`. Claude uses
tool-use to force JSON-schema conformance; OpenAI uses native
`response_format=json_schema`. Both routes drop the LLM JSON-parse failure
rate well below 5%.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol, runtime_checkable

from semanticvibe.config import LLMProvider, get_settings
from semanticvibe.llm.prompts import (
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    build_user_message,
)
from semanticvibe.schemas.decision import Decision
from semanticvibe.schemas.feature_summary import FeatureSummary

log = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """Stage 2 interface. Implementations must be substitutable for A/B work."""

    provider_name: str

    def decide(self, summary: FeatureSummary, *, model: str | None = None) -> Decision: ...


def _decision_json_schema() -> dict[str, Any]:
    """Pydantic-derived JSON schema for `Decision`, ready to feed providers.

    Both Anthropic (tool input_schema) and OpenAI (response_format json_schema)
    accept the standard JSON-schema-with-$defs that Pydantic emits.
    """
    return Decision.model_json_schema()


class ClaudeClient:
    """Anthropic implementation. Uses tool-use for schema-locked JSON output
    and prompt caching for the system prompt + few-shots prefix.
    """

    provider_name = "claude"
    _TOOL_NAME = "emit_decision"

    def __init__(self, api_key: str | None = None) -> None:
        from anthropic import Anthropic

        settings = get_settings()
        key = api_key or settings.anthropic_api_key
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Put it in .env or pass api_key explicitly."
            )
        self._client = Anthropic(api_key=key)
        self._default_model = settings.model_for("claude")

    def _build_messages(self, summary: FeatureSummary) -> list[dict[str, Any]]:
        msgs: list[dict[str, Any]] = []
        # Few-shots: pre-recorded (FeatureSummary, Decision) pairs as
        # alternating user/assistant turns. Empty by default.
        for ex in FEW_SHOT_EXAMPLES:
            msgs.append(
                {"role": "user", "content": build_user_message(json.dumps(ex["summary"]))}
            )
            msgs.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": f"toolu_fewshot_{len(msgs)}",
                            "name": self._TOOL_NAME,
                            "input": ex["decision"],
                        }
                    ],
                }
            )
        msgs.append(
            {"role": "user", "content": build_user_message(summary.model_dump_json(indent=2))}
        )
        return msgs

    def decide(self, summary: FeatureSummary, *, model: str | None = None) -> Decision:
        tool = {
            "name": self._TOOL_NAME,
            "description": "Emit the final Decision describing all overlay elements.",
            "input_schema": _decision_json_schema(),
        }
        # cache_control on the system block enables Anthropic prompt caching
        # for the stable prefix (system prompt + tool definition + few-shots).
        system_blocks = [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        response = self._client.messages.create(
            model=model or self._default_model,
            max_tokens=4096,
            system=system_blocks,
            tools=[tool],
            tool_choice={"type": "tool", "name": self._TOOL_NAME},
            messages=self._build_messages(summary),
        )

        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == self._TOOL_NAME:
                payload = block.input
                if isinstance(payload, str):
                    payload = json.loads(payload)
                log.debug("Claude returned tool_use payload with %d elements", len(payload.get("elements", [])))
                return Decision.model_validate(payload)

        raise RuntimeError(
            f"Claude did not invoke the {self._TOOL_NAME} tool. "
            f"stop_reason={response.stop_reason}; content types="
            f"{[getattr(b, 'type', '?') for b in response.content]}"
        )


class OpenAIClient:
    """OpenAI implementation. Uses `response_format=json_schema`."""

    provider_name = "openai"
    _SCHEMA_NAME = "Decision"

    def __init__(self, api_key: str | None = None) -> None:
        from openai import OpenAI

        settings = get_settings()
        key = api_key or settings.openai_api_key
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Put it in .env or pass api_key explicitly."
            )
        self._client = OpenAI(api_key=key)
        self._default_model = settings.model_for("openai")

    def decide(self, summary: FeatureSummary, *, model: str | None = None) -> Decision:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex in FEW_SHOT_EXAMPLES:
            messages.append(
                {"role": "user", "content": build_user_message(json.dumps(ex["summary"]))}
            )
            messages.append(
                {"role": "assistant", "content": json.dumps(ex["decision"], ensure_ascii=False)}
            )
        messages.append(
            {"role": "user", "content": build_user_message(summary.model_dump_json(indent=2))}
        )

        # OpenAI's strict mode requires `additionalProperties: false` on every
        # object. Pydantic emits that by default for Decision; we leave the
        # schema as-is to avoid drift.
        response = self._client.chat.completions.create(
            model=model or self._default_model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": self._SCHEMA_NAME,
                    "schema": _decision_json_schema(),
                    "strict": False,  # nested $defs trip up strict mode
                },
            },
        )

        text = response.choices[0].message.content or ""
        log.debug("OpenAI returned %d chars", len(text))
        return Decision.model_validate_json(text)


_REGISTRY: dict[LLMProvider, type[LLMClient]] = {
    "claude": ClaudeClient,
    "openai": OpenAIClient,
}


def get_client(provider: LLMProvider | None = None) -> LLMClient:
    settings = get_settings()
    return _REGISTRY[provider or settings.llm_provider]()
