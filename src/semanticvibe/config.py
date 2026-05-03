"""Central config: settings (loaded from .env) + style presets + cost mode table."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Cost modes (spec §4 / scaffolding design §6)
# ---------------------------------------------------------------------------

CostMode = Literal["dev", "prod"]
LLMProvider = Literal["claude", "openai", "ollama"]

COST_MODES = {
    "dev":  {"claude": "claude-haiku-4-5",  "openai": "gpt-4o-mini", "ollama": "llama3"},
    "prod": {"claude": "claude-sonnet-4-6", "openai": "gpt-4o", "ollama": "llama3"}
}


# ---------------------------------------------------------------------------
# Style presets (spec §5.5 — colour palette + vibe descriptor)
# ---------------------------------------------------------------------------

STYLE_PRESETS: dict[str, dict] = {
    "warm_handdrawn": {
        "palette": ["#F4A261", "#E76F51", "#264653", "#2A9D8F", "#E9C46A"],
        "vibe": "warm hand-drawn picturebook, slightly off-kilter strokes, paper-textured background",
    },
    "soft_pastel": {
        "palette": ["#FFD6E0", "#C8E7FF", "#FFF1B6", "#D4F1C5", "#E5D4FF"],
        "vibe": "soft pastel, gentle gradients, dreamy and whimsical",
    },
    "bold_pop": {
        "palette": ["#FF006E", "#FFBE0B", "#3A86FF", "#8338EC", "#06FFA5"],
        "vibe": "bold pop-art, saturated primary colours, high contrast",
    },
    "monochrome_ink": {
        "palette": ["#1A1A1A", "#4D4D4D", "#808080", "#B3B3B3", "#F2F2F2"],
        "vibe": "monochrome ink wash, brush strokes, restrained and meditative",
    },
}


# ---------------------------------------------------------------------------
# Settings — read from .env via pydantic-settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="SEMANTICVIBE_",
        extra="ignore",
    )

    llm_provider: LLMProvider = "claude"
    cost_mode: CostMode = "dev"
    cost_ceiling_usd: float = Field(default=0.50, gt=0)

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")

    # API keys are read directly (no SEMANTICVIBE_ prefix — they belong to vendors).
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    def model_for(self, provider: LLMProvider | None = None) -> str:
        return COST_MODES[self.cost_mode][provider or self.llm_provider]


_settings: Settings | None = None


def get_settings() -> Settings:
    """Lazy singleton — avoids reading .env at import time during tests."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_for_test() -> None:
    """Tests can call this to force a re-read after monkeypatching env vars."""
    global _settings
    _settings = None
