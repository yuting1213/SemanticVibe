"""Settings + cost-mode tests. These exercise pydantic-settings without touching .env."""

from __future__ import annotations

import pytest

from semanticvibe.config import COST_MODES, STYLE_PRESETS, Settings, reset_settings_for_test


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch, tmp_path):
    # Make sure tests don't read a real .env from the repo root.
    monkeypatch.chdir(tmp_path)
    reset_settings_for_test()
    yield
    reset_settings_for_test()


def test_defaults_are_dev_claude(monkeypatch):
    for var in ("SEMANTICVIBE_LLM_PROVIDER", "SEMANTICVIBE_COST_MODE"):
        monkeypatch.delenv(var, raising=False)
    s = Settings()
    assert s.llm_provider == "claude"
    assert s.cost_mode == "dev"
    assert s.model_for() == "claude-haiku-4-5"


def test_prod_mode_picks_sonnet(monkeypatch):
    monkeypatch.setenv("SEMANTICVIBE_COST_MODE", "prod")
    s = Settings()
    assert s.model_for("claude") == "claude-sonnet-4-6"
    assert s.model_for("openai") == "gpt-4o"


def test_cost_modes_table_shape():
    assert set(COST_MODES.keys()) == {"dev", "prod"}
    for mode in ("dev", "prod"):
        assert {"claude", "openai"} <= set(COST_MODES[mode].keys())


def test_style_presets_have_palette_and_vibe():
    for name, preset in STYLE_PRESETS.items():
        assert preset["palette"], f"{name} missing palette"
        assert preset["vibe"], f"{name} missing vibe"
