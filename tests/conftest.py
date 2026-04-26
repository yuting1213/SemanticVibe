"""Shared pytest fixtures.

Tests should NOT touch the network and should NOT require GPU. Anything
heavier (real video render, real LLM call) belongs behind a
@pytest.mark.integration marker we add when those tests come online.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


@pytest.fixture
def hand_written_decision_dict() -> dict:
    with (EXAMPLES_DIR / "hand_written_decision.json").open(encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT
