"""Stage 1 orchestrator: video → FeatureSummary."""

from __future__ import annotations

from pathlib import Path

from semanticvibe.schemas.feature_summary import FeatureSummary


def extract_features(video_path: Path, *, style_preset: str) -> FeatureSummary:
    """Run all Stage 1 sub-steps and assemble a FeatureSummary.

    Wiring lands in Week 2 once the per-model files are implemented.
    """
    raise NotImplementedError(
        "Stage 1 orchestrator wires up in Week 2 (whisper + librosa + mediapipe "
        "+ keyframes + BLIP-2)."
    )
