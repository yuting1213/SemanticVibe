"""Top-level orchestrator: video → mp4 across all 5 stages."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from semanticvibe.config import LLMProvider, get_settings
from semanticvibe.layout.placement import resolve_anchors
from semanticvibe.llm.decide import decide
from semanticvibe.preprocess.pipeline import detect_subjects, extract_features
from semanticvibe.render.composite import render_from_decision
from semanticvibe.schemas.decision import Decision
from semanticvibe.schemas.feature_summary import FeatureSummary

log = logging.getLogger(__name__)


def run(
    video_path: Path,
    output_path: Path,
    *,
    style_preset: str = "warm_handdrawn",
    fonts_dir: Path = Path("data/fonts"),
    assets_dir: Path = Path("data/assets_lib"),
    provider: LLMProvider | None = None,
    preview: bool = False,
    intermediate_dir: Path | None = None,
    device: str = "cuda",
) -> Path:
    """Run all 5 stages end-to-end.

    Args:
        intermediate_dir: If set, dumps FeatureSummary and Decision JSON here
            for inspection / re-rendering without re-running upstream stages.
    """
    settings = get_settings()  # surfaces .env errors early
    log.info("Pipeline cost mode: %s, provider: %s",
             settings.cost_mode, provider or settings.llm_provider)

    # Stage 1: extract_features.
    summary: FeatureSummary = extract_features(
        video_path, style_preset=style_preset, device=device
    )
    if intermediate_dir is not None:
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        (intermediate_dir / "feature_summary.json").write_text(
            summary.model_dump_json(indent=2), encoding="utf-8"
        )

    # Stage 2: LLM decide (or heuristic fallback if no API key).
    decision: Decision = decide(summary, provider=provider)
    if intermediate_dir is not None:
        (intermediate_dir / "decision.json").write_text(
            decision.model_dump_json(indent=2), encoding="utf-8"
        )

    # Stage 4: layout — resolve "auto" anchors. We pre-compute MediaPipe
    # detections once and pass them through to avoid a second model load.
    log.info("Stage 4 (layout)…")
    from moviepy import VideoFileClip

    with VideoFileClip(str(video_path)) as src:
        frame_size = (src.w, src.h)
    subjects = detect_subjects(video_path)
    decision = resolve_anchors(
        decision,
        video_path=video_path,
        frame_size=frame_size,
        fonts_dir=fonts_dir,
        subjects=subjects,
    )
    if intermediate_dir is not None:
        (intermediate_dir / "decision_resolved.json").write_text(
            decision.model_dump_json(indent=2), encoding="utf-8"
        )

    # Stage 5: render.
    log.info("Stage 5 (render)…")
    return render_from_decision(
        video_path,
        decision,
        output_path,
        fonts_dir=fonts_dir,
        assets_dir=assets_dir if assets_dir.exists() else None,
        preview=preview,
    )


def render_from_intermediate(
    video_path: Path,
    decision_json_path: Path,
    output_path: Path,
    *,
    fonts_dir: Path = Path("data/fonts"),
    assets_dir: Path = Path("data/assets_lib"),
    preview: bool = False,
) -> Path:
    """Re-render from a previously emitted Decision JSON without re-running Stages 1–4."""
    decision = Decision.model_validate(
        json.loads(decision_json_path.read_text(encoding="utf-8"))
    )
    return render_from_decision(
        video_path,
        decision,
        output_path,
        fonts_dir=fonts_dir,
        assets_dir=assets_dir if assets_dir.exists() else None,
        preview=preview,
    )
