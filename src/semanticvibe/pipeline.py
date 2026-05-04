"""Top-level orchestrator: video → mp4 across all 5 stages."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

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
    asr_language: str | None = "zh",
    progress_cb: Callable[[str], None] | None = None,
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
    if progress_cb: progress_cb("stage 1/5: extracting features")
    summary: FeatureSummary = extract_features(
        video_path, style_preset=style_preset, device=device, asr_language=asr_language
    )
    if intermediate_dir is not None:
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        (intermediate_dir / "feature_summary.json").write_text(
            summary.model_dump_json(indent=2), encoding="utf-8"
        )

    # Stage 2: LLM decide (or heuristic fallback if no API key).
    if progress_cb: progress_cb("stage 2/5: deciding on vibe and layout")
    decision: Decision = decide(summary, provider=provider)
    if intermediate_dir is not None:
        (intermediate_dir / "decision.json").write_text(
            decision.model_dump_json(indent=2), encoding="utf-8"
        )

    # Stage 2b: beat-driven animation randomization.
    # Snaps each element's start_time to the nearest beat (max 0.15 s) and
    # picks a beat-appropriate entry+idle animation. Kept outside of decide()
    # because it needs the audio (for downbeat energy classification) — the
    # LLM only ever sees text, so this enrichment lives at the pipeline level.
    log.info("Stage 2b (beat sync + animation assignment)…")
    if progress_cb: progress_cb("stage 2b/5: syncing animations to beats")
    from semanticvibe.llm.anim_assignment import assign_random_animations
    from semanticvibe.preprocess.beat_sync import BeatInfo

    beat_info = BeatInfo.from_video(video_path)
    decision = assign_random_animations(decision, beat_info)
    if intermediate_dir is not None:
        (intermediate_dir / "decision_animated.json").write_text(
            decision.model_dump_json(indent=2), encoding="utf-8"
        )

    # Stage 4: layout — resolve "auto" anchors against the FINAL rendered
    # canvas size (i.e. account for the preview downscale). Otherwise layout
    # picks coordinates valid only in the source resolution and the renderer
    # then crops them off the smaller frame.
    log.info("Stage 4 (layout)…")

    if progress_cb: progress_cb("stage 4/5: resolving layout anchors against frame size")
    from moviepy import VideoFileClip

    with VideoFileClip(str(video_path)) as src:
        src_w, src_h = src.w, src.h
    if preview and src_h > 720:
        scale = 720 / src_h
        frame_size = (int(src_w * scale), 720)
    else:
        frame_size = (src_w, src_h)

    subjects = detect_subjects(video_path)
    if preview and src_h > 720:
        # Subject boxes are in source pixel space; rescale to match frame_size
        # so the occupancy mask aligns with what the renderer will see.
        from semanticvibe.preprocess.mediapipe_pose import SubjectBox

        s = 720 / src_h
        subjects = [
            SubjectBox(
                frame_time=b.frame_time,
                x=int(b.x * s),
                y=int(b.y * s),
                w=int(b.w * s),
                h=int(b.h * s),
            )
            for b in subjects
        ]
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
    if progress_cb: progress_cb("stage 5/5: rendering video")
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
