"""Top-level orchestrator: video → mp4 across all 5 stages.

This is the production entry point used by Streamlit (Week 5+) and any CLI
callers that want the full pipeline rather than `render_demo`'s isolated
Stage 5 path.
"""

from __future__ import annotations

from pathlib import Path

from semanticvibe.assets.library import AssetLibrary
from semanticvibe.config import LLMProvider, get_settings
from semanticvibe.layout.placement import resolve_anchors
from semanticvibe.llm.decide import decide
from semanticvibe.preprocess.pipeline import extract_features
from semanticvibe.render.composite import render_from_decision


def run(
    video_path: Path,
    output_path: Path,
    *,
    style_preset: str,
    fonts_dir: Path,
    assets_dir: Path,
    provider: LLMProvider | None = None,
    preview: bool = False,
) -> Path:
    """Run all 5 stages end-to-end.

    Stages 1, 2, 3, 4 are NotImplemented until their respective weeks land.
    Until then, use `semanticvibe.render_demo` with a hand-written Decision.
    """
    settings = get_settings()  # noqa: F841 — surfaces .env errors early

    summary = extract_features(video_path, style_preset=style_preset)
    decision = decide(summary, provider=provider)

    asset_library = AssetLibrary(assets_dir)  # noqa: F841 — Stage 4 will consume

    # Resolve "auto" anchors against frame size.
    from moviepy.editor import VideoFileClip  # local import: heavy

    with VideoFileClip(str(video_path)) as src:
        frame_size = (src.w, src.h)
    decision = resolve_anchors(decision, video_path=video_path, frame_size=frame_size)

    return render_from_decision(
        video_path,
        decision,
        output_path,
        fonts_dir=fonts_dir,
        preview=preview,
    )
