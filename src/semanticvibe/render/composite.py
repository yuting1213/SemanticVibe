"""Stage 5 compositing: source video + Decision → output mp4.

MoviePy gives us video I/O and ffmpeg encoding; we hand it a `make_frame`
callback that returns an RGB frame with our overlays burned in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from semanticvibe.render.animations import AnimationState, evaluate
from semanticvibe.render.text_render import measure_text, render_text
from semanticvibe.schemas.decision import Decision, DecorationElement, TextElement


def _resolve_anchor_xy(
    element: TextElement,
    fonts_dir: Path,
    canvas_size: tuple[int, int],
) -> tuple[int, int]:
    """Pick the top-left pixel for `element`. Falls back to a default region
    if the LLM emitted "auto" but Stage 4 hasn't run yet (Week 1 path).
    """
    if isinstance(element.anchor, tuple):
        return element.anchor

    # "auto" but no layout stage — drop into the lower band, horizontally centred.
    canvas_w, canvas_h = canvas_size
    text_w, text_h = measure_text(element, fonts_dir)
    x = max(0, (canvas_w - text_w) // 2)
    y = max(0, int(canvas_h * 0.78) - text_h // 2)
    return x, y


def _paste_rgba(canvas: Image.Image, overlay: Image.Image, top_left: tuple[int, int]) -> None:
    canvas.alpha_composite(overlay, dest=top_left)


def _make_frame_factory(
    decision: Decision,
    canvas_size: tuple[int, int],
    fonts_dir: Path,
    base_video_clip,
):
    """Build the per-time `make_frame(t)` callback for MoviePy's VideoClip."""

    def make_frame(t: float) -> np.ndarray:
        # Start from the original video frame at time t.
        src = base_video_clip.get_frame(t)  # (H, W, 3) uint8
        canvas = Image.fromarray(src).convert("RGBA")

        for element in decision.elements:
            state: AnimationState = evaluate(
                element.animation if isinstance(element, TextElement) else "fade",
                now=t,
                start=element.start_time,
                end=element.end_time,
            )
            if state.alpha <= 0:
                continue

            if isinstance(element, TextElement):
                tile = render_text(element, state, fonts_dir)
                anchor_x, anchor_y = _resolve_anchor_xy(element, fonts_dir, canvas_size)
                _paste_rgba(canvas, tile, (int(anchor_x + state.dx), int(anchor_y + state.dy)))
            elif isinstance(element, DecorationElement):
                # Decorations are wired in Week 4 once the asset library is real.
                # Skip silently for now — keeps Week 1 demo unblocked.
                continue

        return np.asarray(canvas.convert("RGB"))

    return make_frame


def render_from_decision(
    video_path: Path,
    decision: Decision,
    output_path: Path,
    *,
    fonts_dir: Path,
    fps: int | None = None,
    preview: bool = False,
) -> Path:
    """Burn `decision`'s overlays onto `video_path`, write to `output_path`.

    Args:
        preview: 720p re-encode for fast iteration (spec §10 risk mitigation).

    Returns:
        Resolved `output_path`.
    """
    # Lazy import: moviepy is ~slow to import and pulls in numpy/PIL transitively.
    # moviepy 2.x exposes everything from the top-level package (no `moviepy.editor`).
    from moviepy import VideoClip, VideoFileClip

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    src_clip = VideoFileClip(str(video_path))
    if preview and src_clip.h > 720:
        # moviepy 2.x: `resize` → `resized` (returns a new clip, immutable-style API).
        src_clip = src_clip.resized(height=720)

    canvas_size = (src_clip.w, src_clip.h)
    out_fps = fps or src_clip.fps or 24

    make_frame = _make_frame_factory(decision, canvas_size, fonts_dir, src_clip)

    overlaid = VideoClip(make_frame, duration=src_clip.duration)
    # moviepy 2.x: `set_audio` → `with_audio`.
    overlaid = overlaid.with_audio(src_clip.audio)

    overlaid.write_videofile(
        str(output_path),
        fps=out_fps,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
        logger=None,
    )

    src_clip.close()
    overlaid.close()
    return output_path
