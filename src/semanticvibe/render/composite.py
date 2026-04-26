"""Stage 5 compositing: source video + Decision → output mp4.

MoviePy gives us video I/O and ffmpeg encoding; we hand it a `make_frame`
callback that returns an RGB frame with our overlays burned in.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image

from semanticvibe.assets.clip_search import find_asset
from semanticvibe.assets.library import AssetLibrary
from semanticvibe.render.animations import AnimationState, evaluate
from semanticvibe.render.text_render import measure_text, render_text
from semanticvibe.schemas.decision import (
    Decision,
    DecorationElement,
    Element,
    TextElement,
)


def _resolve_text_anchor(
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


def _prepare_decoration_tile(
    element: DecorationElement,
    library: AssetLibrary,
    seed: int,
) -> Image.Image | None:
    """Load the asset PNG and bake in scale/rotation jitter once, deterministically.

    Returns None if the tag has no matching asset — caller skips silently so a
    missing asset doesn't break the whole render.
    """
    matches = library.by_tag(element.asset_tag)
    if not matches:
        try:
            matches = find_asset(library, element.asset_tag, top_k=1)
        except NotImplementedError:
            # CLIP fallback isn't built yet — Week 1 only does exact-tag matches.
            return None
    if not matches:
        return None

    img = Image.open(matches[0].path).convert("RGBA")

    # Deterministic per-element jitter. Two elements with the same tag still
    # look different; the same render run is reproducible.
    rng = random.Random(seed)
    if element.scale_jitter:
        s = 1.0 + rng.uniform(-element.scale_jitter, element.scale_jitter)
        new_w = max(1, int(round(img.width * s)))
        new_h = max(1, int(round(img.height * s)))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    if element.rotation_jitter:
        deg = rng.uniform(-element.rotation_jitter, element.rotation_jitter)
        img = img.rotate(deg, resample=Image.BICUBIC, expand=True)
    return img


def _resolve_decoration_anchor(
    element: DecorationElement,
    tile: Image.Image,
    text_anchors: dict[int, tuple[int, int]],
    text_sizes: dict[int, tuple[int, int]],
    canvas_size: tuple[int, int],
) -> tuple[int, int]:
    """Pick the top-left pixel for the decoration tile.

    Heuristic for Week 1 (Stage 4 will replace this):
    - If `near_text_id` resolved to a known text element, snug the decoration
      to the upper-right of that text's bbox so it reads as an emphasis
      sticker on the title.
    - Otherwise, top-right safe zone of the frame.
    """
    canvas_w, canvas_h = canvas_size
    tile_w, tile_h = tile.size

    if element.near_text_id is not None and element.near_text_id in text_anchors:
        text_x, text_y = text_anchors[element.near_text_id]
        text_w, text_h = text_sizes[element.near_text_id]
        x = text_x + text_w - tile_w // 3
        y = text_y - tile_h + tile_h // 3
    else:
        x = canvas_w - tile_w - 24
        y = 24

    # Clamp inside the canvas with a small margin.
    x = max(8, min(x, canvas_w - tile_w - 8))
    y = max(8, min(y, canvas_h - tile_h - 8))
    return x, y


def _paste_rgba(canvas: Image.Image, overlay: Image.Image, top_left: tuple[int, int]) -> None:
    canvas.alpha_composite(overlay, dest=top_left)


def _apply_alpha(img: Image.Image, alpha: float) -> Image.Image:
    if alpha >= 1.0:
        return img
    a = img.getchannel("A")
    a = a.point(lambda px: int(px * alpha))
    out = img.copy()
    out.putalpha(a)
    return out


def _make_frame_factory(
    decision: Decision,
    canvas_size: tuple[int, int],
    fonts_dir: Path,
    library: AssetLibrary | None,
    base_video_clip,
):
    """Build the per-time `make_frame(t)` callback for MoviePy's VideoClip."""

    # Pre-resolve everything that doesn't depend on `t` so make_frame is tight.
    text_anchors: dict[int, tuple[int, int]] = {}
    text_sizes: dict[int, tuple[int, int]] = {}
    for idx, el in enumerate(decision.elements):
        if isinstance(el, TextElement):
            text_anchors[idx] = _resolve_text_anchor(el, fonts_dir, canvas_size)
            text_sizes[idx] = measure_text(el, fonts_dir)

    decoration_tiles: dict[int, Image.Image] = {}
    decoration_anchors: dict[int, tuple[int, int]] = {}
    if library is not None:
        for idx, el in enumerate(decision.elements):
            if isinstance(el, DecorationElement):
                tile = _prepare_decoration_tile(el, library, seed=idx)
                if tile is None:
                    continue
                decoration_tiles[idx] = tile
                decoration_anchors[idx] = _resolve_decoration_anchor(
                    el, tile, text_anchors, text_sizes, canvas_size
                )

    elements: list[tuple[int, Element]] = list(enumerate(decision.elements))

    def make_frame(t: float) -> np.ndarray:
        # Start from the original video frame at time t.
        src = base_video_clip.get_frame(t)  # (H, W, 3) uint8
        canvas = Image.fromarray(src).convert("RGBA")

        for idx, element in elements:
            anim_name = element.animation if isinstance(element, TextElement) else "fade"
            state: AnimationState = evaluate(
                anim_name,
                now=t,
                start=element.start_time,
                end=element.end_time,
            )
            if state.alpha <= 0:
                continue

            if isinstance(element, TextElement):
                tile = render_text(element, state, fonts_dir)
                anchor_x, anchor_y = text_anchors[idx]
                _paste_rgba(canvas, tile, (int(anchor_x + state.dx), int(anchor_y + state.dy)))
            elif isinstance(element, DecorationElement):
                base_tile = decoration_tiles.get(idx)
                if base_tile is None:
                    continue  # missing asset — silently skip
                tile = _apply_alpha(base_tile, state.alpha)
                anchor_x, anchor_y = decoration_anchors[idx]
                _paste_rgba(canvas, tile, (anchor_x, anchor_y))

        return np.asarray(canvas.convert("RGB"))

    return make_frame


def render_from_decision(
    video_path: Path,
    decision: Decision,
    output_path: Path,
    *,
    fonts_dir: Path,
    assets_dir: Path | None = None,
    fps: int | None = None,
    preview: bool = False,
) -> Path:
    """Burn `decision`'s overlays onto `video_path`, write to `output_path`.

    Args:
        assets_dir: Root of the decoration asset library. If omitted (or empty),
            DecorationElements are silently skipped — useful for text-only renders.
        preview: 720p re-encode for fast iteration (spec §10 risk mitigation).

    Returns:
        Resolved `output_path`.
    """
    # Lazy import: moviepy is ~slow to import and pulls in numpy/PIL transitively.
    # moviepy 2.x exposes everything from the top-level package (no `moviepy.editor`).
    from moviepy import VideoClip, VideoFileClip

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    library: AssetLibrary | None = None
    if assets_dir is not None and assets_dir.exists():
        library = AssetLibrary(assets_dir)

    src_clip = VideoFileClip(str(video_path))
    if preview and src_clip.h > 720:
        # moviepy 2.x: `resize` → `resized` (returns a new clip, immutable-style API).
        src_clip = src_clip.resized(height=720)

    canvas_size = (src_clip.w, src_clip.h)
    out_fps = fps or src_clip.fps or 24

    make_frame = _make_frame_factory(decision, canvas_size, fonts_dir, library, src_clip)

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
