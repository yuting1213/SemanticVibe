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
from semanticvibe.render.text_render import fit_to_canvas, measure_text, render_text
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


def _tint_rgba(img: Image.Image, hex_color: str) -> Image.Image:
    """Replace the RGB channels with `hex_color`, keeping the source alpha.

    Used to recolour a single asset PNG into the palette colours so a
    confetti scatter of hearts can show up in pink / yellow / cyan / green
    without shipping a separate PNG per colour.
    """
    from PIL import ImageColor

    r, g, b = ImageColor.getrgb(hex_color)[:3]
    rgba = img.convert("RGBA")
    pixels = np.asarray(rgba).copy()
    alpha = pixels[..., 3]
    # Where alpha > 0 keep the original alpha but replace RGB. Soft pixels
    # at the edge keep their alpha so we don't lose anti-aliasing.
    mask = alpha > 0
    pixels[..., 0] = np.where(mask, r, pixels[..., 0])
    pixels[..., 1] = np.where(mask, g, pixels[..., 1])
    pixels[..., 2] = np.where(mask, b, pixels[..., 2])
    return Image.fromarray(pixels, mode="RGBA")


def _load_decoration_base(
    element: DecorationElement,
    library: AssetLibrary,
) -> Image.Image | None:
    """Load the asset PNG (no jitter applied) and resize to base_size."""
    matches = library.by_tag(element.asset_tag)
    if not matches:
        try:
            matches = find_asset(library, element.asset_tag, top_k=1)
        except NotImplementedError:
            return None
    if not matches:
        return None
    img = Image.open(matches[0].path).convert("RGBA")
    if element.base_size is not None:
        scale = element.base_size / max(img.width, img.height)
        new_w = max(1, int(round(img.width * scale)))
        new_h = max(1, int(round(img.height * scale)))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def _jitter_tile(
    base: Image.Image,
    element: DecorationElement,
    rng: random.Random,
    color: str | None = None,
) -> Image.Image:
    """Apply (deterministic) scale + rotation jitter and optional colour tint."""
    img = base
    if color is not None:
        img = _tint_rgba(img, color)
    if element.scale_jitter:
        s = 1.0 + rng.uniform(-element.scale_jitter, element.scale_jitter)
        new_w = max(1, int(round(img.width * s)))
        new_h = max(1, int(round(img.height * s)))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    if element.rotation_jitter:
        deg = rng.uniform(-element.rotation_jitter, element.rotation_jitter)
        img = img.rotate(deg, resample=Image.BICUBIC, expand=True)
    return img


def _prepare_decoration_copies(
    element: DecorationElement,
    library: AssetLibrary,
    canvas_size: tuple[int, int],
    text_anchors: dict[int, tuple[int, int]],
    text_sizes: dict[int, tuple[int, int]],
    seed: int,
) -> list[tuple[Image.Image, tuple[int, int]]]:
    """Build the (tile, top-left) list for `element`.

    Single-shot decorations return one entry. `count > 1 + scatter` returns
    `count` entries spread across the frame (deterministic per-element seed,
    biased away from the centre vertical band where the subject usually sits).
    """
    base = _load_decoration_base(element, library)
    if base is None:
        return []

    rng = random.Random(seed)
    canvas_w, canvas_h = canvas_size
    out: list[tuple[Image.Image, tuple[int, int]]] = []

    if element.scatter and element.count > 1:
        # Confetti spread: pseudo-random positions across the frame, with a
        # mild bias against the central vertical strip (where MediaPipe
        # would have flagged the subject — Stage 4 doesn't run on
        # decorations directly so we approximate).
        for i in range(element.count):
            color = element.color_tint[i % len(element.color_tint)] if element.color_tint else None
            tile = _jitter_tile(base, element, rng, color)
            tw, th = tile.size
            # Rejection sample a few times to dodge the central strip.
            for _attempt in range(5):
                cx = rng.randint(tw // 2 + 8, max(tw // 2 + 9, canvas_w - tw // 2 - 8))
                cy = rng.randint(th // 2 + 8, max(th // 2 + 9, canvas_h - th // 2 - 8))
                rel_x = abs(cx - canvas_w / 2) / (canvas_w / 2)
                # Accept always near edges; harder to accept near horizontal centre.
                if rng.random() < 0.3 + 0.7 * rel_x:
                    break
            x = cx - tw // 2
            y = cy - th // 2
            out.append((tile, (x, y)))
    else:
        # Single (or stacked) at the resolved anchor.
        for i in range(element.count):
            color = element.color_tint[i % len(element.color_tint)] if element.color_tint else None
            tile = _jitter_tile(base, element, rng, color)
            anchor = _resolve_decoration_anchor(
                element, tile, text_anchors, text_sizes, canvas_size
            )
            # Multiple stacked copies: nudge each by a few px so they don't perfectly overlap.
            x, y = anchor
            x += i * 6
            y += i * 6
            out.append((tile, (x, y)))
    return out


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
    # Each text element is first shrunk to fit the canvas (no-op if it
    # already fits) — this prevents anchored "auto" text from bleeding off
    # the right edge on narrow portrait videos.
    fitted_text: dict[int, TextElement] = {}
    text_anchors: dict[int, tuple[int, int]] = {}
    text_sizes: dict[int, tuple[int, int]] = {}
    for idx, el in enumerate(decision.elements):
        if isinstance(el, TextElement):
            fitted = fit_to_canvas(el, fonts_dir, canvas_size)
            fitted_text[idx] = fitted
            text_anchors[idx] = _resolve_text_anchor(fitted, fonts_dir, canvas_size)
            text_sizes[idx] = measure_text(fitted, fonts_dir)

    # Each DecorationElement → list of (tile, anchor) pairs. count=1 yields
    # one pair; count>1 + scatter yields N pairs spread across the canvas.
    decoration_copies: dict[int, list[tuple[Image.Image, tuple[int, int]]]] = {}
    if library is not None:
        for idx, el in enumerate(decision.elements):
            if isinstance(el, DecorationElement):
                copies = _prepare_decoration_copies(
                    el, library, canvas_size, text_anchors, text_sizes, seed=idx
                )
                if copies:
                    decoration_copies[idx] = copies

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
                tile = render_text(fitted_text[idx], state, fonts_dir)
                anchor_x, anchor_y = text_anchors[idx]
                _paste_rgba(canvas, tile, (int(anchor_x + state.dx), int(anchor_y + state.dy)))
            elif isinstance(element, DecorationElement):
                copies = decoration_copies.get(idx)
                if not copies:
                    continue  # missing asset — silently skip
                for base_tile, (cx, cy) in copies:
                    tile = _apply_alpha(base_tile, state.alpha)
                    _paste_rgba(canvas, tile, (cx, cy))

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

    # Force even width AND height. yuv420p (the only pixel format consumer
    # players reliably support — Windows Media Player, browsers, QuickTime)
    # subsamples chroma 2:1 in both axes, so odd dimensions trigger libx264
    # to fall back to yuv444p, which most players refuse to open. Common
    # case: a portrait phone clip (e.g. 1080x1920) downscaled to height=720
    # → width 405, which is odd.
    if src_clip.w % 2 or src_clip.h % 2:
        new_w = src_clip.w - (src_clip.w % 2)
        new_h = src_clip.h - (src_clip.h % 2)
        src_clip = src_clip.resized(new_size=(new_w, new_h))

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
        # Force yuv420p — the only H.264 pixel format consumer players
        # (Windows Media Player, the Movies & TV app, browsers, QuickTime)
        # reliably accept.
        ffmpeg_params=["-pix_fmt", "yuv420p"],
        logger=None,
    )

    src_clip.close()
    overlaid.close()
    return output_path
