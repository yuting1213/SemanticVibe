"""Stage 5 compositing: source video + Decision → output mp4.

MoviePy gives us video I/O and ffmpeg encoding; we hand it a `make_frame`
callback that returns an RGB frame with our overlays burned in.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

from semanticvibe.assets.clip_search import find_asset
from semanticvibe.assets.library import AssetLibrary
from semanticvibe.render import idle_animations
from semanticvibe.render.animations import AnimationState, ENTRY_DURATION, EXIT_DURATION, evaluate
from semanticvibe.render.hero_text import render_hero
from semanticvibe.render.text_render import fit_to_canvas, measure_text, render_text
from semanticvibe.schemas.decision import (
    Decision,
    DecorationElement,
    Element,
    HeroTextElement,
    TextElement,
)


# Simple person-bbox heuristic: assume the subject occupies the central
# vertical strip from 1/3 to 2/3 of the canvas width, full height. Used to
# nudge scatter / decoration anchors out of where the singer typically
# stands. Stage 4 (MediaPipe layout) supersedes this when it runs.
PERSON_BBOX_FRAC = (1 / 3, 0.0, 2 / 3, 1.0)


def _person_bbox(canvas_size: tuple[int, int]) -> tuple[int, int, int, int]:
    w, h = canvas_size
    x1f, y1f, x2f, y2f = PERSON_BBOX_FRAC
    return (int(w * x1f), int(h * y1f), int(w * x2f), int(h * y2f))


def _push_outside_person_bbox(
    rect: tuple[int, int, int, int],
    canvas_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """If `rect` (x, y, x2, y2) overlaps the person bbox, nudge it left or
    right (whichever side is closer) so its centre clears the strip.
    """
    px1, py1, px2, py2 = _person_bbox(canvas_size)
    rx1, ry1, rx2, ry2 = rect
    # Vertical bbox is full height — no point checking y overlap, just x.
    if rx2 <= px1 or rx1 >= px2:
        return rect
    rect_w = rx2 - rx1
    centre_x = (rx1 + rx2) / 2
    canvas_w, _h = canvas_size
    if centre_x < canvas_w / 2:
        # Push left: rect's right edge sits just left of px1.
        new_x1 = max(8, px1 - rect_w - 8)
    else:
        # Push right.
        new_x1 = min(canvas_w - rect_w - 8, px2 + 8)
    return (new_x1, ry1, new_x1 + rect_w, ry2)


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
    *,
    override_size: int | None = None,
) -> Image.Image | None:
    """Load the asset PNG (no jitter) and resize to override_size or base_size."""
    matches = library.by_tag(element.asset_tag)
    if not matches:
        try:
            matches = find_asset(library, element.asset_tag, top_k=1)
        except NotImplementedError:
            return None
    if not matches:
        return None
    img = Image.open(matches[0].path).convert("RGBA")
    target = override_size if override_size is not None else element.base_size
    if target is not None:
        scale = target / max(img.width, img.height)
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
    """Build the (base_tile, top-left) list for `element`.

    Single-shot returns one entry. `count > 1 + scatter` spreads `count`
    entries either across the whole frame (default) or inside `scatter_zone`
    if set. `size_steps` cycles per-copy base sizes for big/medium/small mix.
    """
    canvas_w, canvas_h = canvas_size
    rng = random.Random(seed)
    out: list[tuple[Image.Image, tuple[int, int]]] = []

    # Per-copy size override list (cycled).
    size_cycle = element.size_steps or [None]
    cached_bases: dict[int | None, Image.Image] = {}

    def _get_base(size_key: int | None) -> Image.Image | None:
        if size_key not in cached_bases:
            base = _load_decoration_base(element, library, override_size=size_key)
            if base is None:
                return None
            cached_bases[size_key] = base
        return cached_bases[size_key]

    # If no size_steps provided, _get_base(None) uses element.base_size.
    if _get_base(size_cycle[0]) is None:
        return []

    if element.scatter and element.count > 1:
        # Determine scatter region.
        if element.scatter_zone is not None:
            zx1, zy1, zx2, zy2 = element.scatter_zone
        else:
            zx1, zy1, zx2, zy2 = 8, 8, canvas_w - 8, canvas_h - 8

        for i in range(element.count):
            size_key = size_cycle[i % len(size_cycle)]
            base = _get_base(size_key)
            if base is None:
                continue
            color = element.color_tint[i % len(element.color_tint)] if element.color_tint else None
            tile = _jitter_tile(base, element, rng, color)
            tw, th = tile.size

            # Pick a centre inside the zone; if scatter_zone is set, trust the
            # author and don't apply person-bbox rejection. Otherwise prefer
            # frame edges (avoid centre subject).
            if element.scatter_zone is not None:
                cx = rng.randint(
                    max(zx1, tw // 2 + 4),
                    max(zx1 + 1, min(zx2, canvas_w) - tw // 2 - 4),
                )
                cy = rng.randint(
                    max(zy1, th // 2 + 4),
                    max(zy1 + 1, min(zy2, canvas_h) - th // 2 - 4),
                )
            else:
                for _attempt in range(5):
                    cx = rng.randint(tw // 2 + 8, max(tw // 2 + 9, canvas_w - tw // 2 - 8))
                    cy = rng.randint(th // 2 + 8, max(th // 2 + 9, canvas_h - th // 2 - 8))
                    rel_x = abs(cx - canvas_w / 2) / (canvas_w / 2)
                    if rng.random() < 0.3 + 0.7 * rel_x:
                        break
            x = cx - tw // 2
            y = cy - th // 2

            # Person-bbox push (only when no explicit zone — author-zoned
            # placements are intentional and we trust them).
            if element.scatter_zone is None:
                rect = _push_outside_person_bbox(
                    (x, y, x + tw, y + th), canvas_size
                )
                x, y = rect[0], rect[1]

            out.append((tile, (x, y)))
    else:
        for i in range(element.count):
            size_key = size_cycle[i % len(size_cycle)]
            base = _get_base(size_key)
            if base is None:
                continue
            color = element.color_tint[i % len(element.color_tint)] if element.color_tint else None
            tile = _jitter_tile(base, element, rng, color)
            anchor = _resolve_decoration_anchor(
                element, tile, text_anchors, text_sizes, canvas_size
            )
            x, y = anchor
            x += i * 6
            y += i * 6
            # Push single-anchor decorations out of the central subject strip
            # too. Without this, the heuristic's chorus star regularly
            # ended up sitting on the singer's face. Author-zoned scatters
            # already opt out of this push above; near_text_id placements
            # we trust because the author explicitly tied the decoration to
            # a specific text element.
            if element.near_text_id is None:
                tw, th = tile.size
                pushed = _push_outside_person_bbox(
                    (x, y, x + tw, y + th), canvas_size,
                )
                x, y = pushed[0], pushed[1]
            out.append((tile, (x, y)))
    return out


def _ambient_wiggle_offset(now: float, seed: int, amp: float) -> tuple[int, int]:
    """Steady-state ±amp pixel offset, oscillating at ~1 Hz, deterministic per-seed.

    Legacy support for `DecorationElement.wiggle_amp`. New JSONs should
    prefer `idle_animation: 'wiggle'` which composes through the unified
    idle pipeline.
    """
    if amp <= 0:
        return (0, 0)
    rng = random.Random(seed)
    phase_x = rng.uniform(0, 2 * math.pi)
    phase_y = rng.uniform(0, 2 * math.pi)
    dx = math.sin(now * 2 * math.pi * 1.0 + phase_x) * amp
    dy = math.cos(now * 2 * math.pi * 0.85 + phase_y) * amp
    return (int(round(dx)), int(round(dy)))


def _compose_idle(
    state: AnimationState,
    *,
    idle_name: str,
    now: float,
    start: float,
    end: float,
    seed: int,
) -> AnimationState:
    """Layer an idle animation on top of an entry-animation `AnimationState`.

    Idle modulation is only applied during the *steady-state* portion of
    the visible window — between `start + ENTRY_DURATION` and
    `end - EXIT_DURATION`. During entry/exit transitions the entry curve
    owns the look and idle would fight it.
    """
    if idle_name == "none" or idle_name is None:
        return state
    duration = end - start
    entry_end = start + min(ENTRY_DURATION, duration / 2)
    exit_start = end - min(EXIT_DURATION, duration / 2)
    if now < entry_end or now > exit_start:
        return state

    idle = idle_animations.evaluate(
        idle_name, t_since_start=now - start, seed=seed,
    )
    return AnimationState(
        alpha=state.alpha * idle.alpha_mul,
        scale=state.scale * idle.scale_mul,
        dx=state.dx + idle.dx,
        dy=state.dy + idle.dy,
        rotation_deg=state.rotation_deg + idle.rotation_deg,
        reveal_fraction=state.reveal_fraction,
    )


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
            if isinstance(element, HeroTextElement):
                # Hero handles its own alpha + breathing internally.
                result = render_hero(
                    element, now=t, fonts_dir=fonts_dir, canvas_size=canvas_size,
                )
                if result is None:
                    continue
                tile, top_left = result
                _paste_rgba(canvas, tile, top_left)
                continue

            # Pick entry animation: TextElement and DecorationElement both
            # carry `animation`; Decoration's defaults to "fade" via schema.
            anim_name = element.animation
            entry_state: AnimationState = evaluate(
                anim_name,
                now=t,
                start=element.start_time,
                end=element.end_time,
            )
            if entry_state.alpha <= 0:
                continue
            # Layer idle modulation if set + we're in the steady-state region.
            state = _compose_idle(
                entry_state,
                idle_name=element.idle_animation,
                now=t,
                start=element.start_time,
                end=element.end_time,
                seed=idx,
            )

            if isinstance(element, TextElement):
                tile = render_text(fitted_text[idx], state, fonts_dir)
                anchor_x, anchor_y = text_anchors[idx]
                _paste_rgba(canvas, tile, (int(anchor_x + state.dx), int(anchor_y + state.dy)))
            elif isinstance(element, DecorationElement):
                copies = decoration_copies.get(idx)
                if not copies:
                    continue  # missing asset — silently skip
                for copy_idx, (base_tile, (cx, cy)) in enumerate(copies):
                    # Each copy gets its own idle seed so a flock doesn't move
                    # in lockstep. State.scale + state.rotation_deg apply once
                    # per element though, so we re-use the entry state and
                    # only re-evaluate idle for per-copy phase variety.
                    copy_state = _compose_idle(
                        entry_state,
                        idle_name=element.idle_animation,
                        now=t,
                        start=element.start_time,
                        end=element.end_time,
                        seed=idx * 1000 + copy_idx,
                    )
                    tile = base_tile
                    if copy_state.scale != 1.0:
                        new_w = max(1, int(round(tile.width * copy_state.scale)))
                        new_h = max(1, int(round(tile.height * copy_state.scale)))
                        tile = tile.resize((new_w, new_h), Image.LANCZOS)
                    if copy_state.rotation_deg:
                        tile = tile.rotate(
                            copy_state.rotation_deg, resample=Image.BICUBIC, expand=True,
                        )
                    tile = _apply_alpha(tile, copy_state.alpha)
                    # Legacy wiggle_amp still composes additively for old JSONs.
                    legacy_dx, legacy_dy = _ambient_wiggle_offset(
                        t, seed=idx * 1000 + copy_idx, amp=element.wiggle_amp,
                    )
                    final_x = cx + int(round(copy_state.dx)) + legacy_dx
                    final_y = cy + int(round(copy_state.dy)) + legacy_dy
                    _paste_rgba(canvas, tile, (final_x, final_y))

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
