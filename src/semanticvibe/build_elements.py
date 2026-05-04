"""Highlights + person-masks → Decision.

The v5 replacement for the old "hand-write a Decision JSON" workflow.
Takes the `Highlight` records from `semantic_align` and builds a fully
populated `Decision` whose:

- text content comes from the actual lyrics (via Highlight.lyric_text)
- text positions are auto-picked to avoid the person (via
  layout.find_placement_zone against pose masks)
- decoration tag matches the highlight (Highlight.decoration_tag) so
  「電波」 gets a lightning, 「好き」 gets a heart, etc.
- entry / idle animations vary by Highlight.strength so the most
  punchy moments use scale_pop / stamp / drop_in and filler uses
  fade / wobble_in.

Nothing in here is hardcoded to a specific song / video. The only
constants are the rendering style (font / palette) which the caller
can override.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np

from semanticvibe.layout.zones import find_placement_zone
from semanticvibe.pose_detector import pick_nearest_mask
from semanticvibe.render.text_render import _resolve_font_file, fit_to_canvas, measure_text
from semanticvibe.schemas.decision import (
    Decision,
    DecorationElement,
    GlobalStyle,
    OutlineLayer,
    TextElement,
)
from semanticvibe.semantic_align import Highlight

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Style constants (pink + red + white per v4 baseline brief)
# ---------------------------------------------------------------------------

PALETTE_PRIMARY = "#FF6B9D"     # hot pink
PALETTE_ACCENT = "#E63946"      # red, for high-strength highlights
PALETTE_OUTLINE = "#264653"     # dark teal — outline / shadow
PALETTE_HALO = "#FFFFFF"        # white halo

DEFAULT_FONT = "KleeOne-SemiBold"
DEFAULT_BODY_FONT = "KleeOne-Regular"

# Animation pools. Higher strength → punchier entry.
STRONG_ENTRY = ["scale_pop", "stamp", "drop_in"]
NORMAL_ENTRY = ["fade", "slide_in_left", "slide_in_right", "wobble_in"]
SOFT_ENTRY = ["fade", "draw_in"]

IDLE_POOL_STRONG = ["pulse", "wiggle"]
IDLE_POOL_NORMAL = ["drift", "wiggle"]
IDLE_POOL_SOFT = ["shimmer", "drift"]


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------


def _text_size_for_strength(strength: float) -> int:
    """Bigger text for punchier moments — 64-110 px range."""
    return int(64 + (110 - 64) * strength)


def _pick_entry(strength: float, rng: random.Random) -> str:
    if strength >= 0.7:
        return rng.choice(STRONG_ENTRY)
    if strength >= 0.4:
        return rng.choice(NORMAL_ENTRY)
    return rng.choice(SOFT_ENTRY)


def _pick_idle(strength: float, rng: random.Random) -> str:
    if strength >= 0.7:
        return rng.choice(IDLE_POOL_STRONG)
    if strength >= 0.4:
        return rng.choice(IDLE_POOL_NORMAL)
    return rng.choice(IDLE_POOL_SOFT)


def _highlight_duration(highlight: Highlight, all_highlights: list[Highlight]) -> float:
    """Hold each highlight on screen for X seconds.

    Priority:
    1. `highlight.duration` if explicitly set (came from the lyric JSON's
       optional `duration` field)
    2. Gap to the next highlight minus 0.3 s breathing room, capped at 5 s
    3. 4 s default for the last highlight
    """
    if highlight.duration is not None:
        return float(highlight.duration)
    later = [h for h in all_highlights if h.lyric_time > highlight.lyric_time]
    if not later:
        return 4.0
    next_t = min(h.lyric_time for h in later)
    gap = next_t - highlight.lyric_time
    return max(1.5, min(5.0, gap - 0.3))


def _scale_mask_to_canvas(
    mask: np.ndarray, canvas_size: tuple[int, int]
) -> np.ndarray:
    """Resize a (H, W) bool mask to match the render canvas."""
    import cv2

    canvas_w, canvas_h = canvas_size
    mh, mw = mask.shape
    if (mw, mh) == (canvas_w, canvas_h):
        return mask
    resized = cv2.resize(
        mask.astype(np.uint8), (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def _quadrant_for_index(i: int) -> str:
    """Cycle through quadrants so consecutive highlights don't pile up.

    With 4 highlights this gives one in each corner; with more they wrap.
    """
    rotation = ["left_upper", "right_upper", "left_lower", "right_lower"]
    return rotation[i % len(rotation)]


def build_decision(
    highlights: list[Highlight],
    *,
    person_masks: dict[float, np.ndarray],
    canvas_size: tuple[int, int],
    fonts_dir: Path,
    seed: int = 42,
    palette_primary: str = PALETTE_PRIMARY,
    palette_accent: str = PALETTE_ACCENT,
    palette_outline: str = PALETTE_OUTLINE,
) -> Decision:
    """Assemble a `Decision` from a list of `Highlight`s + per-time pose masks.

    The caller is expected to have already:
    - run semantic_align.align(lyrics, provider) → highlights
    - run pose_detector.detect_person_mask(video) → person_masks
    - measured the render canvas size (after preview downscale)
    """
    rng = random.Random(seed)
    elements: list[TextElement | DecorationElement] = []

    for i, hl in enumerate(highlights):
        # ------- timing -------
        start_time = max(0.0, hl.lyric_time)
        end_time = start_time + _highlight_duration(hl, highlights)

        # ------- text style -------
        font = DEFAULT_FONT if hl.strength >= 0.6 else DEFAULT_BODY_FONT
        size = _text_size_for_strength(hl.strength)
        # Strong highlights go red-on-pink-outline; the rest stay
        # pink-on-dark — this gives the song a clear "punchline" hierarchy.
        if hl.strength >= 0.7:
            color = palette_accent
            outline_color = PALETTE_HALO
        else:
            color = palette_primary
            outline_color = palette_outline

        # ------- placement -------
        # Build a TextElement with anchor=(0,0) just for measurement,
        # then ask layout.find_placement_zone for a real position.
        provisional = TextElement(
            content=hl.lyric_text,
            start_time=start_time,
            end_time=end_time,
            anchor=(0, 0),
            font=font,
            size=size,
            color=color,
            outline_color=outline_color,
            outline_width=5,
            outline_layers=[OutlineLayer(color=PALETTE_HALO, width=4)],
            shadow_offset=(3, 3) if hl.strength >= 0.7 else None,
            animation=_pick_entry(hl.strength, rng),
            idle_animation=_pick_idle(hl.strength, rng),
            rotation_jitter=2.0,
            reasoning=hl.reasoning,
        )
        # Auto-shrink first so we ask layout for the actual rendered size.
        fitted = fit_to_canvas(provisional, fonts_dir, canvas_size)
        text_w, text_h = measure_text(fitted, fonts_dir)

        # Pick a person mask near this highlight's time, scaled to canvas.
        mask = pick_nearest_mask(person_masks, hl.lyric_time)
        if mask is not None:
            mask = _scale_mask_to_canvas(mask, canvas_size)
        else:
            # No mask data → no constraints, use an empty (False) mask.
            mask = np.zeros((canvas_size[1], canvas_size[0]), dtype=bool)

        prefer_quad = _quadrant_for_index(i)
        position = find_placement_zone(
            mask, target_size=(text_w, text_h), prefer=prefer_quad,
        )
        if position is None:
            log.warning(
                "Highlight %d (%r) found no placement; falling back to upper-left corner.",
                i, hl.lyric_text,
            )
            position = (16, 16)

        text_idx = len(elements)
        elements.append(fitted.model_copy(update={"anchor": position}))

        # ------- decoration (if tag specified) -------
        if hl.decoration_tag:
            decoration = DecorationElement(
                asset_tag=hl.decoration_tag,
                near_text_id=text_idx,
                start_time=start_time,
                end_time=end_time,
                base_size=int(size * 1.3),  # decoration ~30% larger than text height
                rotation_jitter=10.0,
                animation=_pick_entry(hl.strength, rng),
                idle_animation=_pick_idle(hl.strength, rng),
                color_tint=[palette_primary if hl.strength < 0.7 else palette_accent],
                reasoning=f"matched tag {hl.decoration_tag!r} via semantic_align",
            )
            elements.append(decoration)

    # ------- global style -------
    return Decision(
        elements=elements,
        global_style=GlobalStyle(
            color_palette=[palette_primary, palette_accent, PALETTE_HALO,
                           palette_outline],
            vibe=(
                "v5 lyrics-driven hand-drawn picturebook — text content from "
                "actual lyrics, positions auto-avoid detected pose, decorations "
                "matched via tag vocabulary (pink + red + white only)"
            ),
        ),
    )
