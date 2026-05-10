"""Highlights + person-masks → Decision (legacy) / list[dict] (v6).

Two public entry points:

- `build_decision(highlights, person_masks, canvas_size, fonts_dir)` — v5
  contract. Returns a Pydantic `Decision`. Used by the existing CLI and
  by `render_from_decision` directly.
- `build_elements_from_lyrics(lyrics, *, song_title, provider, seed)` — v6
  high-level helper. Runs `align_lyrics → highlights`, picks one hero per
  song from the strongest hook, then emits a flat `list[dict]` of element
  records. `is_hook` lines may upgrade to `hero_text`; the rest become
  `text` + (optionally) `decoration` pairs. The dict shape mirrors the
  Pydantic schema so callers can `model_validate` if they want a Decision.

Nothing here is hardcoded to a specific song / video. Style constants
(font, palette) are caller-overridable.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np

from semanticvibe.layout.zones import find_placement_zone
from semanticvibe.pose_detector import pick_nearest_mask
from semanticvibe.render.text_render import _resolve_font_file, fit_to_canvas, measure_text
from semanticvibe.schemas.decision import (
    Decision,
    DecorationElement,
    GlobalStyle,
    HeroTextElement,
    OutlineLayer,
    SubtitleBannerElement,
    TextElement,
)
from semanticvibe.lyrics import LyricLine
from semanticvibe.semantic_align import Highlight, align_lyrics
from semanticvibe.style import default_style_name, get_style

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
    later = [h for h in all_highlights if h.time > highlight.time]
    if not later:
        return 4.0
    next_t = min(h.time for h in later)
    gap = next_t - highlight.time
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


def _hero_substring(hl: Highlight, max_len: int = 3) -> str:
    """Extract the most evocative 2-3 char chunk from a lyric line for hero display.

    Strategy: if the line contains any KEYWORD_TO_TAGS trigger, return the
    longest such trigger (truncated to max_len). This keeps 「可愛い」-bearing
    lines showing 「可愛い」 rather than the line's first few chars.
    """
    from semanticvibe.semantic_align import KEYWORD_TO_TAGS

    lowered = hl.text.lower()
    matches = [
        trig for trig in KEYWORD_TO_TAGS
        if trig and trig.lower() in lowered
    ]
    if matches:
        # Longest trigger wins — most-specific match.
        chosen = max(matches, key=len)
        return chosen[:max_len]
    return hl.text[:max_len]


def _pick_banner_position(
    banner_height: int,
    canvas_size: tuple[int, int],
    mask: np.ndarray | None,
    margin: int = 16,
) -> str:
    """Return 'top_banner' or 'bottom_banner' based on pose-mask occupancy.

    Uses two checks:
    - banner-height band: the actual rectangle the banner will occupy
    - upper/lower quarter: catches "hair / arms above the pose bbox"
      since MediaPipe Pose landmarks are face-centred and miss the hair
      above the head and raised arms above shoulders

    Default-to-top, but switch to bottom whenever the top is meaningfully
    populated AND the bottom is less so. Specifically:
      - top band >5% occupied + bottom less occupied → bottom
      - upper quarter >25% occupied (face/head present in upper region) +
        bottom less occupied → bottom
    """
    if mask is None:
        return "top_banner"
    canvas_w, canvas_h = canvas_size
    band_h = banner_height + margin
    mh, _mw = mask.shape
    band_h_mask = max(1, int(band_h * mh / canvas_h))
    top_band = mask[:band_h_mask, :].mean()
    bot_band = mask[mh - band_h_mask:, :].mean()
    upper_q = mask[: mh // 4, :].mean()
    # Hair safety: MediaPipe Pose's bbox is anchored on face landmarks
    # (nose/eyes/ears) and misses 50-100px of hair above. So when the
    # face is anywhere in the upper quarter, force bottom — even if the
    # top-strip itself looks empty. Covering legs/clothes is far more
    # acceptable than covering the face.
    if upper_q > 0.15:
        return "bottom_banner"
    if top_band > 0.05 and top_band < bot_band:
        return "bottom_banner"
    return "top_banner"


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
    style: str | None = None,
    subtitle_style: str | None = None,
    palette_primary: str | None = None,
    palette_accent: str | None = None,
    palette_outline: str | None = None,
) -> Decision:
    """Assemble a `Decision` from a list of `Highlight`s + per-time pose masks.

    `style` selects a preset from `assets/styles.json`; `subtitle_style`
    overrides the preset's `subtitle_default` ("banner" or "hero").
    Legacy palette overrides still win when explicitly passed in.

    The caller is expected to have already:
    - run semantic_align.align(lyrics, provider) → highlights
    - run pose_detector.detect_person_mask(video) → person_masks
    - measured the render canvas size (after preview downscale)
    """
    rng = random.Random(seed)
    style_name = style or default_style_name()
    preset = get_style(style_name)
    text_cfg = preset.get("text", {})
    banner_cfg = preset.get("subtitle_banner", {})
    palette_primary = palette_primary or text_cfg.get("color", PALETTE_PRIMARY)
    palette_accent = palette_accent or text_cfg.get("color_strong", PALETTE_ACCENT)
    palette_outline = palette_outline or text_cfg.get("outline_color", PALETTE_OUTLINE)
    palette_halo = text_cfg.get("halo_color", PALETTE_HALO)
    decoration_palette = preset.get("decoration_color_palette", [palette_primary])
    ambient_tags = preset.get("ambient_tags", [])
    sub_mode = subtitle_style or preset.get("subtitle_default", "hero")
    if sub_mode not in ("banner", "hero"):
        log.warning("unknown subtitle_style %r, falling back to 'hero'", sub_mode)
        sub_mode = "hero"

    # ---- Tag expansion: pad under-tagged highlights with ambient tags. ----
    # Avoids "everything is sparkle" by guaranteeing ≥2 tags per highlight
    # when ambient_tags are configured. Preserves order: original tags first,
    # then unique ambient additions.
    if ambient_tags:
        for hl in highlights:
            if len(hl.tags) < 2:
                merged = list(dict.fromkeys(hl.tags + ambient_tags))[:3]
                hl.tags = merged
                if hl.primary_tag is None and merged:
                    hl.primary_tag = merged[0]

    elements: list[TextElement | DecorationElement | HeroTextElement | SubtitleBannerElement] = []

    # ------- ambient sparkle scatter (song-wide background confetti) -------
    # Matches the baseline3 reference: tiny sparkle dots scattered across the
    # frame for the full song duration. Always emit when there's at least one
    # highlight; the renderer will skip silently if no `sparkle` asset exists.
    if highlights:
        song_start = max(0.0, min(h.time for h in highlights) - 1.0)
        song_end = max(h.time for h in highlights) + _highlight_duration(
            max(highlights, key=lambda h: h.time), highlights
        ) + 1.0
        elements.append(DecorationElement(
            asset_tag="sparkle",
            start_time=song_start,
            end_time=song_end,
            count=14,
            scatter=True,
            base_size=max(28, canvas_size[0] // 14),
            rotation_jitter=20.0,
            animation="fade",
            idle_animation="shimmer",
            color_tint=[],
            reasoning="ambient sparkle confetti spanning the song",
        ))

    # ------- one hero glyph per song (the strongest-tagged highlight) -------
    # Skip in 'banner' mode — there each lyric line gets its own subtitle
    # banner, and a centred hero would just overlap the dancer's face.
    if sub_mode == "hero":
        tagged = [h for h in highlights if h.primary_tag]
        if tagged:
            hero_hl = max(tagged, key=lambda h: (h.strength, -h.time))
            hero_text = _hero_substring(hero_hl)
            hero_start = max(0.0, hero_hl.time - 0.3)
            hero_end = hero_start + min(3.5, _highlight_duration(hero_hl, highlights) + 1.0)
            budget = int(canvas_size[0] * 0.8)
            n = max(1, len(hero_text))
            hero_size = max(120, min(280, budget // n))
            elements.append(HeroTextElement(
                content=hero_text,
                pos="center_upper",
                size=hero_size,
                color=palette_accent,
                halo_color=palette_halo,
                style="chalk",
                breathing=True,
                grain=True,
                start_time=hero_start,
                end_time=hero_end,
                reasoning=f"hero of song — strongest tag={hero_hl.primary_tag!r}",
            ))

    for i, hl in enumerate(highlights):
        # ------- timing -------
        start_time = max(0.0, hl.time)
        end_time = start_time + _highlight_duration(hl, highlights)

        text_idx: int | None = None

        if sub_mode == "banner":
            # ------- baseline3 style: full-line subtitle chip ----------------
            # Pose-aware position: if the dancer's head is at the top of
            # the frame at this lyric's time, drop the banner to the bottom.
            banner_height_estimate = banner_cfg.get("size", 42) + banner_cfg.get("padding", 18) * 2
            mask = pick_nearest_mask(person_masks, hl.time)
            if mask is not None:
                mask = _scale_mask_to_canvas(mask, canvas_size)
            banner_position = _pick_banner_position(
                banner_height_estimate, canvas_size, mask,
            )
            banner = SubtitleBannerElement(
                content=hl.text,
                start_time=start_time,
                end_time=end_time,
                position=banner_position,
                font=banner_cfg.get("font", DEFAULT_FONT),
                size=banner_cfg.get("size", 42),
                text_color=banner_cfg.get("text_color", palette_halo),
                outline_color=banner_cfg.get("outline_color", palette_outline),
                outline_width=banner_cfg.get("outline_width", 4),
                bg_color=banner_cfg.get("bg_color", palette_primary),
                bg_alpha=banner_cfg.get("bg_alpha", 140),
                corner_radius=banner_cfg.get("corner_radius", 16),
                padding=banner_cfg.get("padding", 18),
                reasoning=(
                    f"{hl.reasoning or 'subtitle banner'} (pos={banner_position} "
                    f"avoiding pose at t={hl.time:.1f}s)"
                ),
            )
            elements.append(banner)
            size = banner_cfg.get("size", 42)
        else:
            # ------- legacy hero/text style: pose-aware floating text -------
            font = DEFAULT_FONT if hl.strength >= 0.6 else DEFAULT_BODY_FONT
            size = _text_size_for_strength(hl.strength)
            if hl.strength >= 0.7:
                color = palette_accent
                outline_color = palette_halo
            else:
                color = palette_primary
                outline_color = palette_outline

            provisional = TextElement(
                content=hl.text,
                start_time=start_time,
                end_time=end_time,
                anchor=(0, 0),
                font=font,
                size=size,
                color=color,
                outline_color=outline_color,
                outline_width=5,
                outline_layers=[OutlineLayer(color=palette_halo, width=4)],
                shadow_offset=(3, 3) if hl.strength >= 0.7 else None,
                animation=_pick_entry(hl.strength, rng),
                idle_animation=_pick_idle(hl.strength, rng),
                rotation_jitter=2.0,
                reasoning=hl.reasoning or "auto-generated by build_decision",
            )
            fitted = fit_to_canvas(provisional, fonts_dir, canvas_size)
            text_w, text_h = measure_text(fitted, fonts_dir)

            mask = pick_nearest_mask(person_masks, hl.time)
            if mask is not None:
                mask = _scale_mask_to_canvas(mask, canvas_size)
            else:
                mask = np.zeros((canvas_size[1], canvas_size[0]), dtype=bool)

            prefer_quad = _quadrant_for_index(i)
            position = find_placement_zone(
                mask, target_size=(text_w, text_h), prefer=prefer_quad,
            )
            if position is None:
                log.warning(
                    "Highlight %d (%r) found no placement; falling back to upper-left corner.",
                    i, hl.text,
                )
                position = (16, 16)

            text_idx = len(elements)
            elements.append(fitted.model_copy(update={"anchor": position}))

        # ------- decorations (one per tag, up to 2 — colour-balanced) -------
        # Walk hl.tags so a single line emits a heart + a star, not just one
        # of either. Cycle decoration_palette so the colour mix stays even
        # across the song.
        for tag_pos, tag in enumerate(hl.tags[:2]):
            tint_color = decoration_palette[
                (i + tag_pos) % max(1, len(decoration_palette))
            ]
            decoration = DecorationElement(
                asset_tag=tag,
                near_text_id=text_idx,  # None in banner mode → renderer scatters
                start_time=start_time,
                end_time=end_time,
                base_size=int(size * 1.3),
                rotation_jitter=10.0,
                animation=_pick_entry(hl.strength, rng),
                idle_animation=_pick_idle(hl.strength, rng),
                # color_tint=[] preserves the asset's own colour (v6 hand-drawn
                # assets are full-colour). The retriever picks a colour-diverse
                # PNG from the available pool via decoration_color_palette.
                color_tint=[],
                reasoning=(
                    f"tag {tag!r} (priority {tag_pos}) for line {hl.text!r}; "
                    f"target colour bucket {tint_color}"
                ),
            )
            elements.append(decoration)

    # ------- global style -------
    return Decision(
        elements=elements,
        global_style=GlobalStyle(
            color_palette=decoration_palette + [palette_outline],
            vibe=(
                f"v7 style={style_name!r} subtitle={sub_mode!r} — "
                "lyrics-driven, pose-aware text + colour-balanced decorations "
                "+ ambient sparkle scatter"
            ),
        ),
    )


# ===========================================================================
# v6 — build_elements_from_lyrics (high-level, JSON-friendly output)
# ===========================================================================
#
# `build_decision` above couples element generation to layout (it needs
# `person_masks` + `canvas_size` to place text). For situations where the
# caller just wants a list of element *intents* (text content, decoration
# tag, animations, timing) and intends to plug them into a separate layout
# pass, `build_elements_from_lyrics` is the right entry point.
#
# Output shape: `list[dict]` matching the Pydantic schema. Callers can
# `Decision.model_validate({"elements": elements, "global_style": {...}})`
# to round-trip into the typed Decision when convenient.

# Hero rules — pick the most-tagged hook with the strongest signal as THE
# hero of the song. Single hero per song keeps the look from getting busy.
_HERO_MAX_LEN = 4   # don't promote phrases longer than 4 chars to hero
_HERO_DURATION = 3.5  # seconds the hero stays on screen


def _pick_hero(highlights: list[Highlight]) -> Highlight | None:
    """Pick at most one highlight to render as a hero glyph.

    Preference: hook + tagged + short. Falls back to None when no candidate
    fits (e.g. all hooks are long phrases)."""
    candidates = [
        h for h in highlights
        if h.is_hook and h.tags and len(h.text) <= _HERO_MAX_LEN
    ]
    if not candidates:
        return None
    # Earliest hook wins — gives the song an opening punch.
    return min(candidates, key=lambda h: h.time)


def build_elements_from_lyrics(
    lyrics: list[LyricLine] | list[dict],
    *,
    song_title: str | None = None,
    provider: str = "rule_based",
    seed: int = 42,
    palette_primary: str = PALETTE_PRIMARY,
    palette_accent: str = PALETTE_ACCENT,
    palette_outline: str = PALETTE_OUTLINE,
) -> list[dict[str, Any]]:
    """Lyrics → flat list of element dicts (text + decoration + hero_text).

    Layout is left to the renderer: text elements emit `anchor="auto"`
    and decoration elements emit `near_text_id` pointing at their paired
    text. Pass the result to `render_from_decision` after wrapping in a
    `Decision` (the renderer owns the placement step).
    """
    rng = random.Random(seed)
    parsed = [
        L if isinstance(L, LyricLine) else LyricLine.model_validate(L) for L in lyrics
    ]

    result = align_lyrics(parsed, provider=provider, song_title=song_title)
    highlights = result.highlights
    hero = _pick_hero(highlights)

    elements: list[dict[str, Any]] = []

    # ----- hero (one per song, if any) -----
    if hero is not None:
        hero_start = max(0.0, hero.time - 0.3)
        hero_end = hero_start + _HERO_DURATION
        elements.append({
            "type": "hero_text",
            "content": hero.text[:_HERO_MAX_LEN],
            "pos": "center_upper",
            "size": 280,
            "color": palette_accent,
            "halo_color": "#FFFFFF",
            "style": "chalk",
            "breathing": True,
            "grain": True,
            "start_time": hero_start,
            "end_time": hero_end,
            "reasoning": f"hero of song — strongest hook tag={hero.primary_tag!r}",
        })

    # ----- per-highlight text + decoration -----
    for hl in highlights:
        # Skip the hook we already lifted into hero_text — don't duplicate.
        if hero is not None and hl is hero:
            continue
        # Skip totally-untagged filler unless caller wants it as plain text.
        # Default: drop them so non_hooks don't crowd the screen.
        if not hl.tags and not hl.is_hook:
            continue

        start_time = max(0.0, hl.time)
        end_time = start_time + _highlight_duration(hl, highlights)

        size = _text_size_for_strength(hl.strength)
        if hl.strength >= 0.7:
            color = palette_accent
        else:
            color = palette_primary

        text_idx = len(elements)
        elements.append({
            "type": "text",
            "content": hl.text,
            "anchor": "auto",
            "font": DEFAULT_FONT if hl.strength >= 0.6 else DEFAULT_BODY_FONT,
            "size": size,
            "color": color,
            "outline_color": PALETTE_HALO,
            "outline_width": 5,
            "outline_layers": [{"color": palette_outline, "width": 3}],
            "shadow_offset": [3, 3] if hl.strength >= 0.7 else None,
            "animation": _pick_entry(hl.strength, rng),
            "idle_animation": _pick_idle(hl.strength, rng),
            "rotation_jitter": 2.0,
            "start_time": start_time,
            "end_time": end_time,
            "reasoning": hl.reasoning or "auto-generated text overlay",
        })

        # Emit one decoration per tag (cap at 2 to avoid clutter).
        for tag in hl.tags[:2]:
            elements.append({
                "type": "decoration",
                "asset_tag": tag,
                "near_text_id": text_idx,
                "base_size": int(size * 1.3),
                "rotation_jitter": 10.0,
                "animation": _pick_entry(hl.strength, rng),
                "idle_animation": _pick_idle(hl.strength, rng),
                "color_tint": [palette_primary if hl.strength < 0.7 else palette_accent],
                "start_time": start_time,
                "end_time": end_time,
                "reasoning": f"semantic_align tag={tag!r} for line {hl.text!r}",
            })

    return elements
