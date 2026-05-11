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

from semanticvibe.layout.forbidden_map import build_forbidden_map_at_time
from semanticvibe.layout.zones import find_placement_zone
from semanticvibe.pose_detector import pick_nearest_mask
from semanticvibe.render.text_render import (
    _resolve_font_file,
    fit_subtitle_outlined_to_canvas,
    fit_to_canvas,
    measure_subtitle_outlined,
    measure_text,
    resolve_subtitle_outlined_anchor,
)
from semanticvibe.schemas.decision import (
    Decision,
    DecorationElement,
    GlobalStyle,
    HeroTextElement,
    OutlineLayer,
    SubtitleBannerElement,
    SubtitleOutlinedElement,
    TextElement,
)
from semanticvibe.beat_sync import (
    BeatInfo,
    average_beat_period,
    detect_beats,
    is_downbeat,
    is_high_energy,
    snap_to_beat,
)
from semanticvibe.motion_detector import (
    MotionInfo,
    detect_motion_peaks,
    motion_intensity_at,
)
from semanticvibe.vlm_gesture import GestureEvent, detect_gestures
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

# Beat-aware overrides: when a highlight lands ON a downbeat we want the
# punchiest entry available regardless of the line's "strength" score —
# the music's drum hit already commits to drama and the visual should
# match. spin_in is reserved for "magical" downbeat moments.
DOWNBEAT_ENTRY = ["stamp", "drop_in", "scale_pop", "spin_in"]

# v12 motion-aware overrides — landed when the dancer hits a body-motion
# peak (MediaPipe upper-body velocity, z-score normalised). Motion wins
# over downbeat because the viewer's eye is already locked on the dancer
# at a peak gesture; the animation hits the visible beat.
MOTION_ENTRY_HIGH = ["stamp", "spin_in", "drop_in"]
MOTION_ENTRY_MEDIUM = ["scale_pop", "wobble_in"]
MOTION_ENTRY_LOW = ["fade", "slide_in_left", "slide_in_right"]

IDLE_POOL_STRONG = ["pulse", "wiggle"]
IDLE_POOL_NORMAL = ["drift", "wiggle"]
IDLE_POOL_SOFT = ["shimmer", "drift"]
# In a high-energy chorus segment, force pulse so the on-screen elements
# breathe with the music's loudest moments.
CHORUS_IDLE = "pulse"


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------


def _text_size_for_strength(strength: float) -> int:
    """Bigger text for punchier moments — 64-110 px range."""
    return int(64 + (110 - 64) * strength)


def _pick_entry(
    strength: float,
    rng: random.Random,
    hint: str | None = None,
    *,
    is_downbeat_hit: bool = False,
    motion_intensity: str | None = None,
) -> str:
    """Prefer the LLM-supplied hint; fall back to strength-bucketed pool.

    Priority chain (v12):
        explicit hint  >  motion peak  >  downbeat  >  strength bucket

    Motion peak wins over downbeat: the viewer's eye is already on the
    dancer at a body-motion peak, so the punchy animation lands on the
    visible beat. Downbeat is auditory and may not coincide with on-
    screen action.
    """
    if hint:
        return hint
    if motion_intensity == "high":
        return rng.choice(MOTION_ENTRY_HIGH)
    if motion_intensity == "medium":
        return rng.choice(MOTION_ENTRY_MEDIUM)
    if motion_intensity == "low":
        return rng.choice(MOTION_ENTRY_LOW)
    if is_downbeat_hit:
        return rng.choice(DOWNBEAT_ENTRY)
    if strength >= 0.7:
        return rng.choice(STRONG_ENTRY)
    if strength >= 0.4:
        return rng.choice(NORMAL_ENTRY)
    return rng.choice(SOFT_ENTRY)


def _pick_idle(
    strength: float,
    rng: random.Random,
    hint: str | None = None,
    *,
    in_chorus: bool = False,
) -> str:
    if hint:
        return hint
    if in_chorus:
        return CHORUS_IDLE
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


def _corner_zones(canvas_size: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """Four corner regions for hero-decoration placement preference."""
    w, h = canvas_size
    return [
        (int(w * 0.65), int(h * 0.05), int(w * 0.95), int(h * 0.20)),  # right_upper
        (int(w * 0.05), int(h * 0.05), int(w * 0.35), int(h * 0.20)),  # left_upper
        (int(w * 0.05), int(h * 0.65), int(w * 0.35), int(h * 0.85)),  # left_lower
        (int(w * 0.65), int(h * 0.65), int(w * 0.95), int(h * 0.85)),  # right_lower
    ]


def _scaled_person_masks(
    person_masks: dict[float, np.ndarray] | None,
    canvas_size: tuple[int, int],
) -> dict[float, np.ndarray]:
    """Pre-scale every pose mask to canvas resolution so ForbiddenMap can
    OR them in directly without re-scaling per highlight."""
    if not person_masks:
        return {}
    out: dict[float, np.ndarray] = {}
    for t, m in person_masks.items():
        if m is None:
            continue
        out[t] = _scale_mask_to_canvas(m, canvas_size)
    return out


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
    audio_path: Path | str | None = None,
    beat_sync: bool = True,
    video_path: Path | str | None = None,
    motion_aware: bool = True,
    vlm_gestures: bool = True,
    vlm_model: str = "qwen2.5vl:7b",
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
    if sub_mode not in ("banner", "hero", "outlined"):
        log.warning("unknown subtitle_style %r, falling back to 'outlined'", sub_mode)
        sub_mode = "outlined"

    # ---- v9: beat detection (optional) -----------------------------------
    beat_info: BeatInfo | None = None
    beat_period: float | None = None
    snap_count = 0
    if beat_sync and audio_path:
        try:
            beat_info = detect_beats(str(audio_path))
            beat_period = average_beat_period(beat_info["beat_times"])
            log.info(
                "[beat_sync] driving build_decision: tempo=%.1f BPM, beat_period=%.3fs",
                beat_info["tempo"], beat_period or 0.0,
            )
            for hl in highlights:
                snapped = snap_to_beat(hl.time, beat_info["beat_times"])
                if snapped != hl.time:
                    snap_count += 1
                    hl.time = snapped
            log.info(
                "[beat_sync] snapped %d / %d highlights to nearest beat (max ±0.15s)",
                snap_count, len(highlights),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("beat detection failed (%s); continuing without beat sync", exc)
            beat_info = None

    # ---- v12: motion detection (pose-velocity peaks) ---------------------
    motion_info: MotionInfo | None = None
    if motion_aware and video_path:
        try:
            motion_info = detect_motion_peaks(str(video_path))
            if motion_info["peak_times"]:
                log.info(
                    "[motion_sync] %d peaks driving entry-animation choice",
                    len(motion_info["peak_times"]),
                )
            else:
                log.info("[motion_sync] no peaks detected; falling through to beat/strength")
                motion_info = None
        except Exception as exc:  # noqa: BLE001
            log.warning("motion detection failed (%s); continuing without it", exc)
            motion_info = None

    # ---- v13: VLM gesture anchoring (one per motion peak) ----------------
    # Gesture events become first-class decorations placed at the peak time
    # with the gesture-implied tag + animation. Lyric-driven decorations
    # whose tag matches a gesture event within ±0.5s are deduped later.
    gesture_events: list[GestureEvent] = []
    if vlm_gestures and video_path and motion_info:
        try:
            gi = detect_gestures(
                str(video_path),
                motion_info["peak_times"],
                model=vlm_model,
            )
            gesture_events = gi["events"]
            log.info(
                "[vlm_gesture] %d gesture events from %d peaks (cache=%s)",
                len(gesture_events),
                len(motion_info["peak_times"]),
                gi["cache_hit"],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "VLM gesture detection failed (%s); continuing without it", exc,
            )
            gesture_events = []

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

    # ---- v10 layout pre-pass ---------------------------------------------
    # 1) Pre-resolve every subtitle's pixel rect so the forbidden-map
    #    knows where the lyrics will land before placing decorations.
    outlined_cfg = preset.get("subtitle_outlined", {})
    subtitle_rects: list[tuple[float, float, int, int, int, int]] = []  # (st, et, x, y, w, h)
    pre_subtitles: list[Highlight | None] = [None] * len(highlights)  # holds the SubtitleOutlinedElement we'll emit
    canvas_pose_masks = _scaled_person_masks(person_masks, canvas_size)

    if sub_mode in ("outlined", "banner"):
        for i, hl in enumerate(highlights):
            hl_start = max(0.0, hl.time)
            hl_end = hl_start + _highlight_duration(hl, highlights)
            if sub_mode == "outlined":
                provisional_sub = SubtitleOutlinedElement(
                    content=hl.text,
                    start_time=hl_start,
                    end_time=hl_end,
                    position="top_banner",
                    font=outlined_cfg.get("font", DEFAULT_FONT),
                    size=outlined_cfg.get("size", 64),
                    text_color=outlined_cfg.get("text_color", palette_halo),
                    outline_color=outlined_cfg.get("outline_color", palette_primary),
                    outline_width=outlined_cfg.get("outline_width", 6),
                    shadow_offset=outlined_cfg.get("shadow_offset", 2),
                    shadow_alpha=outlined_cfg.get("shadow_alpha", 120),
                    reasoning=hl.reasoning or "v10 outlined subtitle",
                )
                fitted_sub = fit_subtitle_outlined_to_canvas(
                    provisional_sub, fonts_dir, canvas_size,
                )
                # Pose-aware top vs bottom (reuses the v9 banner heuristic).
                sub_w, sub_h = measure_subtitle_outlined(
                    fitted_sub, fonts_dir, canvas_size,
                )
                pose_mask = canvas_pose_masks.get(
                    min(canvas_pose_masks, key=lambda k: abs(k - hl.time))
                ) if canvas_pose_masks else None
                pos = _pick_banner_position(sub_h, canvas_size, pose_mask)
                fitted_sub = fitted_sub.model_copy(update={"position": pos})
                # Per-line outline alternation for baseline_kenpa-style preset.
                alt_color = outlined_cfg.get("outline_color_alt")
                if alt_color and i % 2 == 1:
                    fitted_sub = fitted_sub.model_copy(update={"outline_color": alt_color})
                anchor_x, anchor_y = resolve_subtitle_outlined_anchor(
                    fitted_sub, fonts_dir, canvas_size,
                )
                pre_subtitles[i] = fitted_sub
                subtitle_rects.append(
                    (hl_start, hl_end, anchor_x, anchor_y, sub_w, sub_h)
                )

    # ---- emit subtitles in the order we just computed --------------------
    subtitle_index_for_highlight: list[int | None] = [None] * len(highlights)
    if sub_mode == "outlined":
        for i, sub in enumerate(pre_subtitles):
            if sub is None:
                continue
            subtitle_index_for_highlight[i] = len(elements)
            elements.append(sub)

    # ---- per-highlight emission (text in hero mode, decorations always) -
    corner_zones = _corner_zones(canvas_size)
    layout_stats = {"placed": 0, "shrunk": 0, "skipped": 0}

    for i, hl in enumerate(highlights):
        start_time = max(0.0, hl.time)
        end_time = start_time + _highlight_duration(hl, highlights)

        if beat_info:
            hl_is_downbeat = is_downbeat(hl.time, beat_info["downbeat_times"])
            hl_in_chorus = is_high_energy(hl.time, beat_info["high_energy_segments"])
        else:
            hl_is_downbeat = False
            hl_in_chorus = False

        # v12: motion-peak intensity bucket ("high"/"medium"/"low"/None).
        # When set, _pick_entry routes through MOTION_ENTRY_* pools instead
        # of the strength/downbeat fallback chain.
        hl_motion = motion_intensity_at(hl.time, motion_info) if motion_info else None

        text_idx: int | None = subtitle_index_for_highlight[i]

        # Banner-mode + hero-mode legacy paths preserved -------------------
        if sub_mode == "banner":
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
                reasoning=hl.reasoning or "v10 banner (legacy)",
            )
            elements.append(banner)
            size = banner_cfg.get("size", 42)
        elif sub_mode == "hero":
            font = DEFAULT_FONT if hl.strength >= 0.6 else DEFAULT_BODY_FONT
            size = _text_size_for_strength(hl.strength)
            if hl.strength >= 0.7:
                color = palette_accent
                outline_color = palette_halo
            else:
                color = palette_primary
                outline_color = palette_outline
            provisional = TextElement(
                content=hl.text, start_time=start_time, end_time=end_time,
                anchor=(0, 0), font=font, size=size, color=color,
                outline_color=outline_color, outline_width=5,
                outline_layers=[OutlineLayer(color=palette_halo, width=4)],
                shadow_offset=(3, 3) if hl.strength >= 0.7 else None,
                animation=_pick_entry(hl.strength, rng, hl.entry_animation,
                                      is_downbeat_hit=hl_is_downbeat,
                                      motion_intensity=hl_motion),
                idle_animation=_pick_idle(hl.strength, rng, hl.idle_animation,
                                          in_chorus=hl_in_chorus),
                rotation_jitter=2.0,
                reasoning=hl.reasoning or "v10 hero-mode floating text",
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
                    "Highlight %d (%r) found no placement; falling back to upper-left.",
                    i, hl.text,
                )
                position = (16, 16)
            text_idx = len(elements)
            elements.append(fitted.model_copy(update={"anchor": position}))
        else:
            # outlined mode — subtitle already emitted in pre-pass
            size = outlined_cfg.get("size", 64)

        # ---- v10 decoration placement via ForbiddenMap ------------------
        max_tags_per_line = 4 if hl_in_chorus else 2
        for tag_pos, tag in enumerate(hl.tags[:max_tags_per_line]):
            # Hero (first tag) prefers a corner zone; ambient tags free-place.
            is_hero_dec = tag_pos == 0
            target_size_px = (
                int(size * 1.6) if is_hero_dec else int(size * 1.0)
            )
            target_size_tuple = (target_size_px, target_size_px)

            fmap = build_forbidden_map_at_time(
                hl.time + 0.2,
                person_masks=canvas_pose_masks,
                subtitle_rects=subtitle_rects,
                canvas_size=canvas_size,
            )

            prefer = rng.choice(corner_zones) if is_hero_dec else None
            anchor = fmap.find_free_position(
                target_size_tuple, prefer_zone=prefer, rng=rng,
            )
            if anchor is None:
                # First retry: shrink to half size.
                fallback = (target_size_px // 2, target_size_px // 2)
                anchor = fmap.find_free_position(fallback, rng=rng)
                if anchor is not None:
                    target_size_px = fallback[0]
                    layout_stats["shrunk"] += 1
                    log.warning(
                        "Highlight %d / tag %r: shrunk decoration to %dpx "
                        "(coverage=%.0f%%)",
                        i, tag, target_size_px, fmap.coverage_pct() * 100,
                    )
            if anchor is None:
                layout_stats["skipped"] += 1
                log.warning(
                    "Highlight %d / tag %r: SKIPPED (no free space, "
                    "coverage=%.0f%%)",
                    i, tag, fmap.coverage_pct() * 100,
                )
                continue
            layout_stats["placed"] += 1

            tint_color = decoration_palette[
                (i + tag_pos) % max(1, len(decoration_palette))
            ]
            decoration = DecorationElement(
                asset_tag=tag,
                near_text_id=text_idx,
                start_time=start_time,
                end_time=end_time,
                base_size=target_size_px,
                rotation_jitter=10.0,
                animation=_pick_entry(hl.strength, rng, hl.entry_animation,
                                      is_downbeat_hit=hl_is_downbeat,
                                      motion_intensity=hl_motion),
                idle_animation=_pick_idle(hl.strength, rng, hl.idle_animation,
                                          in_chorus=hl_in_chorus),
                color_tint=[],
                prefer_color_bucket=hl.decoration_color_hint,
                pixel_anchor=anchor,
                reasoning=(
                    f"tag {tag!r} (pri {tag_pos}, "
                    f"{'hero' if is_hero_dec else 'ambient'}) for "
                    f"line {hl.text!r}; placed at {anchor} "
                    f"(forbidden coverage={fmap.coverage_pct()*100:.0f}%); "
                    f"target colour {tint_color}"
                ),
            )
            elements.append(decoration)

    log.info(
        "[layout/v10] decorations placed=%d, shrunk=%d, skipped=%d",
        layout_stats["placed"], layout_stats["shrunk"], layout_stats["skipped"],
    )

    # ---- v13: emit a first-class decoration at each gesture event -------
    # These decorations are anchored to the dancer's actual gesture time,
    # not the lyric time — that's the whole point of v13.
    #
    # v13.1: when the VLM reported a `best_empty_zone`, we honour it and
    # compute a pixel_anchor inside that zone (rng-jittered for variety).
    # Otherwise we leave pixel_anchor=None and the renderer falls back to
    # ForbiddenMap geometric placement.
    if gesture_events:
        gesture_size = max(64, int(canvas_size[0] * 0.18))
        cw, ch = canvas_size
        zone_to_box = {
            "top_left":     (int(cw * 0.05), int(ch * 0.05), int(cw * 0.35), int(ch * 0.20)),
            "top_right":    (int(cw * 0.65), int(ch * 0.05), int(cw * 0.95), int(ch * 0.20)),
            "bottom_left":  (int(cw * 0.05), int(ch * 0.70), int(cw * 0.35), int(ch * 0.85)),
            "bottom_right": (int(cw * 0.65), int(ch * 0.70), int(cw * 0.95), int(ch * 0.85)),
        }
        for ev in gesture_events:
            if ev.tag is None:
                continue
            pixel_anchor = None
            if ev.zone and ev.zone in zone_to_box:
                x1, y1, x2, y2 = zone_to_box[ev.zone]
                # Centre-of-zone minus half-size, clamped to canvas.
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                ax = max(8, min(cw - gesture_size - 8, cx - gesture_size // 2))
                ay = max(8, min(ch - gesture_size - 8, cy - gesture_size // 2))
                pixel_anchor = (ax, ay)
            elements.append(DecorationElement(
                asset_tag=ev.tag,
                start_time=ev.time,
                end_time=ev.time + 2.0,
                base_size=gesture_size,
                rotation_jitter=10.0,
                animation=ev.animation or "scale_pop",
                idle_animation="pulse",
                color_tint=[],
                pixel_anchor=pixel_anchor,
                reasoning=(
                    f"v13 gesture={ev.gesture!r} (conf={ev.confidence:.2f}, "
                    f"zone={ev.zone or 'auto'}) at peak {ev.time:.2f}s"
                    + (f" — action: {ev.action!r}" if ev.action else "")
                ),
            ))

        # Dedup: if a lyric-driven decoration uses the same tag within
        # 0.5s of a gesture event, the gesture wins (better timing).
        to_drop: set[int] = set()
        for ev in gesture_events:
            if ev.tag is None:
                continue
            for i, el in enumerate(elements):
                if not isinstance(el, DecorationElement):
                    continue
                if (el.reasoning or "").startswith("v13 gesture="):
                    continue  # never dedup gesture-spawned ones against each other
                if el.asset_tag == ev.tag and abs(el.start_time - ev.time) < 0.5:
                    to_drop.add(i)
        if to_drop:
            elements = [el for i, el in enumerate(elements) if i not in to_drop]
            log.info(
                "[vlm_gesture] deduped %d lyric-driven decorations "
                "(gesture took precedence)", len(to_drop),
            )

    # ------- global style -------
    vibe_extras = ""
    if beat_info:
        vibe_extras = (
            f" + beat_sync (tempo={beat_info['tempo']:.1f} BPM, "
            f"{snap_count}/{len(highlights)} snapped)"
        )
    if motion_info:
        vibe_extras += f" + motion_sync ({len(motion_info['peak_times'])} peaks)"
    if gesture_events:
        vibe_extras += f" + vlm_gesture ({len(gesture_events)} events)"
    return Decision(
        elements=elements,
        global_style=GlobalStyle(
            color_palette=decoration_palette + [palette_outline],
            vibe=(
                f"v9 style={style_name!r} subtitle={sub_mode!r} — "
                "lyrics-driven, pose-aware text + colour-balanced decorations"
                f"{vibe_extras}"
            ),
            # Pulse one full breath every two beats — gives roughly 1 Hz at
            # 120 BPM which reads as "alive" without being seasick-fast.
            beat_period_sec=(beat_period * 2) if beat_period else None,
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
            "animation": _pick_entry(hl.strength, rng, hl.entry_animation),
            "idle_animation": _pick_idle(hl.strength, rng, hl.idle_animation),
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
                "animation": _pick_entry(hl.strength, rng, hl.entry_animation),
                "idle_animation": _pick_idle(hl.strength, rng, hl.idle_animation),
                "color_tint": [palette_primary if hl.strength < 0.7 else palette_accent],
                "start_time": start_time,
                "end_time": end_time,
                "reasoning": f"semantic_align tag={tag!r} for line {hl.text!r}",
            })

    return elements
