"""Heuristic fallback Decision generator.

Used when no API key is available, for offline iteration, or as a sanity
floor for the LLM clients. The Decision it produces is intentionally
conservative: a small number of well-timed text elements + a few
decorations clustered on the chorus.

This is NOT how the system should be run in production — the LLM is what
gives the overlays their semantic relevance. But "no API key" should not
mean "no demo".
"""

from __future__ import annotations

import random

from semanticvibe.config import STYLE_PRESETS
from semanticvibe.schemas.decision import (
    Decision,
    DecorationElement,
    GlobalStyle,
    TextElement,
)
from semanticvibe.schemas.feature_summary import FeatureSummary

DEFAULT_TITLE_FONT = "KleeOne-SemiBold"
DEFAULT_BODY_FONT = "KleeOne-Regular"


def _palette_for(style_preset: str) -> tuple[list[str], str]:
    preset = STYLE_PRESETS.get(style_preset)
    if preset is not None:
        return preset["palette"], preset["vibe"]
    # Unknown preset — use the warm hand-drawn defaults.
    fallback = STYLE_PRESETS["warm_handdrawn"]
    return fallback["palette"], fallback["vibe"]


def heuristic_decision(summary: FeatureSummary) -> Decision:
    """Build a Decision from `summary` using only deterministic rules.

    Rules:
    - Take up to 3 lyric lines — first, middle, and the one nearest to the
      first chorus start (when present). These render as text overlays.
    - Drop one decoration ("sparkle") on the title and a second ("star")
      on the chorus opener if a chorus was found.
    """
    palette, vibe = _palette_for(summary.style_preset)
    rng = random.Random(0xC0FFEE)

    elements: list = []

    used_indices: set[int] = set()

    def _pick_distinct(target_idx: int) -> int | None:
        """Pick a lyric index near `target_idx` that hasn't been used yet."""
        if not summary.lyrics:
            return None
        order = sorted(
            range(len(summary.lyrics)),
            key=lambda i: (i in used_indices, abs(i - target_idx)),
        )
        for i in order:
            if i not in used_indices:
                used_indices.add(i)
                return i
        return None

    n_lyrics = len(summary.lyrics)

    if n_lyrics:
        # Title: first non-empty lyric.
        title_idx = _pick_distinct(0)
        title = summary.lyrics[title_idx]
        title_end = min(title.time + 4.0, summary.video_duration)
        elements.append(
            TextElement(
                content=title.text[:14],
                start_time=max(0.5, title.time),
                end_time=title_end,
                anchor="auto",
                font=DEFAULT_TITLE_FONT,
                size=96,
                color=palette[0],
                outline_color=palette[2 % len(palette)],
                outline_width=6,
                animation="bounce_in",
                rotation_jitter=rng.uniform(-2.0, 2.0),
                reasoning="Title hook on the first lyric line — bounce-in lands the entry beat.",
            )
        )

        # Mid-song line — pick the lyric closest to the song midpoint.
        if n_lyrics > 2:
            mid_idx = _pick_distinct(n_lyrics // 2)
            if mid_idx is not None:
                mid = summary.lyrics[mid_idx]
                mid_end = min(mid.time + 3.5, summary.video_duration)
                elements.append(
                    TextElement(
                        content=mid.text[:12],
                        start_time=mid.time,
                        end_time=mid_end,
                        anchor="auto",
                        font=DEFAULT_BODY_FONT,
                        size=64,
                        color=palette[1 % len(palette)],
                        outline_color=palette[2 % len(palette)],
                        outline_width=4,
                        animation="typewriter",
                        rotation_jitter=0,
                        reasoning="Mid-song lyric reveal — typewriter follows the cadence.",
                    )
                )

        # Chorus-anchored line — but only if the chorus actually starts AFTER
        # the title. If the heuristic chorus segmenter returned the whole
        # song (common on instrumental loops), skip — there's no specific
        # opener to highlight.
        if (
            summary.chorus_segments
            and n_lyrics > 3
            and summary.chorus_segments[0][0] > title.time + 5.0
        ):
            cs, _ce = summary.chorus_segments[0]
            # Find lyric nearest the chorus start, excluding already-used ones.
            best_i = min(
                (i for i in range(n_lyrics) if i not in used_indices),
                key=lambda i: abs(summary.lyrics[i].time - cs),
                default=None,
            )
            if best_i is not None:
                used_indices.add(best_i)
                chorus_lyric = summary.lyrics[best_i]
                ch_end = min(chorus_lyric.time + 4.0, summary.video_duration)
                elements.append(
                    TextElement(
                        content=chorus_lyric.text[:12],
                        start_time=chorus_lyric.time,
                        end_time=ch_end,
                        anchor="auto",
                        font=DEFAULT_TITLE_FONT,
                        size=80,
                        color=palette[3 % len(palette)],
                        outline_color=palette[2 % len(palette)],
                        outline_width=5,
                        animation="wiggle",
                        rotation_jitter=rng.uniform(-3.0, 3.0),
                        reasoning="Chorus emphasis — wiggle gives playful energy on the hook.",
                    )
                )

        # Outro line — last lyric, if it didn't land too close to the others.
        if n_lyrics >= 4:
            outro_idx = _pick_distinct(n_lyrics - 1)
            if outro_idx is not None:
                outro = summary.lyrics[outro_idx]
                ou_end = min(outro.time + 3.0, summary.video_duration)
                if ou_end > outro.time:
                    elements.append(
                        TextElement(
                            content=outro.text[:12],
                            start_time=outro.time,
                            end_time=ou_end,
                            anchor="auto",
                            font=DEFAULT_BODY_FONT,
                            size=64,
                            color=palette[4 % len(palette)],
                            outline_color=palette[2 % len(palette)],
                            outline_width=4,
                            animation="fade",
                            rotation_jitter=0,
                            reasoning="Outro line — fade lets the song breathe.",
                        )
                    )

    # Decorations: sparkle on the title, star on chorus if present.
    if elements:
        title_idx = 0
        elements.append(
            DecorationElement(
                asset_tag="sparkle",
                near_text_id=title_idx,
                start_time=elements[title_idx].start_time,
                end_time=elements[title_idx].end_time,
                scale_jitter=0.15,
                rotation_jitter=12.0,
                reasoning="Sparkle marks the title entry.",
            )
        )

    if summary.chorus_segments:
        cs, ce = summary.chorus_segments[0]
        # Clamp against video_duration — Whisper / librosa segmentation can
        # over-shoot past the end of the audio track on noisy material.
        cs = min(cs, max(0.0, summary.video_duration - 0.5))
        end = min(cs + 4.0, ce, summary.video_duration)
        if end > cs:
            elements.append(
                DecorationElement(
                    asset_tag="star",
                    near_text_id=None,
                    start_time=cs,
                    end_time=end,
                    scale_jitter=0.1,
                    rotation_jitter=8.0,
                    reasoning="Star highlights the chorus opener; falls back to safe-zone anchor.",
                )
            )

    if not elements:
        # Truly empty input — emit a single placeholder so the render still produces overlays.
        elements.append(
            TextElement(
                content="片刻",
                start_time=1.0,
                end_time=4.0,
                anchor="auto",
                font=DEFAULT_TITLE_FONT,
                size=96,
                color=palette[0],
                outline_color=palette[2 % len(palette)],
                outline_width=6,
                animation="fade",
                rotation_jitter=0,
                reasoning="Placeholder — no lyrics or beats detected.",
            )
        )

    return Decision(
        elements=elements,
        global_style=GlobalStyle(color_palette=list(palette), vibe=vibe),
    )
