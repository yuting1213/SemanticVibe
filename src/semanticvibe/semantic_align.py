"""Lyrics → highlights with decoration tags.

Walks a list of `(time, text)` lyric tuples and produces `Highlight`
records — each picks a phrase to overlay on screen and (optionally)
which decoration from the asset library to place alongside it.

Three providers, all with the same `align(lyrics) -> list[Highlight]`
signature:

- **rule_based** (default, offline) — keyword matching against
  TAG_VOCABULARY. Cheap, deterministic, multilingual.
- **claude** — sends lyrics + tag vocab to Claude (Haiku 4.5 in dev
  mode), which picks the most evocative phrases and assigns tags.
- **openai** — same flow via GPT-4o-mini.

This module is the v5 successor to the old `llm.heuristic` /
`llm.client.decide` pair: it produces structured `Highlight`s that
`build_elements` then assembles into a `Decision`. The point of the
indirection is to keep "what to say" (alignment) separate from "where
to put it" (layout) — which is what makes auto person-avoidance
tractable.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

Provider = Literal["rule_based", "claude", "openai"]


# ---------------------------------------------------------------------------
# Tag vocabulary — keyword → asset_tag mapping
# ---------------------------------------------------------------------------
#
# Each tag corresponds to a sticker filename in data/assets_lib/. Triggers
# are matched case-insensitively as substrings. Multilingual on purpose:
# Mandarin / Cantonese / Japanese / English overlap a lot in idol-pop
# lyrics so we just throw all spellings of the same idea into one bucket.
#
# When a lyric phrase matches multiple tags, the first one wins (priority
# ordering is the dict iteration order). To bias toward "wow" tags
# (lightning / fire / impact) put those above "soft" tags (heart / flower).

TAG_VOCABULARY: dict[str, list[str]] = {
    # Tag must exist in data/assets_lib/metadata.json — otherwise the
    # render path silently drops the decoration. The current procedural
    # asset set is: heart / mini_heart / sparkle / star / dot / burst /
    # arrow / fire / exclaim. The vocabulary maps cross-language semantic
    # buckets onto those available tags. Lightning / flower / etc. that
    # don't have an asset get aliased to the closest visual match
    # (lightning → exclaim, since both are "energy / impact" stickers).

    # Wow / energy
    "exclaim":   ["wow", "ah", "oh", "わあ", "やった", "impact", "bam", "drop",
                  # lightning / 電波 / 雷 — alias to exclaim (jagged impact star).
                  "lightning", "電波", "雷", "閃電", "電", "shock", "thunder"],
    "fire":      ["fire", "火", "flame", "炎", "情熱", "燃", "hot", "spicy", "やばい"],
    "burst":     ["burst", "爆", "彈", "pop", "celebration"],
    # Affection / cuteness
    "heart":     ["heart", "好き", "愛", "love", "戀", "愛してる", "心", "嬉しい"],
    "mini-heart":["可愛い", "kawaii", "cute", "cutie", "可愛", "lovely",
                  # flower / 花 / 桜 — alias to mini-heart since we have no flower asset.
                  "flower", "花", "blossom", "桜", "petal", "甜"],
    # Reflection / softer notes
    "star":      ["star", "星", "twinkle", "輝", "夢", "dream",
                  # moon / 月 / cloud — alias to star (closest "celestial" sticker).
                  "moon", "月", "夜", "night", "midnight",
                  "cloud", "雲", "sky", "空"],
    "sparkle":   ["sparkle", "shine", "kira", "キラ", "shimmer", "春"],
    "arrow":     ["arrow", "→", "look", "here", "this", "向"],
    "dot":       ["dot", "spot", "point", "圓"],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LyricLine:
    time: float
    text: str


@dataclass
class Highlight:
    """A single overlay-worthy moment.

    `decoration_tag` is what we'll feed to the asset library; None means
    no decoration (just the text). `strength` is 0–1 and roughly maps to
    "how punchy should the entry animation be" — used downstream by
    build_elements to pick scale_pop / fade / etc.
    """

    lyric_time: float
    lyric_text: str
    decoration_tag: str | None = None
    strength: float = 0.5
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Provider: rule_based
# ---------------------------------------------------------------------------


def _find_tag(text: str) -> str | None:
    """Return the TAG_VOCABULARY tag whose longest trigger appears in `text`.

    Longest-match wins: 「可愛い」 (5 chars) beats heart's 「愛」 (1 char) so
    'kawaii' lyrics resolve to mini-heart instead of generic heart. Without
    this, dict iteration order silently steals more-specific tags away from
    the right bucket.
    """
    lowered = text.lower()
    # Build (tag, trigger, len) tuples sorted by trigger length, longest first.
    candidates = [
        (tag, trigger.lower())
        for tag, triggers in TAG_VOCABULARY.items()
        for trigger in triggers
    ]
    candidates.sort(key=lambda tt: len(tt[1]), reverse=True)
    for tag, trigger in candidates:
        if trigger and trigger in lowered:
            return tag
    return None


def _rule_based_align(lyrics: list[LyricLine]) -> list[Highlight]:
    """Walk lyrics, emit a Highlight for every line.

    Each line gets:
    - decoration_tag from TAG_VOCABULARY if any keyword matches, else None
    - strength = 0.8 if a tag matched (it's evocative), else 0.4 (filler)
    """
    highlights: list[Highlight] = []
    for line in lyrics:
        tag = _find_tag(line.text)
        highlights.append(
            Highlight(
                lyric_time=line.time,
                lyric_text=line.text,
                decoration_tag=tag,
                strength=0.8 if tag else 0.4,
                reasoning=(
                    f"keyword match → tag={tag!r}" if tag else "no keyword match"
                ),
            )
        )
    return highlights


# ---------------------------------------------------------------------------
# Provider: claude / openai
# ---------------------------------------------------------------------------


_LLM_SYSTEM_PROMPT = """You are an art director for animated lyric videos.

You receive a JSON list of lyric lines, each with a timestamp and text. You
must pick which lines are worth highlighting on screen and assign each one
a decoration tag from the provided vocabulary.

Constraints:
1. Output JSON only. No prose, no explanations outside the schema.
2. For each input lyric line, output one Highlight object:
   - lyric_time: copy from input
   - lyric_text: the text to display (you may shorten if the original line
     is too long — aim for ≤ 8 characters for impact)
   - decoration_tag: a tag from the vocabulary, or null if no tag fits
   - strength: 0.0 to 1.0 — 1.0 for the song's most punchy moments,
     0.3 for filler. Roughly 1/3 of lines should be ≥ 0.7.
   - reasoning: one sentence on why this line + tag.

Vocabulary (only these tags are valid; null is also valid):
{vocab_summary}

Output schema:
{
  "highlights": [
    {"lyric_time": 2.5, "lyric_text": "...", "decoration_tag": "heart",
     "strength": 0.8, "reasoning": "..."},
    ...
  ]
}
"""


def _llm_align(
    lyrics: list[LyricLine],
    *,
    provider: Provider,
) -> list[Highlight]:
    """Send lyrics + tag vocabulary to an LLM, parse Highlights from JSON."""
    if provider == "claude":
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        client = Anthropic(api_key=api_key)

        vocab_summary = "\n".join(
            f"  - {tag}: {', '.join(triggers[:5])}"
            for tag, triggers in TAG_VOCABULARY.items()
        )
        sys_prompt = _LLM_SYSTEM_PROMPT.format(vocab_summary=vocab_summary)
        user_prompt = "Lyrics:\n" + json.dumps(
            [{"time": L.time, "text": L.text} for L in lyrics],
            ensure_ascii=False,
            indent=2,
        )

        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=2048,
            system=sys_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        return _parse_highlights_json(text)

    if provider == "openai":
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key)

        vocab_summary = "\n".join(
            f"  - {tag}: {', '.join(triggers[:5])}"
            for tag, triggers in TAG_VOCABULARY.items()
        )
        sys_prompt = _LLM_SYSTEM_PROMPT.format(vocab_summary=vocab_summary)
        user_prompt = "Lyrics:\n" + json.dumps(
            [{"time": L.time, "text": L.text} for L in lyrics],
            ensure_ascii=False,
            indent=2,
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return _parse_highlights_json(response.choices[0].message.content or "")

    raise ValueError(f"unknown provider: {provider!r}")


def _parse_highlights_json(raw: str) -> list[Highlight]:
    """Tolerant JSON parser — strips ```json fences, finds first { ... }."""
    text = raw.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences.
        text = text.strip("`").lstrip("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Find the outermost {...} via brace counting.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise
        data = json.loads(text[start : end + 1])

    out: list[Highlight] = []
    for h in data.get("highlights", []):
        out.append(
            Highlight(
                lyric_time=float(h["lyric_time"]),
                lyric_text=str(h["lyric_text"]),
                decoration_tag=h.get("decoration_tag"),
                strength=float(h.get("strength", 0.5)),
                reasoning=str(h.get("reasoning", "")),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def align(
    lyrics: list[LyricLine] | list[dict],
    *,
    provider: Provider = "rule_based",
) -> list[Highlight]:
    """Align lyrics → list of Highlight.

    Accepts either pre-parsed LyricLine objects or raw dicts of
    `{"time": float, "text": str}` for convenience.
    """
    parsed: list[LyricLine] = []
    for item in lyrics:
        if isinstance(item, LyricLine):
            parsed.append(item)
        else:
            parsed.append(LyricLine(time=float(item["time"]), text=str(item["text"])))

    if provider == "rule_based":
        return _rule_based_align(parsed)
    if provider in ("claude", "openai"):
        try:
            return _llm_align(parsed, provider=provider)
        except Exception as exc:  # noqa: BLE001 — fall back to rule_based on any LLM failure
            log.warning(
                "LLM provider %s failed (%s); falling back to rule_based.",
                provider, exc,
            )
            return _rule_based_align(parsed)
    raise ValueError(f"unknown provider: {provider!r}")


def load_lyrics(path: Path) -> list[LyricLine]:
    """Convenience loader for samples/lyrics_*.json files."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [LyricLine(time=float(r["time"]), text=str(r["text"])) for r in raw]
