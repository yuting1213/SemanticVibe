"""Lyrics → AlignmentResult (highlights + non-hooks).

v6 rewrite. Closed-vocabulary semantic alignment:

- The tag set is loaded once from `assets/tag_vocabulary.json` and is
  authoritative. The LLM is told to pick *only* from this list; rule-based
  matches the same set. Anything else falls back to FALLBACK_TAG.
- `Highlight` is a Pydantic model carrying `tags: list[str]` and a
  `primary_tag` so a single line can imply multiple decorations
  (e.g. 「電波好き」 → [lightning, heart]).
- `AlignmentResult` separates "highlights" (lines worth painting on screen)
  from "non_hooks" (filler — the LLM saw them but chose not to highlight).
- Two providers:
    * `rule_based` — offline keyword dict (`KEYWORD_TO_TAGS`).
    * `claude`     — strict-JSON Claude call, MD5-cached at
                     `.cache/alignment/<key>.json` so repeated renders cost
                     zero. Cache key = md5(model + lyrics + song_title +
                     vocab fingerprint).

The legacy `align(...)` function (returning `list[Highlight]`-as-dataclass)
is kept as a thin shim over `align_lyrics(...)` so the existing CLI keeps
working until callers migrate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from semanticvibe.lyrics import LyricLine

log = logging.getLogger(__name__)

Provider = Literal["rule_based", "claude", "ollama"]


# ---------------------------------------------------------------------------
# Closed tag vocabulary — single source of truth, loaded from JSON
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_VOCAB_PATH = _REPO_ROOT / "assets" / "tag_vocabulary.json"


def _load_vocab() -> tuple[list[dict], str]:
    """Returns (tag_records, fallback_tag). Errors hard if file is missing —
    the closed vocab is not optional for v6."""
    if not _VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"tag vocabulary not found at {_VOCAB_PATH}; run from a complete checkout."
        )
    with _VOCAB_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)
    return list(raw["tags"]), str(raw.get("fallback_tag", "heart"))


VOCAB, FALLBACK_TAG = _load_vocab()
VALID_TAGS: set[str] = {t["id"] for t in VOCAB}
TAG_CATEGORY: dict[str, str] = {t["id"]: t["category"] for t in VOCAB}


# ---------------------------------------------------------------------------
# rule_based keyword → tags dictionary
# ---------------------------------------------------------------------------
#
# Multi-tag on purpose: a single phrase can imply more than one sticker,
# and primary_tag is just `tags[0]`. Triggers are matched as case-insensitive
# substrings; longest-matching trigger across all keys wins so 「可愛い」 (kawaii)
# doesn't get poached by 「愛」 (love).
#
# Every tag here MUST be in VALID_TAGS — defensive check at module load.

KEYWORD_TO_TAGS: dict[str, list[str]] = {
    # ----- emotion -----
    "好き":      ["heart"],
    "愛":        ["heart"],
    "love":      ["heart"],
    "戀":        ["heart"],
    "心":        ["heart"],
    "嬉しい":     ["heart"],
    "可愛い":     ["heart", "sparkle"],
    "kawaii":    ["heart", "sparkle"],
    "cute":      ["heart"],
    "tear":      ["teardrop"],
    "cry":       ["teardrop"],
    "涙":        ["teardrop"],
    "泣":        ["teardrop"],
    "悲":        ["teardrop"],
    "sad":       ["teardrop"],
    "kiss":      ["kiss"],
    "唇":        ["kiss"],
    "ちゅ":       ["kiss"],

    # ----- decorative -----
    "sparkle":   ["sparkle"],
    "shine":     ["sparkle"],
    "kira":      ["sparkle"],
    "キラ":       ["sparkle"],
    "shimmer":   ["sparkle"],
    "輝":        ["sparkle"],
    "ribbon":    ["ribbon"],
    "bow":       ["ribbon"],
    "リボン":     ["ribbon"],
    "flower":    ["flower"],
    "花":        ["flower"],
    "桜":        ["flower"],
    "petal":     ["flower"],
    "blossom":   ["flower"],
    "star":      ["star"],
    "星":        ["star"],
    "twinkle":   ["star", "sparkle"],
    "夢":        ["star"],
    "dream":     ["star"],

    # ----- energy -----
    "fire":      ["fire"],
    "火":        ["fire"],
    "炎":        ["fire"],
    "flame":     ["fire"],
    "情熱":      ["fire"],
    "燃":        ["fire"],
    "hot":       ["fire"],
    "lightning": ["lightning"],
    "電波":      ["lightning"],
    "雷":        ["lightning"],
    "閃電":      ["lightning"],
    "shock":     ["lightning"],
    "thunder":   ["lightning"],
    "burst":     ["burst"],
    "爆":        ["burst"],
    "彈":        ["burst"],
    "pop":       ["burst"],

    # ----- emphasis -----
    "wow":       ["exclaim"],
    "ah":        ["exclaim"],
    "oh":        ["exclaim"],
    "わあ":       ["exclaim"],
    "やった":     ["exclaim"],
    "impact":    ["exclaim"],
    "やばい":     ["exclaim", "fire"],
    "!":         ["exclaim"],
    "!":         ["exclaim"],
    "dot":       ["dot"],

    # ----- weather -----
    "sun":       ["sun"],
    "太陽":      ["sun"],
    "陽":        ["sun"],
    "暖":        ["sun"],
    "moon":      ["moon"],
    "月":        ["moon"],
    "夜":        ["moon"],
    "midnight":  ["moon"],
    "night":     ["moon"],
    "cloud":     ["cloud"],
    "雲":        ["cloud"],
    "sky":       ["cloud"],
    "空":        ["cloud"],
    "rainbow":   ["rainbow"],
    "虹":        ["rainbow"],

    # ----- nature -----
    "leaf":      ["leaf"],
    "葉":        ["leaf"],
    "草":        ["leaf"],
    "green":     ["leaf"],

    # ----- audio -----
    "歌":        ["music_note"],
    "music":     ["music_note"],
    "song":      ["music_note"],
    "sing":      ["music_note"],
    "melody":    ["music_note"],
    "beat":      ["music_note"],
    "♪":         ["music_note"],
    "♫":         ["music_note"],

    # ----- communication -----
    "もしもし":   ["speech_bubble"],
    "hello":     ["speech_bubble"],
    "hi":        ["speech_bubble"],
    "hey":       ["speech_bubble"],
    "你好":      ["speech_bubble"],
    "話":        ["speech_bubble"],
    "言":        ["speech_bubble"],
    "arrow":     ["arrow"],
    "→":          ["arrow"],
    "look":      ["arrow"],

    # ----- animals -----
    "cat":       ["animal"],
    "dog":       ["animal"],
    "bear":      ["animal"],
    "bird":      ["animal"],
    "fish":      ["animal"],
    "frog":      ["animal"],
    "猫":        ["animal"],
    "犬":        ["animal"],
    "鳥":        ["animal"],
    "魚":        ["animal"],
    "貓":        ["animal"],
    "鳥兒":      ["animal"],
    "ペット":    ["animal"],
    "pet":       ["animal"],

    # ----- food (general) -----
    "food":      ["food"],
    "eat":       ["food"],
    "meal":      ["food"],
    "bread":     ["food"],
    "cheese":    ["food"],
    "sushi":     ["food"],
    "snack":     ["food"],
    "飯":        ["food"],
    "食":        ["food"],
    "ご飯":       ["food"],

    # ----- fruit -----
    "fruit":     ["fruit"],
    "apple":     ["fruit"],
    "cherry":    ["fruit"],
    "grapes":    ["fruit"],
    "banana":    ["fruit"],
    "蘋果":      ["fruit"],
    "果":        ["fruit"],
    "いちご":     ["fruit"],
    "berry":     ["fruit"],
    "甜美":      ["fruit"],

    # ----- icecream -----
    "ice cream": ["icecream"],
    "icecream":  ["icecream"],
    "popsicle":  ["icecream"],
    "アイス":     ["icecream"],
    "アイスクリーム": ["icecream"],
    "冰淇淋":    ["icecream"],
    "雪糕":      ["icecream"],
    "dessert":   ["icecream"],

    # ----- numbers / typographic emphasis -----
    "1":         ["numbers"],
    "2":         ["numbers"],
    "3":         ["numbers"],
    "4":         ["numbers"],
    "5":         ["numbers"],
    "6":         ["numbers"],
    "7":         ["numbers"],
    "8":         ["numbers"],
    "9":         ["numbers"],
    "0":         ["numbers"],
    "first":     ["numbers"],
    "second":    ["numbers"],
    "count":     ["numbers"],

    # ----- transport -----
    "car":       ["transport"],
    "bus":       ["transport"],
    "train":     ["transport"],
    "plane":     ["transport"],
    "ship":      ["transport"],
    "boat":      ["transport"],
    "bike":      ["transport"],
    "車":        ["transport"],
    "電車":       ["transport"],
    "飛行機":     ["transport"],
    "船":        ["transport"],
    "旅":        ["transport"],
    "journey":   ["transport"],
    "travel":    ["transport"],
}

# Defensive — every tag in KEYWORD_TO_TAGS must be in the closed vocab.
_unknown = {t for tags in KEYWORD_TO_TAGS.values() for t in tags} - VALID_TAGS
if _unknown:
    raise RuntimeError(
        f"KEYWORD_TO_TAGS references unknown tags: {sorted(_unknown)} — "
        f"must be subset of {sorted(VALID_TAGS)}"
    )


# ---------------------------------------------------------------------------
# Pydantic schemas (v6)
# ---------------------------------------------------------------------------


class Highlight(BaseModel):
    """A single overlay-worthy moment."""

    time: float = Field(ge=0)
    text: str = Field(min_length=1)
    is_hook: bool = Field(
        default=False,
        description="True for the song's most punchy / quotable moments. "
        "build_elements emits hero_text only for hooks; non-hooks become "
        "regular text overlays.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Decoration tags applied to this line, in priority order. "
        "All values must be members of the closed vocabulary; primary_tag is "
        "simply tags[0] when non-empty.",
    )
    primary_tag: str | None = None
    reasoning: str = ""
    duration: float | None = Field(default=None, gt=0)
    # ---- v8: LLM-driven animation + colour direction ----
    entry_animation: str | None = Field(
        default=None,
        description="Optional entry-animation hint from the LLM. Must be a "
        "name from the entry registry; otherwise build_decision falls back "
        "to a strength-bucketed random pick.",
    )
    idle_animation: str | None = Field(
        default=None,
        description="Optional idle-animation hint. Must be in the idle "
        "registry, else build_decision falls back to its random pool.",
    )
    decoration_color_hint: str | None = Field(
        default=None,
        description="Optional colour-bucket hint for the picked decoration "
        "PNG (e.g. 'pink', 'green', 'yellow'). The retriever uses this to "
        "narrow the candidate pool; if no PNG matches the hint within the "
        "tag the request falls through to the regular palette-balanced pick.",
    )

    # Backward-compat alias: older callers used .lyric_time / .lyric_text.
    @property
    def lyric_time(self) -> float:
        return self.time

    @property
    def lyric_text(self) -> str:
        return self.text

    @property
    def decoration_tag(self) -> str | None:
        """Compat shim: pre-v6 callers expect a single tag."""
        return self.primary_tag

    @property
    def strength(self) -> float:
        """Compat shim: pre-v6 callers branch on this."""
        return 0.85 if self.is_hook else (0.6 if self.tags else 0.35)


class AlignmentResult(BaseModel):
    """The full alignment output: tagged highlights + ungated lyric lines."""

    highlights: list[Highlight]
    non_hooks: list[str] = Field(
        default_factory=list,
        description="Plain lyric texts that did not get a hook treatment "
        "(no decoration, no hero). build_elements may still display them as "
        "ordinary text overlays.",
    )


# ---------------------------------------------------------------------------
# rule_based provider
# ---------------------------------------------------------------------------


def _match_tags_for_text(text: str) -> list[str]:
    """Return all tags whose triggers appear in `text`, longest-trigger-first.

    Multi-tag: 「電波好き」 → [lightning, heart]. Each tag is reported once.
    """
    lowered = text.lower()
    triggers = sorted(KEYWORD_TO_TAGS.items(), key=lambda kv: -len(kv[0]))
    seen: list[str] = []
    for trig, tags in triggers:
        if trig and trig.lower() in lowered:
            for t in tags:
                if t not in seen:
                    seen.append(t)
    return seen


def _is_hook(text: str, tags: list[str]) -> bool:
    """Heuristic: a line is a hook if it matched ≥1 tag *and* is short.
    Long lines (more than 6 CJK / 16 Latin chars) read more like prose, so
    we keep them as regular highlights even if tagged."""
    if not tags:
        return False
    cjk = sum(1 for c in text if "　" <= c <= "鿿")
    if cjk:
        return cjk <= 6
    return len(text) <= 16


def _rule_based_align(lyrics: list[LyricLine]) -> AlignmentResult:
    highlights: list[Highlight] = []
    non_hooks: list[str] = []
    for line in lyrics:
        tags = _match_tags_for_text(line.text)
        if tags:
            highlights.append(Highlight(
                time=line.time,
                text=line.text,
                is_hook=_is_hook(line.text, tags),
                tags=tags,
                primary_tag=tags[0],
                reasoning=f"keyword match → {tags!r}",
                duration=line.duration,
            ))
        else:
            # Untagged line — keep the text but mark non-hook so downstream
            # can decide whether to render it plainly.
            highlights.append(Highlight(
                time=line.time,
                text=line.text,
                is_hook=False,
                tags=[],
                primary_tag=None,
                reasoning="no keyword match",
                duration=line.duration,
            ))
            non_hooks.append(line.text)
    return AlignmentResult(highlights=highlights, non_hooks=non_hooks)


# ---------------------------------------------------------------------------
# claude provider — strict JSON, MD5-cached
# ---------------------------------------------------------------------------


_CACHE_DIR = _REPO_ROOT / ".cache" / "alignment"


def _vocab_fingerprint() -> str:
    """Stable fingerprint of the closed vocab — busts cache when vocab edits."""
    return hashlib.md5(
        json.dumps(sorted(VALID_TAGS), ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:8]


def _cache_key(
    model: str,
    lyrics: list[LyricLine],
    song_title: str | None,
    *,
    backend: str = "claude",
) -> str:
    payload = {
        "backend": backend,
        "model": model,
        "vocab": _vocab_fingerprint(),
        "song_title": song_title or "",
        "lyrics": [{"time": L.time, "text": L.text, "duration": L.duration} for L in lyrics],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.md5(blob).hexdigest()


def _load_cache(key: str) -> AlignmentResult | None:
    f = _CACHE_DIR / f"{key}.json"
    if not f.exists():
        return None
    try:
        return AlignmentResult.model_validate_json(f.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - corrupt cache, regenerate
        log.warning("cache miss (corrupt): %s (%s)", f.name, exc)
        return None


def _save_cache(key: str, result: AlignmentResult) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (_CACHE_DIR / f"{key}.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8",
    )


_CLAUDE_SYSTEM_PROMPT = """You are an art director for animated lyric music videos.

Given a list of lyric lines and (optionally) a song title, decide which lines
deserve a visual highlight on screen. For each highlight, choose:
1. decoration sticker tags from the CLOSED vocabulary
2. an entry animation that fits the line's emotional moment
3. an idle animation that fits the line's mood
4. (optional) a colour-bucket hint to bias which PNG variant we pick

Output STRICT JSON. No prose, no markdown fences, no explanations outside the
JSON. The schema is:

{
  "highlights": [
    {
      "time":                  <float, copy from input>,
      "text":                  <string, copy or shorten the lyric for impact>,
      "is_hook":               <bool, true for the song's punchiest 1-3 moments>,
      "tags":                  [<tag>, <tag>, ...],   // 0-3 tags from the closed list
      "primary_tag":           <tag or null>,         // null when tags == []
      "entry_animation":       <one of the entry-animation names below>,
      "idle_animation":        <one of the idle-animation names below>,
      "decoration_color_hint": <one of the colour buckets below, or null>,
      "reasoning":             <one sentence on the artistic choices>
    }
  ],
  "non_hooks": [<lyric texts that you did NOT highlight>]
}

Closed sticker vocabulary (id — category — meaning):
{vocab_block}

Entry animations (pick the one that fits the line's *moment*):
  - fade           — soft, no motion. For tender / quiet lines.
  - bounce_in      — playful spring. For cute, light-hearted lines.
  - typewriter     — characters reveal one-by-one. For statements, declarations.
  - draw_in        — line-by-line ink reveal. For deliberate, written-feeling lines.
  - wiggle         — gentle wobble in. For shy / uncertain lines.
  - scale_pop      — pops from small to big with overshoot. For punchy hooks.
  - drop_in        — falls in from above with bounce. For surprising / dramatic lines.
  - slide_in_left  — slides in from left. For "incoming" feeling.
  - slide_in_right — slides in from right.
  - slide_in_top   — slides in from top. Good for sky / heavenly themes.
  - slide_in_bottom— slides in from bottom. Good for emerging / rising themes.
  - stamp          — slams down with shake. For impact / declaration.
  - wobble_in      — wobble + scale up. For excited, off-balance feelings.
  - spin_in        — spins in 360°. For magical / transformative moments.

Idle animations (the steady-state behaviour after entry settles):
  - none           — stays still. For statements you want held firmly.
  - pulse          — subtle scale breathing. Default for "alive" feeling.
  - wiggle         — high-frequency tiny shake. For nervous / energetic lines.
  - drift          — slow horizontal/vertical drift. For floating / dreamy lines.
  - rotate_slow    — continuous slow rotation. For magical / hypnotic loops.
  - shimmer        — opacity flicker. For sparkly / fading-glow feelings.

Decoration colour-bucket hints (optional, narrows the PNG pick):
  - red / pink / orange / yellow / green / cyan / blue / purple
  - white / grey / black / brown
  - null — let the renderer pick the most palette-balanced variant.

Rules:
- Output one highlight object per *highlighted* lyric line. Skip filler.
- ~1 in 3 lines should be marked is_hook=true; the rest are normal highlights.
- primary_tag must be tags[0] when tags is non-empty, else null.
- Every tag in `tags` MUST be a literal id from the closed vocabulary above.
- entry_animation + idle_animation MUST be exact names from the lists above.
- decoration_color_hint must be a single bucket name (lowercase) or null.
- non_hooks holds the texts of lyric lines you decided not to highlight.
- Match animation energy to lyric energy: a love confession deserves
  scale_pop + pulse, a parting sigh deserves fade + drift, a magical line
  deserves spin_in + shimmer, etc. Avoid picking the same combo for every
  line — diversity reads as more thoughtful art direction.
"""


def _format_vocab_block() -> str:
    return "\n".join(
        f"  - {t['id']:<14} — {t['category']:<13} — {t['description']}"
        for t in VOCAB
    )


def _claude_align(
    lyrics: list[LyricLine],
    *,
    song_title: str | None = None,
    model: str = "claude-haiku-4-5",
    use_cache: bool = True,
) -> AlignmentResult:
    key = _cache_key(model, lyrics, song_title, backend="claude")
    if use_cache:
        hit = _load_cache(key)
        if hit is not None:
            log.info("[align/claude] cache hit: %s", key[:8])
            return hit

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    from anthropic import Anthropic
    client = Anthropic(api_key=api_key)

    system = _CLAUDE_SYSTEM_PROMPT.replace("{vocab_block}", _format_vocab_block())
    user_payload = {
        "song_title": song_title or "",
        "lyrics": [{"time": L.time, "text": L.text} for L in lyrics],
    }
    user_msg = (
        "Lyrics to align:\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
        + "\n\nReturn STRICT JSON matching the schema."
    )

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(
        b.text for b in response.content if getattr(b, "type", None) == "text"
    )
    result = _parse_strict_alignment_json(text, lyrics)
    if use_cache:
        _save_cache(key, result)
        log.info("[align/claude] cache store: %s", key[:8])
    return result


# ---------------------------------------------------------------------------
# ollama provider — local LLM via http://localhost:11434, JSON mode
# ---------------------------------------------------------------------------


_OLLAMA_DEFAULT_HOST = "http://localhost:11434"
_OLLAMA_DEFAULT_MODEL = "gemma3:4b"


def _ollama_align(
    lyrics: list[LyricLine],
    *,
    song_title: str | None = None,
    model: str = _OLLAMA_DEFAULT_MODEL,
    host: str | None = None,
    use_cache: bool = True,
    timeout: float = 120.0,
) -> AlignmentResult:
    """Align via a local Ollama instance. Requires `ollama serve` listening
    on `host` (default localhost:11434) with `model` already pulled.

    Uses Ollama's `format: "json"` parameter to constrain output. Falls
    back to the same strict-JSON parser used by the Claude path so the
    cleanup + closed-vocab enforcement is identical.
    """
    import urllib.request
    import urllib.error

    key = _cache_key(model, lyrics, song_title, backend="ollama")
    if use_cache:
        hit = _load_cache(key)
        if hit is not None:
            log.info("[align/ollama] cache hit: %s", key[:8])
            return hit

    host = host or _OLLAMA_DEFAULT_HOST
    system = _CLAUDE_SYSTEM_PROMPT.replace("{vocab_block}", _format_vocab_block())
    user_payload = {
        "song_title": song_title or "",
        "lyrics": [{"time": L.time, "text": L.text} for L in lyrics],
    }
    user_msg = (
        "Lyrics to align:\n"
        + json.dumps(user_payload, ensure_ascii=False, indent=2)
        + "\n\nReturn STRICT JSON matching the schema. Output only JSON, no prose."
    )

    req_body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
            "num_ctx": 8192,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{host}/api/chat",
        data=req_body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Ollama unreachable at {host} ({exc}). Is `ollama serve` running?"
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Ollama timed out after {timeout}s on model {model!r}"
        ) from exc

    text = (payload.get("message") or {}).get("content", "")
    if not text:
        raise RuntimeError(f"Ollama returned no content; raw payload: {payload}")
    result = _parse_strict_alignment_json(text, lyrics)
    if use_cache:
        _save_cache(key, result)
        log.info("[align/ollama] cache store: %s (model=%s)", key[:8], model)
    return result


# Registry-name validation lives here so the parser can scrub LLM picks
# without importing the renderer (avoids a circular import in test paths).
_VALID_ENTRY_ANIMATIONS = {
    "fade", "bounce_in", "typewriter", "draw_in", "wiggle",
    "scale_pop", "drop_in", "slide_in_left", "slide_in_right",
    "slide_in_top", "slide_in_bottom", "stamp", "wobble_in", "spin_in",
}
_VALID_IDLE_ANIMATIONS = {
    "none", "pulse", "wiggle", "drift", "rotate_slow", "shimmer",
}
_VALID_COLOR_BUCKETS = {
    "red", "pink", "orange", "yellow", "green", "cyan", "blue",
    "purple", "white", "grey", "black", "brown",
}


def _parse_strict_alignment_json(
    raw: str, lyrics: list[LyricLine],
) -> AlignmentResult:
    text = raw.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences if the model wrapped its output anyway.
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1:
            raise
        data = json.loads(text[s : e + 1])

    cleaned: list[Highlight] = []
    for h in data.get("highlights", []):
        raw_tags = h.get("tags") or []
        tags = [t for t in raw_tags if isinstance(t, str) and t in VALID_TAGS]
        # Filter out invalid tags but keep the highlight; if the LLM picked
        # *only* invalid tags, fall back to FALLBACK_TAG so the line still
        # gets some visual treatment.
        if raw_tags and not tags:
            log.warning(
                "LLM returned only invalid tags %r for line %r; falling back to %s",
                raw_tags, h.get("text", ""), FALLBACK_TAG,
            )
            tags = [FALLBACK_TAG]
        primary = tags[0] if tags else None

        # Animation hints — validate against the renderer's registries; on
        # invalid pick (or the LLM omitting the field) leave None so
        # build_decision falls back to the strength-bucketed random pool.
        entry_anim = h.get("entry_animation")
        if isinstance(entry_anim, str) and entry_anim not in _VALID_ENTRY_ANIMATIONS:
            log.warning("LLM picked unknown entry_animation %r; ignoring", entry_anim)
            entry_anim = None
        idle_anim = h.get("idle_animation")
        if isinstance(idle_anim, str) and idle_anim not in _VALID_IDLE_ANIMATIONS:
            log.warning("LLM picked unknown idle_animation %r; ignoring", idle_anim)
            idle_anim = None
        color_hint = h.get("decoration_color_hint")
        if isinstance(color_hint, str):
            color_hint = color_hint.lower()
            if color_hint not in _VALID_COLOR_BUCKETS:
                log.warning(
                    "LLM picked unknown decoration_color_hint %r; ignoring",
                    color_hint,
                )
                color_hint = None

        cleaned.append(Highlight(
            time=float(h["time"]),
            text=str(h["text"]),
            is_hook=bool(h.get("is_hook", False)),
            tags=tags,
            primary_tag=primary,
            reasoning=str(h.get("reasoning", "")),
            entry_animation=entry_anim if isinstance(entry_anim, str) else None,
            idle_animation=idle_anim if isinstance(idle_anim, str) else None,
            decoration_color_hint=color_hint if isinstance(color_hint, str) else None,
        ))

    non_hooks = [str(s) for s in data.get("non_hooks", []) if isinstance(s, str)]

    # ---- Backfill missing lyric lines as untagged highlights ----
    # Smaller LLMs (e.g. gemma3:4b) tend to under-emit highlights — they
    # may flag only the most obvious hook and dump everything else into
    # non_hooks. In banner mode that means most lyric lines get NO subtitle
    # at all. So: any input lyric NOT covered by a returned highlight gets
    # synthesised as a no-tag, non-hook highlight using its original time
    # and text. The renderer can then choose to display it (banner mode
    # always does).
    covered_texts = {h.text for h in cleaned}
    for orig in lyrics:
        if orig.text not in covered_texts:
            cleaned.append(Highlight(
                time=orig.time,
                text=orig.text,
                is_hook=False,
                tags=[],
                primary_tag=None,
                reasoning="backfill: LLM omitted this line",
                duration=orig.duration,
            ))
    cleaned.sort(key=lambda h: h.time)
    return AlignmentResult(highlights=cleaned, non_hooks=non_hooks)


# ---------------------------------------------------------------------------
# Public entry — v6
# ---------------------------------------------------------------------------


def align_lyrics(
    lyrics: list[LyricLine] | list[dict],
    *,
    provider: Provider = "rule_based",
    song_title: str | None = None,
    use_cache: bool = True,
    ollama_model: str = _OLLAMA_DEFAULT_MODEL,
    ollama_host: str | None = None,
) -> AlignmentResult:
    """v6 alignment entry point.

    Returns a Pydantic `AlignmentResult` with `highlights` and `non_hooks`.
    LLM providers (`claude`, `ollama`) cache hits in `.cache/alignment/<key>.json`.
    On any provider failure, silently falls back to rule_based so we never
    block the render pipeline.
    """
    parsed = [
        L if isinstance(L, LyricLine) else LyricLine.model_validate(L)
        for L in lyrics
    ]
    if provider == "rule_based":
        return _rule_based_align(parsed)
    if provider == "claude":
        try:
            return _claude_align(parsed, song_title=song_title, use_cache=use_cache)
        except Exception as exc:  # noqa: BLE001 — never block the pipeline
            log.warning(
                "Claude alignment failed (%s); falling back to rule_based.", exc,
            )
            return _rule_based_align(parsed)
    if provider == "ollama":
        try:
            return _ollama_align(
                parsed,
                song_title=song_title,
                model=ollama_model,
                host=ollama_host,
                use_cache=use_cache,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "Ollama alignment failed (%s); falling back to rule_based.", exc,
            )
            return _rule_based_align(parsed)
    raise ValueError(f"unknown provider: {provider!r}")


# ---------------------------------------------------------------------------
# Legacy compat — pre-v6 `align(...)` signature returning list[Highlight]
# ---------------------------------------------------------------------------


def align(
    lyrics: list[LyricLine] | list[dict],
    *,
    provider: str = "rule_based",
) -> list[Highlight]:
    """Legacy shim. Returns just the highlights list (no AlignmentResult)."""
    norm: Provider = "claude" if provider in ("claude", "openai") else "rule_based"
    return align_lyrics(lyrics, provider=norm).highlights


def load_lyrics(path: Path) -> list[LyricLine]:
    """Convenience loader. Delegates to `semanticvibe.lyrics.load_lyrics`."""
    from semanticvibe.lyrics import load_lyrics as _impl
    return _impl(path)
