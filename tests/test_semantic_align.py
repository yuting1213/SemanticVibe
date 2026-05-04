"""semantic_align rule_based aligner + LLM JSON parser tests."""

from __future__ import annotations

import json

import pytest

from semanticvibe.semantic_align import (
    Highlight,
    LyricLine,
    TAG_VOCABULARY,
    _parse_highlights_json,
    align,
)


def _lyrics(*pairs: tuple[float, str]) -> list[LyricLine]:
    return [LyricLine(time=t, text=txt) for t, txt in pairs]


def test_rule_based_japanese_routing():
    """The mosimosi sample case — Japanese keyword → tag mapping.

    Tag aliases reflect the actual asset_lib: 電波 / lightning aren't
    distinct stickers in our library, so both alias to `exclaim` (the
    jagged red impact star — same energy, available shape).
    """
    lyrics = _lyrics(
        (2.5, "もしもし"),
        (5.0, "電波"),
        (8.0, "好き"),
        (11.0, "可愛い"),
    )
    highlights = align(lyrics, provider="rule_based")
    assert len(highlights) == 4
    by_text = {h.lyric_text: h.decoration_tag for h in highlights}
    assert by_text["電波"] == "exclaim"
    assert by_text["好き"] == "heart"
    assert by_text["可愛い"] == "mini-heart"
    assert by_text["もしもし"] is None  # no idiomatic trigger matches


def test_rule_based_strength_higher_when_tag_matched():
    # "qqxxzz" is engineered to NOT contain any TAG_VOCABULARY trigger as a
    # substring (we deliberately don't word-boundary-match short Latin
    # triggers like "ah" / "oh" because CJK lyrics often write "ahhh" inline).
    lyrics = _lyrics((1.0, "好き"), (2.0, "qqxxzz"))
    [matched, unmatched] = align(lyrics, provider="rule_based")
    assert matched.strength > unmatched.strength
    assert matched.decoration_tag == "heart"
    assert unmatched.decoration_tag is None


def test_rule_based_accepts_dicts():
    """Convenience: align should accept raw dicts, not just LyricLine."""
    raw = [{"time": 1.0, "text": "fire"}, {"time": 2.0, "text": "love"}]
    highlights = align(raw, provider="rule_based")
    assert highlights[0].decoration_tag == "fire"
    assert highlights[1].decoration_tag == "heart"


def test_tag_vocabulary_first_trigger_resolves():
    """Every tag's *first* trigger (the canonical one) must resolve to a tag.

    Tag names are filenames (e.g. 'exclaim', 'mini-heart') and are not
    necessarily themselves valid triggers — what matters is that idiomatic
    triggers in TAG_VOCABULARY succeed.
    """
    for tag, triggers in TAG_VOCABULARY.items():
        if not triggers:
            continue
        canonical = triggers[0]
        lyrics = _lyrics((1.0, canonical))
        highlights = align(lyrics, provider="rule_based")
        assert highlights[0].decoration_tag is not None, (
            f"tag {tag!r}'s canonical trigger {canonical!r} should resolve"
        )


def test_parse_highlights_json_strips_code_fence():
    raw = "```json\n" + json.dumps({
        "highlights": [
            {"lyric_time": 1.0, "lyric_text": "x", "decoration_tag": "heart",
             "strength": 0.7, "reasoning": "test"},
        ]
    }) + "\n```"
    parsed = _parse_highlights_json(raw)
    assert len(parsed) == 1
    assert parsed[0].decoration_tag == "heart"


def test_parse_highlights_finds_inner_object_when_prose_around():
    raw = "Sure, here's the JSON:\n" + json.dumps({
        "highlights": [
            {"lyric_time": 1.0, "lyric_text": "x", "decoration_tag": None,
             "strength": 0.5, "reasoning": ""},
        ]
    }) + "\nLet me know if you want changes."
    parsed = _parse_highlights_json(raw)
    assert len(parsed) == 1
    assert parsed[0].decoration_tag is None


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        align([], provider="not_a_real_provider")  # type: ignore[arg-type]
