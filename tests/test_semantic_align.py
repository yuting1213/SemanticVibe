"""semantic_align v6 — closed-vocab alignment + rule_based provider."""

from __future__ import annotations

import json

import pytest

from semanticvibe.lyrics import LyricLine
from semanticvibe.semantic_align import (
    FALLBACK_TAG,
    KEYWORD_TO_TAGS,
    VALID_TAGS,
    AlignmentResult,
    Highlight,
    _parse_strict_alignment_json,
    align,
    align_lyrics,
)


def _lyrics(*pairs: tuple[float, str]) -> list[LyricLine]:
    return [LyricLine(time=t, text=txt) for t, txt in pairs]


# --- vocabulary integrity ----------------------------------------------------


def test_keyword_to_tags_subset_of_valid_tags():
    """Every tag returned by rule_based must live in the closed vocab."""
    used = {t for tags in KEYWORD_TO_TAGS.values() for t in tags}
    assert used.issubset(VALID_TAGS), (
        f"KEYWORD_TO_TAGS leaks tags outside vocab: {used - VALID_TAGS}"
    )


def test_fallback_tag_is_valid():
    assert FALLBACK_TAG in VALID_TAGS


# --- rule_based behaviour ----------------------------------------------------


def test_rule_based_japanese_routing():
    """The mosimosi sample case under the v6 closed vocab."""
    lyrics = _lyrics(
        (2.5, "もしもし"),
        (5.0, "電波"),
        (8.0, "好き"),
        (11.0, "可愛い"),
    )
    res = align_lyrics(lyrics, provider="rule_based")
    assert isinstance(res, AlignmentResult)
    by_text = {h.text: h for h in res.highlights}
    assert "speech_bubble" in by_text["もしもし"].tags
    assert "lightning" in by_text["電波"].tags
    assert "heart" in by_text["好き"].tags
    # 可愛い triggers heart + sparkle multi-tag.
    assert by_text["可愛い"].tags[:2] == ["heart", "sparkle"]


def test_primary_tag_is_first_tag():
    res = align_lyrics(_lyrics((1.0, "好き")), provider="rule_based")
    h = res.highlights[0]
    assert h.tags == ["heart"]
    assert h.primary_tag == "heart"


def test_unmatched_line_goes_to_non_hooks_and_has_no_tags():
    """Lines with no keyword match end up in non_hooks; the highlight still
    exists (so the renderer can choose to display the plain text) but tags=[]."""
    lyrics = _lyrics((1.0, "好き"), (2.0, "qqxxzz"))
    res = align_lyrics(lyrics, provider="rule_based")
    matched = next(h for h in res.highlights if h.text == "好き")
    unmatched = next(h for h in res.highlights if h.text == "qqxxzz")
    assert matched.tags == ["heart"]
    assert unmatched.tags == []
    assert "qqxxzz" in res.non_hooks
    assert "好き" not in res.non_hooks


def test_dict_input_accepted():
    raw = [{"time": 1.0, "text": "fire"}, {"time": 2.0, "text": "love"}]
    res = align_lyrics(raw, provider="rule_based")
    assert res.highlights[0].primary_tag == "fire"
    assert res.highlights[1].primary_tag == "heart"


def test_longest_trigger_wins_for_kawaii():
    """可愛い is a longer trigger than 愛, so it shouldn't be poached by heart-only."""
    res = align_lyrics(_lyrics((1.0, "可愛い")), provider="rule_based")
    tags = res.highlights[0].tags
    # heart appears (because 可愛い → [heart, sparkle]) AND sparkle appears,
    # AND we did NOT pick up plain 愛's heart-only mapping in addition.
    assert "heart" in tags and "sparkle" in tags


def test_multi_tag_per_line():
    """A line can imply multiple tags."""
    res = align_lyrics(_lyrics((1.0, "電波好き")), provider="rule_based")
    h = res.highlights[0]
    assert "lightning" in h.tags
    assert "heart" in h.tags


def test_is_hook_short_tagged_lines():
    """Short tagged Japanese lines are flagged as hooks."""
    res = align_lyrics(_lyrics((1.0, "好き")), provider="rule_based")
    assert res.highlights[0].is_hook is True


def test_is_hook_false_for_untagged():
    res = align_lyrics(_lyrics((1.0, "qqxxzz")), provider="rule_based")
    assert res.highlights[0].is_hook is False


def test_legacy_align_returns_list():
    """Pre-v6 callers used `align()` returning a flat list of Highlights."""
    res = align(_lyrics((1.0, "好き"), (2.0, "fire")), provider="rule_based")
    assert isinstance(res, list)
    assert all(isinstance(h, Highlight) for h in res)


def test_legacy_compat_properties():
    """The lyric_time / lyric_text / decoration_tag / strength shims still work."""
    res = align(_lyrics((1.0, "好き")), provider="rule_based")
    h = res[0]
    assert h.lyric_time == 1.0
    assert h.lyric_text == "好き"
    assert h.decoration_tag == "heart"
    assert h.strength > 0.5


# --- strict JSON parser ------------------------------------------------------


def test_strict_parser_rejects_invalid_tags_with_fallback():
    """Tags outside the closed vocab get scrubbed; the line still gets a
    fallback tag instead of being silently dropped."""
    bad = json.dumps({
        "highlights": [
            {"time": 1.0, "text": "x", "is_hook": False,
             "tags": ["nonexistent_tag", "another_fake"], "primary_tag": "nonexistent_tag",
             "reasoning": "test"},
        ],
        "non_hooks": [],
    })
    res = _parse_strict_alignment_json(bad, _lyrics((1.0, "x")))
    assert res.highlights[0].tags == [FALLBACK_TAG]
    assert res.highlights[0].primary_tag == FALLBACK_TAG


def test_strict_parser_strips_code_fences():
    raw = "```json\n" + json.dumps({
        "highlights": [
            {"time": 1.0, "text": "x", "is_hook": True,
             "tags": ["heart"], "primary_tag": "heart", "reasoning": "test"},
        ],
        "non_hooks": [],
    }) + "\n```"
    res = _parse_strict_alignment_json(raw, _lyrics((1.0, "x")))
    assert res.highlights[0].primary_tag == "heart"


def test_strict_parser_finds_inner_object_when_prose_around():
    raw = "Here is the JSON you asked for:\n" + json.dumps({
        "highlights": [
            {"time": 1.0, "text": "x", "is_hook": False,
             "tags": ["heart"], "primary_tag": "heart", "reasoning": "x"},
        ],
        "non_hooks": [],
    }) + "\nHope that helps!"
    res = _parse_strict_alignment_json(raw, _lyrics((1.0, "x")))
    assert res.highlights[0].primary_tag == "heart"


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        align_lyrics([], provider="not_a_real_provider")  # type: ignore[arg-type]


# --- sample-file integration -------------------------------------------------


def test_sample_lyrics_test_aligns(repo_root):
    lyrics = json.loads((repo_root / "samples" / "lyrics_test.json").read_text(encoding="utf-8"))
    res = align_lyrics(lyrics, provider="rule_based")
    by_text = {h.text: h for h in res.highlights}
    assert by_text["もしもし"].primary_tag == "speech_bubble"
    assert by_text["電波"].primary_tag == "lightning"
    assert by_text["好き"].primary_tag == "heart"
    assert by_text["夢"].primary_tag == "star"
    # 'qqxxzz' is engineered to match nothing.
    assert by_text["qqxxzz"].tags == []
    assert "qqxxzz" in res.non_hooks
