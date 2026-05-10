"""build_elements.build_elements_from_lyrics — v6 high-level helper."""

from __future__ import annotations

import json

from semanticvibe.build_elements import build_elements_from_lyrics
from semanticvibe.schemas.decision import Decision


def test_build_elements_rule_based_basic():
    lyrics = [
        {"time":  1.0, "text": "もしもし"},
        {"time":  4.0, "text": "電波"},
        {"time":  7.0, "text": "好き"},
        {"time": 10.0, "text": "夢"},
        {"time": 13.0, "text": "qqxxzz"},
    ]
    elements = build_elements_from_lyrics(lyrics, provider="rule_based", seed=0)
    assert isinstance(elements, list)
    assert all(isinstance(e, dict) for e in elements)

    types = [e["type"] for e in elements]
    assert "text" in types
    assert "decoration" in types
    # 'qqxxzz' has no tags → must NOT appear among the rendered elements.
    text_contents = {e["content"] for e in elements if e["type"] == "text"}
    assert "qqxxzz" not in text_contents


def test_build_elements_emits_one_hero_at_most():
    lyrics = [
        {"time":  1.0, "text": "好き"},
        {"time":  4.0, "text": "夢"},
        {"time":  7.0, "text": "fire"},
    ]
    elements = build_elements_from_lyrics(lyrics, provider="rule_based", seed=0)
    heroes = [e for e in elements if e["type"] == "hero_text"]
    assert len(heroes) <= 1


def test_build_elements_decoration_links_to_text():
    """Each decoration's near_text_id must point at a real text element index."""
    lyrics = [{"time": 1.0, "text": "好き"}, {"time": 4.0, "text": "fire"}]
    elements = build_elements_from_lyrics(lyrics, provider="rule_based", seed=0)
    for i, e in enumerate(elements):
        if e["type"] == "decoration":
            assert isinstance(e["near_text_id"], int)
            ref = elements[e["near_text_id"]]
            assert ref["type"] == "text"
            # near_text_id must precede the decoration in the list.
            assert e["near_text_id"] < i


def test_build_elements_round_trips_to_decision():
    """The list[dict] output must validate against the Pydantic Decision schema."""
    lyrics = [{"time": 1.0, "text": "好き"}, {"time": 4.0, "text": "夢"}]
    elements = build_elements_from_lyrics(lyrics, provider="rule_based", seed=0)
    decision = Decision.model_validate({
        "elements": elements,
        "global_style": {
            "color_palette": ["#FF6B9D", "#E63946", "#FFFFFF"],
            "vibe": "test",
        },
    })
    assert len(decision.elements) == len(elements)


def test_build_elements_uses_only_closed_vocab_tags():
    from semanticvibe.semantic_align import VALID_TAGS

    lyrics = [
        {"time":  1.0, "text": "好き"},
        {"time":  4.0, "text": "夢"},
        {"time":  7.0, "text": "fire"},
        {"time": 10.0, "text": "電波"},
    ]
    elements = build_elements_from_lyrics(lyrics, provider="rule_based", seed=0)
    for e in elements:
        if e["type"] == "decoration":
            assert e["asset_tag"] in VALID_TAGS, (
                f"decoration tag {e['asset_tag']!r} escaped the closed vocab"
            )


def test_build_elements_serialises_to_json():
    """Output must be JSON-clean (no tuples, no non-serialisable types)."""
    lyrics = [{"time": 1.0, "text": "好き"}]
    elements = build_elements_from_lyrics(lyrics, provider="rule_based", seed=0)
    blob = json.dumps(elements, ensure_ascii=False)
    parsed = json.loads(blob)
    assert parsed == elements
