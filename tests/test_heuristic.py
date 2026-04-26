"""Heuristic Decision generator tests — pure functions, no API key needed."""

from __future__ import annotations

import pytest

from semanticvibe.llm.heuristic import heuristic_decision
from semanticvibe.schemas.decision import DecorationElement, TextElement
from semanticvibe.schemas.feature_summary import FeatureSummary, LyricSegment


def _summary(**overrides) -> FeatureSummary:
    base = {
        "lyrics": [
            LyricSegment(time=1.0, text="第一句"),
            LyricSegment(time=5.0, text="第二句"),
            LyricSegment(time=9.0, text="第三句"),
            LyricSegment(time=13.0, text="第四句"),
        ],
        "video_description": "A test clip.",
        "beat_times": [0.5, 1.0, 1.5, 2.0],
        "chorus_segments": [(8.0, 14.0)],
        "video_duration": 30.0,
        "style_preset": "warm_handdrawn",
    }
    base.update(overrides)
    return FeatureSummary.model_validate(base)


def test_heuristic_emits_title_text():
    decision = heuristic_decision(_summary())
    titles = [e for e in decision.elements if isinstance(e, TextElement)]
    assert titles, "expected at least one text element"
    assert titles[0].content.startswith("第一")


def test_heuristic_emits_burst_on_title():
    decision = heuristic_decision(_summary())
    bursts = [
        e for e in decision.elements
        if isinstance(e, DecorationElement) and e.asset_tag == "burst"
    ]
    assert len(bursts) == 1
    assert bursts[0].near_text_id == 0


def test_heuristic_emits_heart_confetti():
    decision = heuristic_decision(_summary())
    confetti = [
        e for e in decision.elements
        if isinstance(e, DecorationElement) and e.scatter
    ]
    assert len(confetti) == 1
    assert confetti[0].count >= 8
    assert confetti[0].asset_tag == "mini-heart"
    assert confetti[0].color_tint, "confetti must carry a colour palette"


def test_heuristic_text_has_white_halo():
    decision = heuristic_decision(_summary())
    titles = [e for e in decision.elements if isinstance(e, TextElement)]
    assert titles
    # Every text element should carry the standard white halo layer.
    assert all(len(t.outline_layers) >= 1 for t in titles)
    assert all(t.outline_layers[0].color.upper() == "#FFFFFF" for t in titles)


def test_heuristic_emits_chorus_decoration():
    decision = heuristic_decision(_summary())
    stars = [
        e for e in decision.elements
        if isinstance(e, DecorationElement) and e.asset_tag == "star"
    ]
    assert len(stars) == 1
    # Chorus starts at 8.0s in the fixture.
    assert stars[0].start_time == pytest.approx(8.0)


def test_heuristic_handles_empty_lyrics():
    """Even without ASR output, heuristic must produce something renderable."""
    decision = heuristic_decision(_summary(lyrics=[], chorus_segments=[]))
    assert decision.elements, "must emit at least one placeholder"
    assert any(isinstance(e, TextElement) for e in decision.elements)


def test_heuristic_unknown_style_falls_back_to_warm_handdrawn():
    decision = heuristic_decision(_summary(style_preset="not_a_real_preset"))
    # Warm hand-drawn palette starts with #F4A261.
    assert decision.global_style.color_palette[0] == "#F4A261"


def test_heuristic_text_times_within_video_duration():
    decision = heuristic_decision(_summary(video_duration=10.0))
    for el in decision.elements:
        assert el.end_time <= 10.0


def test_heuristic_serialises_round_trip():
    """Output must validate against the Decision schema (it's how we ship it downstream)."""
    decision = heuristic_decision(_summary())
    raw = decision.model_dump_json()
    from semanticvibe.schemas.decision import Decision

    assert Decision.model_validate_json(raw) == decision
