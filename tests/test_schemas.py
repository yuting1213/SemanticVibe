"""Schemas are the system's narrow waist — these tests gate every downstream stage."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from semanticvibe.schemas.decision import (
    Decision,
    DecorationElement,
    GlobalStyle,
    TextElement,
)
from semanticvibe.schemas.feature_summary import FeatureSummary, LyricSegment


# ---------------------------------------------------------------------------
# FeatureSummary
# ---------------------------------------------------------------------------


def _valid_feature_summary_dict() -> dict:
    return {
        "lyrics": [{"time": 0.5, "text": "夏天的尾巴"}],
        "video_description": "A young couple walks through a sunlit alley.",
        "beat_times": [0.5, 1.0, 1.5],
        "chorus_segments": [(8.0, 16.0)],
        "video_duration": 30.0,
        "style_preset": "warm_handdrawn",
    }


def test_feature_summary_round_trip():
    fs = FeatureSummary.model_validate(_valid_feature_summary_dict())
    again = FeatureSummary.model_validate(fs.model_dump())
    assert again == fs


def test_feature_summary_rejects_non_monotonic_beats():
    bad = _valid_feature_summary_dict()
    bad["beat_times"] = [1.0, 0.5]
    with pytest.raises(ValidationError):
        FeatureSummary.model_validate(bad)


def test_feature_summary_rejects_inverted_chorus():
    bad = _valid_feature_summary_dict()
    bad["chorus_segments"] = [(10.0, 5.0)]
    with pytest.raises(ValidationError):
        FeatureSummary.model_validate(bad)


def test_feature_summary_rejects_zero_duration():
    bad = _valid_feature_summary_dict()
    bad["video_duration"] = 0
    with pytest.raises(ValidationError):
        FeatureSummary.model_validate(bad)


def test_lyric_segment_rejects_negative_time():
    with pytest.raises(ValidationError):
        LyricSegment(time=-1.0, text="x")


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def test_decision_loads_hand_written_example(hand_written_decision_dict):
    decision = Decision.model_validate(hand_written_decision_dict)
    assert len(decision.elements) == 3
    assert len(decision.text_elements()) == 2
    assert len(decision.decoration_elements()) == 1


def test_text_element_anchor_list_becomes_tuple():
    el = TextElement.model_validate(
        {
            "type": "text",
            "content": "x",
            "start_time": 0,
            "end_time": 1,
            "anchor": [10, 20],
            "font": "Klee",
            "size": 32,
            "color": "#fff",
            "outline_color": "#000",
            "outline_width": 2,
            "animation": "fade",
            "reasoning": "test",
        }
    )
    assert el.anchor == (10, 20)


def test_text_element_anchor_auto_default():
    el = TextElement.model_validate(
        {
            "type": "text",
            "content": "x",
            "start_time": 0,
            "end_time": 1,
            "font": "Klee",
            "size": 32,
            "color": "#fff",
            "outline_color": "#000",
            "outline_width": 2,
            "animation": "fade",
            "reasoning": "test",
        }
    )
    assert el.anchor == "auto"


def test_element_rejects_inverted_times():
    with pytest.raises(ValidationError):
        TextElement.model_validate(
            {
                "type": "text",
                "content": "x",
                "start_time": 5,
                "end_time": 3,
                "font": "Klee",
                "size": 32,
                "color": "#fff",
                "outline_color": "#000",
                "outline_width": 2,
                "animation": "fade",
                "reasoning": "test",
            }
        )


def test_reasoning_is_required_on_decoration():
    with pytest.raises(ValidationError):
        DecorationElement.model_validate(
            {
                "type": "decoration",
                "asset_tag": "sparkle",
                "start_time": 1,
                "end_time": 2,
            }
        )


def test_discriminated_union_dispatches_on_type():
    decision = Decision(
        elements=[
            TextElement(
                content="x",
                start_time=0,
                end_time=1,
                font="Klee",
                size=32,
                color="#fff",
                outline_color="#000",
                outline_width=2,
                animation="fade",
                reasoning="r",
            ),
            DecorationElement(
                asset_tag="sparkle",
                start_time=0,
                end_time=1,
                reasoning="r",
            ),
        ],
        global_style=GlobalStyle(color_palette=["#fff"], vibe="soft"),
    )
    # Round-trip through JSON to make sure the discriminator survives.
    raw = decision.model_dump_json()
    again = Decision.model_validate_json(raw)
    assert isinstance(again.elements[0], TextElement)
    assert isinstance(again.elements[1], DecorationElement)
