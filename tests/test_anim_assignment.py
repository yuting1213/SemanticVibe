"""assign_random_animations + beat_sync.snap_to_beat unit tests.

Doesn't touch the audio pipeline — uses a hand-built BeatInfo so the
test is offline-clean.
"""

from __future__ import annotations

import pytest

from semanticvibe.llm.anim_assignment import (
    POOL_IDLE,
    POOL_NORMAL_ENTRY,
    POOL_STRONG_ENTRY,
    assign_random_animations,
)
from semanticvibe.preprocess.beat_sync import BeatInfo, snap_to_beat
from semanticvibe.schemas.decision import (
    Decision,
    DecorationElement,
    GlobalStyle,
    HeroTextElement,
    TextElement,
)


def _text(start: float, end: float = None) -> TextElement:
    return TextElement(
        content="x",
        start_time=start,
        end_time=end if end is not None else start + 2.0,
        font="KleeOne-Regular",
        size=48,
        color="#fff",
        outline_color="#000",
        outline_width=2,
        animation="fade",
        reasoning="t",
    )


def _decoration(start: float) -> DecorationElement:
    return DecorationElement(
        asset_tag="heart",
        start_time=start,
        end_time=start + 2.0,
        reasoning="t",
    )


def _hero(start: float) -> HeroTextElement:
    return HeroTextElement(
        content="夢",
        start_time=start,
        end_time=start + 5.0,
        reasoning="t",
    )


def _decision(elements) -> Decision:
    return Decision(
        elements=elements,
        global_style=GlobalStyle(color_palette=["#fff"], vibe="t"),
    )


def test_snap_to_beat_within_tolerance():
    beats = [1.0, 2.0, 3.0, 4.0]
    assert snap_to_beat(2.05, beats) == 2.0
    assert snap_to_beat(1.9, beats) == 2.0


def test_snap_to_beat_outside_tolerance_keeps_original():
    beats = [1.0, 2.0, 3.0]
    # 0.30 s away from any beat — outside default 0.15 s tolerance.
    assert snap_to_beat(2.30, beats) == 2.30


def test_snap_to_beat_empty_beats_passthrough():
    assert snap_to_beat(1.5, []) == 1.5


def test_assign_picks_strong_entry_on_downbeat():
    info = BeatInfo(beats=[1.0], downbeats={1.0})
    decision = _decision([_text(1.0)])
    out = assign_random_animations(decision, info, seed=0)
    assert out.elements[0].animation in POOL_STRONG_ENTRY


def test_assign_picks_normal_entry_off_downbeat():
    info = BeatInfo(beats=[1.0], downbeats=set())
    decision = _decision([_text(1.0)])
    out = assign_random_animations(decision, info, seed=0)
    assert out.elements[0].animation in POOL_NORMAL_ENTRY


def test_assign_idle_always_set():
    info = BeatInfo(beats=[1.0], downbeats={1.0})
    decision = _decision([_text(1.0), _decoration(1.0)])
    out = assign_random_animations(decision, info, seed=0)
    for el in out.elements:
        assert el.idle_animation in POOL_IDLE


def test_assign_leaves_hero_text_alone():
    """HeroTextElement has its own envelope — randomization shouldn't touch it."""
    info = BeatInfo(beats=[3.0], downbeats={3.0})
    hero = _hero(3.0)
    decision = _decision([hero])
    out = assign_random_animations(decision, info, seed=0)
    # Hero is returned unchanged (no animation field changed, breathing intact).
    assert out.elements[0] is hero  # same instance — no copy


def test_assign_preserves_duration_after_snap():
    """Snapping start_time should also shift end_time so duration is preserved."""
    info = BeatInfo(beats=[1.0, 5.0], downbeats=set())
    el = _text(start=1.05, end=4.05)  # 3-second duration; will snap to 1.0
    decision = _decision([el])
    out = assign_random_animations(decision, info, seed=0)
    out_el = out.elements[0]
    assert out_el.start_time == pytest.approx(1.0, abs=1e-9)
    assert (out_el.end_time - out_el.start_time) == pytest.approx(3.0, abs=1e-9)


def test_assign_deterministic_for_same_seed():
    info = BeatInfo(beats=[1.0, 2.0], downbeats={1.0})
    decision = _decision([_text(1.0), _text(2.0), _decoration(2.0)])
    a = assign_random_animations(decision, info, seed=42)
    b = assign_random_animations(decision, info, seed=42)
    assert [(e.animation, e.idle_animation) for e in a.elements] == [
        (e.animation, e.idle_animation) for e in b.elements
    ]
