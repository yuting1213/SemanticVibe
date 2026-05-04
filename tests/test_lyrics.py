"""Lyrics schema (Pydantic) + I/O round-trip tests."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from semanticvibe.lyrics import LyricLine, load_lyrics, save_lyrics, to_dict_list


def test_minimal_line_has_no_duration():
    line = LyricLine(time=2.5, text="もしもし")
    assert line.duration is None


def test_with_duration():
    line = LyricLine(time=2.5, text="hi", duration=1.2)
    assert line.duration == 1.2


def test_negative_time_rejected():
    with pytest.raises(ValidationError):
        LyricLine(time=-1.0, text="x")


def test_empty_text_rejected():
    with pytest.raises(ValidationError):
        LyricLine(time=0.0, text="")


def test_non_positive_duration_rejected():
    with pytest.raises(ValidationError):
        LyricLine(time=0.0, text="x", duration=0)
    with pytest.raises(ValidationError):
        LyricLine(time=0.0, text="x", duration=-0.5)


def test_round_trip_preserves_optional_duration(tmp_path):
    lyrics = [
        LyricLine(time=2.5, text="もしもし"),               # no duration
        LyricLine(time=5.0, text="電波", duration=1.2),     # explicit
    ]
    out = save_lyrics(lyrics, tmp_path / "lyrics.json")
    loaded = load_lyrics(out)
    assert loaded == lyrics


def test_save_omits_none_duration_field(tmp_path):
    """When duration is None, the JSON shouldn't carry an explicit null —
    just leave the key out so files stay clean for hand-editing."""
    save_lyrics([LyricLine(time=1.0, text="x")], tmp_path / "lyrics.json")
    raw = json.loads((tmp_path / "lyrics.json").read_text(encoding="utf-8"))
    assert raw == [{"time": 1.0, "text": "x"}]
    assert "duration" not in raw[0]


def test_load_rejects_bad_schema_with_clear_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"timestamp": 1.0, "text": "x"}]), encoding="utf-8")
    with pytest.raises(ValidationError) as exc_info:
        load_lyrics(bad)
    # Pydantic's error mentions the missing required field.
    assert "time" in str(exc_info.value)


def test_load_accepts_existing_lyrics_mosimosi(repo_root):
    """The committed sample file must validate (no `duration` field is fine)."""
    lyrics = load_lyrics(repo_root / "samples" / "lyrics_mosimosi.json")
    assert len(lyrics) == 4
    assert lyrics[0].text == "もしもし"
    assert all(L.duration is None for L in lyrics)


def test_to_dict_list_drops_none():
    lines = [LyricLine(time=1.0, text="a"), LyricLine(time=2.0, text="b", duration=0.5)]
    out = to_dict_list(lines)
    assert out == [{"time": 1.0, "text": "a"}, {"time": 2.0, "text": "b", "duration": 0.5}]
