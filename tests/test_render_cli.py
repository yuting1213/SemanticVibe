"""render/__main__ CLI surface — argument validation + get_lyrics priority.

End-to-end render tests are integration territory (they need
real fonts + Whisper model downloads). Here we just verify the
flag-parsing + lyrics-priority logic is correct.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from semanticvibe.lyrics import LyricLine
from semanticvibe.render.__main__ import _parse_args, get_lyrics


def _fake_whisper(*args, **kwargs):
    # Returns 2 fake LyricLines so we can tell the source apart per call.
    media = args[0] if args else kwargs.get("media_path")
    return [LyricLine(time=1.0, text=f"whisper-on:{Path(media).name}")]


def test_priority_1_manual_lyrics_wins(tmp_path):
    """When --lyrics is given, Whisper must NOT run."""
    lyrics_file = tmp_path / "lyrics.json"
    lyrics_file.write_text('[{"time": 0.5, "text": "manual"}]', encoding="utf-8")
    args = _parse_args([
        "--video", "v.mp4", "--lyrics", str(lyrics_file), "--out", "out.mp4",
    ])
    log = logging.getLogger("test")
    with patch("semanticvibe.render.__main__._whisper_to_lyric_lines") as mock_w:
        result = get_lyrics(args, log)
    mock_w.assert_not_called()
    assert len(result) == 1
    assert result[0].text == "manual"


def test_priority_2_audio_routes_to_whisper(tmp_path):
    """When --audio is given (and no --lyrics), Whisper runs on --audio."""
    args = _parse_args([
        "--video", "v.mp4", "--audio", "song.mp3", "--out", "out.mp4",
    ])
    log = logging.getLogger("test")
    with patch(
        "semanticvibe.render.__main__._whisper_to_lyric_lines",
        side_effect=_fake_whisper,
    ) as mock_w:
        result = get_lyrics(args, log)
    mock_w.assert_called_once()
    # First positional arg is the media path being transcribed.
    transcribed_media = mock_w.call_args.args[0]
    assert Path(transcribed_media).name == "song.mp3"
    assert result[0].text == "whisper-on:song.mp3"


def test_priority_3_default_to_video_audio(tmp_path):
    """No --lyrics and no --audio → Whisper on the video itself."""
    args = _parse_args(["--video", "dance.mp4", "--out", "out.mp4"])
    log = logging.getLogger("test")
    with patch(
        "semanticvibe.render.__main__._whisper_to_lyric_lines",
        side_effect=_fake_whisper,
    ) as mock_w:
        result = get_lyrics(args, log)
    mock_w.assert_called_once()
    transcribed_media = mock_w.call_args.args[0]
    assert Path(transcribed_media).name == "dance.mp4"
    assert result[0].text == "whisper-on:dance.mp4"


def test_mix_audio_choices_validated():
    """--mix-audio only accepts replace / overlay; argparse rejects others."""
    with pytest.raises(SystemExit):
        _parse_args([
            "--video", "v.mp4", "--out", "out.mp4",
            "--audio", "a.mp3", "--mix-audio", "scramble",
        ])


def test_mix_audio_default_is_none():
    args = _parse_args(["--video", "v.mp4", "--out", "out.mp4"])
    assert args.mix_audio is None


@pytest.mark.parametrize("mode", ["replace", "overlay"])
def test_mix_audio_modes_parse(mode):
    args = _parse_args([
        "--video", "v.mp4", "--out", "out.mp4",
        "--audio", "a.mp3", "--mix-audio", mode,
    ])
    assert args.mix_audio == mode
