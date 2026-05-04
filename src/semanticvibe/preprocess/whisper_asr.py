"""Whisper ASR via faster-whisper, run in a subprocess.

Why subprocess: faster-whisper (via ctranslate2) bundles its own cuDNN, and
PyTorch (used by BLIP and Open-CLIP downstream) bundles a different cuDNN
build. Loading both into the same process on Windows produces "Could not
load symbol cudnnGetLibConfig" / heap corruption. Isolating Whisper in a
short-lived subprocess sidesteps the symbol conflict cleanly.

Two public entry points (v5 split):

- `transcribe_audio(audio_path, ...)` — core. Takes any ffmpeg-decodable
  file (mp3 / wav / m4a / mp4 / mov / etc.) and returns lyric segments.
- `transcribe_video(video_path, ...)` — convenience alias that documents
  intent ("I'm asking for the embedded audio track of a video"). Same
  implementation; both pipe through `extract_wav` for loudness
  normalisation.

The legacy `transcribe(video_path)` name is kept as an alias for
backwards compat with existing callers (preprocess/pipeline.py and
the full-pipeline cli.py).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from semanticvibe.preprocess.librosa_beats import extract_wav
from semanticvibe.schemas.feature_summary import LyricSegment

log = logging.getLogger(__name__)


_WORKER_SOURCE = '''
import json, sys
sys.stdout.reconfigure(encoding="utf-8")
from faster_whisper import WhisperModel

audio_path, model_size, language, device, vad = sys.argv[1:6]
compute_type = "float16" if device == "cuda" else "int8"
model = WhisperModel(model_size, device=device, compute_type=compute_type)
segments, _info = model.transcribe(
    audio_path,
    language=(None if language == "auto" else language),
    beam_size=5,
    vad_filter=(vad == "1"),
    vad_parameters={"min_silence_duration_ms": 400} if vad == "1" else None,
)
out = []
for s in segments:
    text = (s.text or "").strip()
    if text:
        out.append({"time": float(s.start), "text": text})
print("__SVIBE_RESULT__" + json.dumps(out, ensure_ascii=False))
'''


def transcribe_audio(
    audio_path: Path | str,
    *,
    model_size: str = "large-v3",
    language: str | None = "zh",
    device: str = "cuda",
    vad: bool = False,
    loudnorm: bool = True,
) -> list[LyricSegment]:
    """Transcribe any ffmpeg-decodable audio/video file.

    Args:
        audio_path: a file path. mp3 / wav / m4a / mp4 / mov — anything
            ffmpeg can read. We pipe through extract_wav internally for
            loudness normalisation.
        model_size: faster-whisper model name. "large-v3" gives best CJK
            quality (~3 GB VRAM); "medium" is the fast fallback.
        language: ISO code or None for auto-detect. Default is "zh".
        device: "cuda" or "cpu".
        vad: Voice-activity-detection filter. Default OFF — Silero VAD is
            tuned for speech and drops segments where music shares the
            vocal band (typical for music videos). Turn ON for podcasts /
            interviews with long silences.
        loudnorm: Pre-amplify to ~-16 LUFS via ffmpeg's loudnorm filter.
            Default ON — phone recordings often sit at -35 to -45 dB mean
            which is below Whisper's speech threshold.

    Returns:
        List of `LyricSegment(time, text)` in source order. Empty list if
        the subprocess fails or finds no speech.
    """
    audio_path = Path(audio_path)
    norm_wav = extract_wav(audio_path, sr=16000, loudnorm=loudnorm)

    cmd = [
        sys.executable,
        "-c",
        _WORKER_SOURCE,
        str(norm_wav),
        model_size,
        language or "auto",
        device,
        "1" if vad else "0",
    ]
    # PYTHONIOENCODING ensures the worker's print() doesn't fall back to
    # the system codepage (cp950 on Traditional-Chinese Windows) and
    # crash on Simplified-Chinese characters Whisper sometimes emits.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", env=env
    )
    if proc.returncode != 0:
        log.error(
            "Whisper subprocess failed (exit %d): %s",
            proc.returncode, proc.stderr[-2000:],
        )
        return []

    payload: list[dict] = []
    for line in proc.stdout.splitlines():
        if line.startswith("__SVIBE_RESULT__"):
            payload = json.loads(line[len("__SVIBE_RESULT__"):])
            break
    return [LyricSegment(time=item["time"], text=item["text"]) for item in payload]


def transcribe_video(
    video_path: Path | str,
    *,
    model_size: str = "large-v3",
    language: str | None = "zh",
    device: str = "cuda",
    vad: bool = False,
    loudnorm: bool = True,
) -> list[LyricSegment]:
    """Convenience: transcribe the audio track embedded in a video file.

    Same implementation as `transcribe_audio` (extract_wav handles the
    container peeling), but documents intent at call sites.
    """
    return transcribe_audio(
        video_path,
        model_size=model_size,
        language=language,
        device=device,
        vad=vad,
        loudnorm=loudnorm,
    )


# Legacy alias — preprocess/pipeline.py + cli.py + tests still call this name.
transcribe = transcribe_video
