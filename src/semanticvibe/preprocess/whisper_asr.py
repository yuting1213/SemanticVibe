"""Whisper ASR via faster-whisper, run in a subprocess.

Why subprocess: faster-whisper (via ctranslate2) bundles its own cuDNN, and
PyTorch (used by BLIP and Open-CLIP downstream) bundles a different cuDNN
build. Loading both into the same process on Windows produces "Could not
load symbol cudnnGetLibConfig" / heap corruption. Isolating Whisper in a
short-lived subprocess sidesteps the symbol conflict cleanly: the OS frees
ctranslate2's cuDNN at process exit, and PyTorch loads its own cleanly.

The subprocess emits JSON on stdout; the parent decodes into LyricSegment.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

from semanticvibe.schemas.feature_summary import LyricSegment

log = logging.getLogger(__name__)


_WORKER_SOURCE = '''
import json, sys
from faster_whisper import WhisperModel

video_path, model_size, language, device = sys.argv[1:5]
compute_type = "float16" if device == "cuda" else "int8"
model = WhisperModel(model_size, device=device, compute_type=compute_type)
segments, _info = model.transcribe(
    video_path,
    language=(None if language == "auto" else language),
    beam_size=5,
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 400},
)
out = []
for s in segments:
    text = (s.text or "").strip()
    if text:
        out.append({"time": float(s.start), "text": text})
print("__SVIBE_RESULT__" + json.dumps(out, ensure_ascii=False))
'''


def transcribe(
    video_path: Path,
    *,
    model_size: str = "large-v3",
    language: str | None = "zh",
    device: str = "cuda",
) -> list[LyricSegment]:
    """Extract lyric segments from `video_path`'s audio track.

    Args:
        model_size: faster-whisper model name. "large-v3" gives best CJK
            quality (~3 GB VRAM); "medium" is the fast fallback.
        language: ISO code or None for auto-detect. Default is "zh".
        device: "cuda" or "cpu".
    """
    cmd = [
        sys.executable,
        "-c",
        _WORKER_SOURCE,
        str(video_path),
        model_size,
        language or "auto",
        device,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        log.error("Whisper subprocess failed (exit %d): %s", proc.returncode, proc.stderr[-2000:])
        return []

    payload: list[dict] = []
    for line in proc.stdout.splitlines():
        if line.startswith("__SVIBE_RESULT__"):
            payload = json.loads(line[len("__SVIBE_RESULT__"):])
            break
    return [LyricSegment(time=item["time"], text=item["text"]) for item in payload]
