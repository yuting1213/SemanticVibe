"""One-off: run Whisper directly on a video, with and without VAD, to debug
why ASR returns nothing. Not part of the production pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

WORKER_SOURCE = """
import json, sys
from faster_whisper import WhisperModel
video_path, vad = sys.argv[1], sys.argv[2] == "1"
m = WhisperModel("large-v3", device="cuda", compute_type="float16")
segs, _info = m.transcribe(
    video_path, language="zh", beam_size=5, vad_filter=vad
)
out = [
    {"time": float(s.start), "text": (s.text or "").strip()}
    for s in segs
    if (s.text or "").strip()
]
print("__R__" + json.dumps(out, ensure_ascii=False))
"""


def run(video: Path, vad: bool) -> list[dict]:
    cmd = [sys.executable, "-c", WORKER_SOURCE, str(video), "1" if vad else "0"]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stderr[-1000:], file=sys.stderr)
        return []
    for line in proc.stdout.splitlines():
        if line.startswith("__R__"):
            return json.loads(line[5:])
    return []


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, default=Path("data/test_videos/demo.mp4"))
    args = p.parse_args()

    for vad in (True, False):
        label = "with VAD" if vad else "no VAD"
        segs = run(args.video, vad)
        print(f"=== {label}: {len(segs)} segments ===")
        for s in segs[:20]:
            print(f"  {s['time']:6.2f}s  {s['text']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
