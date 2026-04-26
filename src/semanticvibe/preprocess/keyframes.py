"""Keyframe selection — pick a sparse set of frames that summarise the video.

Strategy: scene-change detection by histogram delta + uniform fallback. Output
is a list of saved JPEG paths, ready to feed BLIP-2.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np


def select_keyframes(
    video_path: Path,
    *,
    target_count: int = 12,
    output_dir: Path | None = None,
    min_gap_seconds: float = 1.0,
) -> list[Path]:
    """Pick ~`target_count` frames; save as JPEG; return their paths.

    Args:
        target_count: Soft upper bound. We may emit fewer if the video is
            short or scene-cut detection finds nothing.
        output_dir: Where to write JPEGs. Defaults to a fresh temp dir whose
            lifetime is the caller's problem (we don't clean up).
        min_gap_seconds: Reject candidates closer than this to the previous
            kept frame, to avoid clusters at fast cuts.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="semanticvibe_keyframes_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        cap.release()
        return []

    # Walk frames at ~2 fps; compute histogram deltas to score scene cuts.
    sample_step = max(1, int(round(fps / 2.0)))
    prev_hist = None
    scores: list[tuple[int, float]] = []  # (frame_idx, delta)

    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            delta = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR))
            scores.append((idx, delta))
        prev_hist = hist
        idx += sample_step

    # Top-N by score, then enforce min-gap; pad with uniform sampling.
    scores.sort(key=lambda x: x[1], reverse=True)
    chosen: list[int] = []
    min_gap_frames = int(round(min_gap_seconds * fps))
    for frame_idx, _ in scores:
        if all(abs(frame_idx - c) >= min_gap_frames for c in chosen):
            chosen.append(frame_idx)
        if len(chosen) >= target_count:
            break

    if len(chosen) < target_count:
        # Pad with uniformly spaced frames so we never return zero on flat content.
        padding_n = target_count - len(chosen)
        for k in range(padding_n):
            f = int((k + 0.5) * n_frames / padding_n)
            if all(abs(f - c) >= min_gap_frames for c in chosen):
                chosen.append(f)

    chosen = sorted(set(chosen))[:target_count]

    out_paths: list[Path] = []
    for i, frame_idx in enumerate(chosen):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ts = frame_idx / fps
        out_path = output_dir / f"kf_{i:03d}_t{ts:.2f}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        out_paths.append(out_path)

    cap.release()
    return out_paths


def frame_time(keyframe_path: Path) -> float:
    """Recover the timestamp encoded in a keyframe filename (kf_NNN_tSEC.jpg)."""
    stem = keyframe_path.stem
    # Format: kf_<idx>_t<seconds>
    try:
        return float(stem.split("_t", 1)[1])
    except (IndexError, ValueError):
        return 0.0
