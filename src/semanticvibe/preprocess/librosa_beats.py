"""Beat tracking + chorus segmentation via librosa.

librosa's default backend (libsndfile) does not handle .mp4. On Windows the
audioread fallback also has no backend. We extract audio to a temp .wav
using imageio-ffmpeg (already a project dep) and feed librosa the wav.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np


def extract_wav(video_path: Path, sr: int = 22050, *, loudnorm: bool = True) -> Path:
    """Extract mono PCM audio from `video_path` into a cached temp .wav.

    Args:
        loudnorm: If True (default), normalise to ~-16 LUFS via ffmpeg's
            EBU R128 loudnorm filter. Quiet phone recordings (mean
            -35 dB and below) otherwise fall under Whisper's speech
            threshold and return zero segments.
    """
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Cache key includes the loudnorm flag so the two variants don't collide.
    key = f"{video_path.resolve()}|{sr}|ln={int(loudnorm)}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    wav = Path(tempfile.gettempdir()) / f"semanticvibe_{h}_{sr}.wav"
    if wav.exists():
        return wav

    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sr), "-sample_fmt", "s16",
    ]
    if loudnorm:
        # EBU R128 single-pass loudnorm. Two-pass would be more accurate but
        # we don't need broadcast-quality consistency, just enough headroom
        # for Whisper to find the speech.
        cmd += ["-af", "loudnorm=I=-16:LRA=11:TP=-1.5"]
    cmd.append(str(wav))
    subprocess.run(cmd, check=True, capture_output=True)
    return wav


# Back-compat alias for the older private name used inside this module.
_extract_wav = extract_wav


@lru_cache(maxsize=8)
def _load_audio(video_path_str: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load mono audio at `sr` Hz. Cached so beat + chorus share one decode."""
    src = Path(video_path_str)
    if src.suffix.lower() in {".wav", ".flac", ".ogg"}:
        y, sr_ret = librosa.load(str(src), sr=sr, mono=True)
    else:
        wav = _extract_wav(src, sr)
        y, sr_ret = librosa.load(str(wav), sr=sr, mono=True)
    return y, sr_ret


def detect_beats(video_path: Path) -> list[float]:
    """Beat onsets in seconds from start of audio track."""
    y, sr = _load_audio(str(video_path))
    _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    times = librosa.frames_to_time(beat_frames, sr=sr)
    # Sort + dedupe defensively — beat_track is monotonic but downstream code
    # uses these as a contract (FeatureSummary validates monotonicity).
    return sorted({float(t) for t in times})


def detect_chorus_segments(video_path: Path) -> list[tuple[float, float]]:
    """Identify likely chorus regions by structural repetition.

    Heuristic: compute a self-similarity matrix on chroma+MFCC, segment
    the audio into ~10 sections, then return the segments whose feature
    centroid recurs most often (i.e. the most-repeated motif). Returns
    an empty list if the audio is too short to segment meaningfully.
    """
    y, sr = _load_audio(str(video_path))
    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 20.0:
        return []  # too short to call a "chorus"

    hop_length = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
    feats = np.vstack([chroma, mfcc])

    # Beat-synchronous aggregation reduces noise vs. raw frames.
    _tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, units="frames")
    if len(beats) < 8:
        return []
    feats_sync = librosa.util.sync(feats, beats, aggregate=np.median)

    # Agglomerative segmentation into ~k sections, k chosen by length.
    k = max(4, min(10, int(duration // 15)))
    try:
        bounds = librosa.segment.agglomerative(feats_sync, k=k)
    except Exception:
        return []
    bound_frames = beats[bounds]
    bound_frames = np.append(bound_frames, beats[-1])
    bound_times = librosa.frames_to_time(bound_frames, sr=sr, hop_length=hop_length)

    # Cluster segment centroids; the largest cluster is the chorus motif.
    centroids = []
    for i in range(len(bound_frames) - 1):
        s, e = bounds[i], (bounds[i + 1] if i + 1 < len(bounds) else len(feats_sync[0]))
        if e <= s:
            centroids.append(np.zeros(feats_sync.shape[0]))
            continue
        centroids.append(feats_sync[:, s:e].mean(axis=1))
    if not centroids:
        return []
    centroids_arr = np.vstack(centroids)

    # Cosine-similarity-based clustering: count near-neighbours per segment.
    norms = np.linalg.norm(centroids_arr, axis=1, keepdims=True) + 1e-9
    normed = centroids_arr / norms
    sim = normed @ normed.T
    threshold = 0.85
    counts = (sim > threshold).sum(axis=1)
    if counts.max() < 2:
        return []  # nothing repeats — no chorus

    chorus_idxs = np.where(counts == counts.max())[0]

    # Merge consecutive chorus segments and emit (start, end).
    out: list[tuple[float, float]] = []
    if len(chorus_idxs) == 0:
        return []
    cur_start = float(bound_times[chorus_idxs[0]])
    cur_end = float(bound_times[chorus_idxs[0] + 1])
    for j in chorus_idxs[1:]:
        seg_start = float(bound_times[j])
        seg_end = float(bound_times[j + 1])
        if seg_start - cur_end < 1e-3:
            cur_end = seg_end
        else:
            out.append((cur_start, cur_end))
            cur_start, cur_end = seg_start, seg_end
    out.append((cur_start, cur_end))
    # Filter degenerate segments.
    return [(s, e) for s, e in out if e - s >= 3.0]
