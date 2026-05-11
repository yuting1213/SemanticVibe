"""ffmpeg compositing: base video + transparent WebM overlay → mp4.

The WebM overlay (yuva420p VP9) carries alpha through ffmpeg's overlay
filter directly — no chromakey hack needed.

Audio handling:
  - If `audio_path` is given, that audio is mapped onto the output.
  - Else the base video's existing audio track passes through.
  - When the base video has no audio, the output is silent (audio map
    fails gracefully).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _resolve_ffmpeg() -> str:
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _has_audio_stream(path: Path) -> bool:
    """ffprobe-less probe: read the first few bytes via ffmpeg -i and look
    for an audio stream marker. Returns False on any failure (overlay
    still works without audio)."""
    ffmpeg = _resolve_ffmpeg()
    try:
        out = subprocess.run(
            [ffmpeg, "-i", str(path), "-hide_banner"],
            capture_output=True, text=True, errors="replace",
        )
    except Exception:  # noqa: BLE001
        return False
    return "Audio:" in (out.stderr or "")


def composite_overlay(
    base_video: Path,
    overlay_source: Path,
    output_mp4: Path,
    *,
    audio_path: Path | None = None,
    canvas_size: tuple[int, int] | None = None,
    preview: bool = False,
    fps: int = 30,
) -> Path:
    """Overlay the transparent layer onto base_video; encode to H.264 mp4.

    `overlay_source` can be either:
    - a transparent WebM (yuva420p VP9) — legacy
    - a directory containing `frame_%06d.png` (preferred, since libvpx-vp9
      alpha support in some ffmpeg builds is unreliable). PNG sequence
      preserves the alpha channel through to ffmpeg's overlay filter
      without any intermediate codec drop.

    When `canvas_size` is given the base video is scaled to that size
    first (matches semanticvibe's preview canvas — usually 720p portrait).
    """
    ffmpeg = _resolve_ffmpeg()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    # ---- Decide overlay input args ----
    if overlay_source.is_dir():
        overlay_input = [
            "-framerate", str(fps),
            "-i", str(overlay_source / "frame_%06d.png"),
        ]
    else:
        overlay_input = ["-i", str(overlay_source)]

    # ---- Decide audio source ----
    if audio_path and audio_path.exists():
        audio_input = ["-i", str(audio_path)]
        audio_map = ["-map", "2:a:0?"]
    elif _has_audio_stream(base_video):
        audio_input = []
        audio_map = ["-map", "0:a:0?"]
    else:
        audio_input = []
        audio_map = ["-an"]

    # ---- Build the filter graph ----
    # `format=auto` makes the overlay filter respect the overlay's alpha
    # channel; rgba PNGs flow through unchanged.
    if canvas_size:
        cw, ch = canvas_size
        filter_complex = (
            f"[0:v]scale={cw}:{ch}:force_original_aspect_ratio=decrease,"
            f"pad={cw}:{ch}:(ow-iw)/2:(oh-ih)/2:color=black[bg];"
            f"[1:v]scale={cw}:{ch}[ov];"
            f"[bg][ov]overlay=0:0:format=auto[v]"
        )
    else:
        filter_complex = "[0:v][1:v]overlay=0:0:format=auto[v]"

    cmd = [
        ffmpeg, "-y",
        "-i", str(base_video),
        *overlay_input,
        *audio_input,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        *audio_map,
        "-c:v", "libx264",
        "-preset", "fast" if preview else "medium",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_mp4),
    ]
    log.info("[hf-compositor] ffmpeg overlay → %s", output_mp4)
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        log.error("ffmpeg overlay failed:\n%s", proc.stderr[-3000:])
        raise RuntimeError("compositor: ffmpeg overlay failed")
    return output_mp4
