"""composition.html → transparent WebM overlay.

Two-step process:
  1. Run `playground/hf_workspace/render_frames.js` (Puppeteer) to dump
     a PNG sequence with `omitBackground: true` so alpha is preserved.
  2. Encode the sequence to WebM with libvpx-vp9 + yuva420p so the
     resulting video carries alpha through to the compositor step.

Returns the final WebM path. Frame directory is kept (caller can clean
up via the parent temp dir) so a render mishap is debuggable.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


# Repo layout: <repo>/playground/hf_workspace/{render_frames.js,node_modules/}
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_HF_WORKSPACE = _REPO_ROOT / "playground" / "hf_workspace"


def _resolve_node() -> str:
    """Find a usable node executable; prefer system PATH."""
    candidates = ["node", r"C:\Program Files\nodejs\node.exe"]
    for c in candidates:
        if shutil.which(c) or Path(c).exists():
            return c
    raise RuntimeError("node executable not found — install Node.js >= 22")


def _resolve_ffmpeg() -> str:
    """Find ffmpeg. Prefer system, fall back to imageio-ffmpeg bundle so
    we don't break on machines without a PATH-installed ffmpeg."""
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def capture_frames(
    html_path: Path,
    frames_dir: Path,
    *,
    width: int,
    height: int,
    fps: int,
    duration: float,
    node_exe: str | None = None,
) -> int:
    """Drive the Puppeteer script. Returns the number of frames captured."""
    if not _HF_WORKSPACE.exists():
        raise RuntimeError(
            f"hf_workspace missing at {_HF_WORKSPACE} — "
            f"run `cd playground && mkdir hf_workspace && cd hf_workspace && "
            f"npm init -y && npm i puppeteer`"
        )
    script = _HF_WORKSPACE / "render_frames.js"
    if not script.exists():
        raise RuntimeError(f"{script} not found")

    frames_dir.mkdir(parents=True, exist_ok=True)
    node = node_exe or _resolve_node()
    cmd = [
        node, str(script),
        "--html", str(html_path),
        "--out", str(frames_dir),
        "--width", str(width),
        "--height", str(height),
        "--fps", str(fps),
        "--duration", f"{duration:.3f}",
    ]
    log.info(
        "[hf-renderer] node render_frames.js: %dx%d %dfps × %.2fs → %s",
        width, height, fps, duration, frames_dir,
    )
    env = {**os.environ, "PUPPETEER_DISABLE_HEADLESS_WARNING": "1"}
    proc = subprocess.run(
        cmd, cwd=str(_HF_WORKSPACE), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    if proc.returncode != 0:
        log.error("Puppeteer failed:\n%s\n---stderr---\n%s",
                  proc.stdout[-2000:], proc.stderr[-2000:])
        raise RuntimeError(f"Puppeteer rendering failed (exit {proc.returncode})")
    # Last stdout line is the JSON summary.
    last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "{}"
    try:
        summary = json.loads(last_line)
        log.info("[hf-renderer] %s", summary)
        return int(summary.get("frames", 0))
    except json.JSONDecodeError:
        log.warning("Could not parse Puppeteer summary: %r", last_line)
        return len(list(frames_dir.glob("frame_*.png")))


def encode_overlay_webm(
    frames_dir: Path,
    output_path: Path,
    *,
    fps: int,
    quality: str = "good",
) -> Path:
    """Encode the PNG sequence to WebM with libvpx-vp9 + yuva420p alpha.

    Notes on the encode args:
    - `-pix_fmt yuva420p` is the magic that preserves the alpha channel.
      Without it ffmpeg picks yuv420p and drops alpha silently.
    - `-row-mt 1 -threads 0` lets libvpx-vp9 use all cores for the slow
      encode path.
    - `-crf 28` is balanced — bump down for higher quality, up for smaller
      files. For overlay video the perceived quality threshold is forgiving
      (we never see the encode artifacts when alpha-composited).
    - `-deadline good -cpu-used 4` is the libvpx 'realtime-ish' preset;
      faster than 'best', acceptable artifacts for overlay text.
    """
    ffmpeg = _resolve_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%06d.png"),
        "-c:v", "libvpx-vp9",
        "-pix_fmt", "yuva420p",
        "-auto-alt-ref", "0",
        "-deadline", quality,
        "-cpu-used", "4",
        "-row-mt", "1",
        "-threads", "0",
        "-crf", "28",
        "-b:v", "0",
        "-an",  # overlay has no audio; we add base video's audio downstream
        str(output_path),
    ]
    log.info("[hf-renderer] ffmpeg encoding WebM/VP9-alpha → %s", output_path)
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        log.error("ffmpeg encode failed:\n%s", proc.stderr[-3000:])
        raise RuntimeError("WebM/VP9-alpha encode failed")
    return output_path


def render_overlay_webm(
    html_path: Path,
    output_webm: Path,
    *,
    canvas_size: tuple[int, int],
    fps: int,
    duration: float,
    frames_dir: Path | None = None,
    keep_frames: bool = False,
) -> Path:
    """Full pipeline: composition.html → frames → WebM with alpha."""
    width, height = canvas_size
    if frames_dir is None:
        frames_dir = output_webm.parent / "_frames"
    capture_frames(
        html_path, frames_dir,
        width=width, height=height, fps=fps, duration=duration,
    )
    encode_overlay_webm(frames_dir, output_webm, fps=fps)
    if not keep_frames:
        try:
            shutil.rmtree(frames_dir)
        except OSError as exc:  # noqa: BLE001
            log.warning("Could not clean frames dir %s (%s)", frames_dir, exc)
    return output_webm
