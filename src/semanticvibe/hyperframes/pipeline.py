"""Top-level orchestrator — Decision + base video → final mp4 via Hyperframes.

Drop-in replacement for `semanticvibe.render.composite.render_from_decision`.
The MoviePy path is preserved so `--renderer moviepy` still works for
ablation studies (per spec Phase-2 completion criteria).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from semanticvibe.hyperframes.adapter import build_composition
from semanticvibe.hyperframes.compositor import composite_overlay
from semanticvibe.hyperframes.overlay_renderer import (
    capture_frames,
    render_overlay_webm,
)
from semanticvibe.schemas.decision import Decision

log = logging.getLogger(__name__)


def render_from_decision_hyperframes(
    base_video: Path,
    decision: Decision,
    output_mp4: Path,
    *,
    canvas_size: tuple[int, int],
    fonts_dir: Path | None = None,  # unused; web fonts via CDN/system
    assets_dir: Path | None = None,  # unused; assets resolved through v6 retriever
    fps: int = 30,
    audio_path: Path | None = None,
    preview: bool = False,
    workdir: Path | None = None,
    keep_workdir: bool = False,
) -> Path:
    """End-to-end: Decision → composition.html → overlay.webm → output.mp4.

    `workdir` is the temp folder where composition.html / frames / overlay.webm
    are stored. When None, a fresh tempfile.mkdtemp is created and cleaned
    unless `keep_workdir=True`.
    """
    base_video = Path(base_video)
    output_mp4 = Path(output_mp4)

    cleanup_workdir = False
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="semanticvibe_hf_"))
        cleanup_workdir = not keep_workdir
    workdir.mkdir(parents=True, exist_ok=True)
    log.info("[hf-pipeline] workdir = %s", workdir)

    # 1) Adapter: Decision → composition.html (+ assets/)
    comp = build_composition(
        decision, canvas_size=canvas_size, out_dir=workdir, fps=fps,
    )

    # 2) Renderer: composition.html → PNG sequence (alpha preserved).
    # Skipping the intermediate transparent WebM because libvpx-vp9's alpha
    # support is unreliable on some Windows ffmpeg builds (silently drops
    # to yuv420p). Feeding PNGs directly to the compositor's overlay
    # filter is simpler AND faster (one less encode pass).
    frames_dir = workdir / "_frames"
    capture_frames(
        comp.html_path, frames_dir,
        width=canvas_size[0], height=canvas_size[1],
        fps=fps, duration=comp.duration_sec,
    )

    # 3) Compositor: base + frames → mp4 (+ audio)
    composite_overlay(
        base_video, frames_dir, output_mp4,
        audio_path=audio_path, canvas_size=canvas_size,
        preview=preview, fps=fps,
    )

    if cleanup_workdir:
        try:
            import shutil
            shutil.rmtree(workdir)
        except OSError as exc:  # noqa: BLE001
            log.warning("Could not clean workdir %s (%s)", workdir, exc)

    return output_mp4
