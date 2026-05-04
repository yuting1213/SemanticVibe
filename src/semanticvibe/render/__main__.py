"""v5+ entry point: video [+ optional lyrics / audio] → mp4.

Three lyrics-input modes, picked automatically by priority:

1. `--lyrics path/to/lyrics.json`  — manual override (highest)
2. `--audio path/to/song.mp3`      — Whisper on independent audio file
3. (default)                       — Whisper on video's embedded audio

Optional `--mix-audio` controls what audio ends up in the final mp4
when the user supplies `--audio`:

- (omitted)         keep the source video's audio
- `replace`         replace video's audio with --audio
- `overlay`         mix video audio + --audio together

Pipeline:

    lyrics  ──┐
              ├─ semantic_align.align()    → list[Highlight]
    video   ──┤
              ├─ pose_detector.detect_person_mask()
              ├─ build_elements.build_decision()
              ├─ render.composite.render_from_decision()
              └─ (optional) ffmpeg audio mixing → final.mp4

Usage:
    # Mode 1: manual lyrics
    python -m semanticvibe.render --video v.mp4 --lyrics lyrics.json --out out.mp4

    # Mode 2: Whisper on independent audio + replace video's audio
    python -m semanticvibe.render --video v.mp4 --audio song.mp3 \\
        --mix-audio replace --out out.mp4

    # Mode 3: full auto (Whisper on video's embedded track)
    python -m semanticvibe.render --video v.mp4 --out out.mp4
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from semanticvibe.build_elements import build_decision
from semanticvibe.lyrics import LyricLine, load_lyrics
from semanticvibe.pose_detector import detect_person_mask
from semanticvibe.render.composite import render_from_decision
from semanticvibe.semantic_align import align


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="semanticvibe.render")
    p.add_argument("--video", type=Path, required=True, help="Input video file.")
    p.add_argument(
        "--lyrics", type=Path, default=None,
        help="Optional manual lyrics JSON. Highest priority — bypasses Whisper.",
    )
    p.add_argument(
        "--audio", type=Path, default=None,
        help="Optional independent audio file. When set, Whisper transcribes "
        "this instead of the video's embedded audio. Combine with --mix-audio "
        "to also splice it into the final mp4's audio track.",
    )
    p.add_argument(
        "--mix-audio", choices=["replace", "overlay"], default=None,
        help="How to combine --audio into the output. 'replace' swaps the "
        "video's audio for --audio; 'overlay' mixes both. Omit to keep the "
        "video's original audio. Ignored when --audio is unset.",
    )
    p.add_argument("--whisper-model", default="large-v3",
                   help="faster-whisper model size: tiny / base / small / "
                   "medium / large-v3 (default).")
    p.add_argument("--language", default=None,
                   help="ISO language code (zh / ja / en / ...). Default: "
                   "Whisper auto-detects.")
    p.add_argument("--out", type=Path, required=True, help="Output mp4 path.")
    p.add_argument("--provider", choices=["rule_based", "claude", "openai"],
                   default="rule_based",
                   help="Highlight aligner. rule_based = offline / free.")
    p.add_argument("--fonts-dir", type=Path, default=Path("data/fonts"))
    p.add_argument("--assets-dir", type=Path, default=Path("data/assets_lib"))
    p.add_argument("--preview", action="store_true",
                   help="720p re-encode for fast iteration.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-fps", type=float, default=2.0,
                   help="Pose detection sample rate (default 2 fps).")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def get_lyrics(args: argparse.Namespace, log: logging.Logger) -> list[LyricLine]:
    """Resolve lyrics by priority: --lyrics > --audio > video embedded."""
    # Priority 1: explicit --lyrics JSON.
    if args.lyrics:
        log.info("[Lyrics] manual override: %s", args.lyrics)
        return load_lyrics(args.lyrics)

    # Priority 2: explicit --audio file via Whisper.
    if args.audio:
        log.info("[Lyrics] Whisper on independent audio: %s", args.audio)
        return _whisper_to_lyric_lines(
            args.audio, args.whisper_model, args.language, args.device,
        )

    # Priority 3: video's embedded audio via Whisper.
    log.info("[Lyrics] Whisper on video's embedded audio: %s", args.video)
    return _whisper_to_lyric_lines(
        args.video, args.whisper_model, args.language, args.device,
    )


def _whisper_to_lyric_lines(
    media_path: Path, model_size: str, language: str | None, device: str,
) -> list[LyricLine]:
    """Run Whisper and convert LyricSegment → LyricLine (no duration)."""
    from semanticvibe.preprocess.whisper_asr import transcribe_audio

    segments = transcribe_audio(
        media_path, model_size=model_size, language=language, device=device,
    )
    return [LyricLine(time=s.time, text=s.text) for s in segments]


def _maybe_mix_audio(
    rendered: Path, audio_source: Path, mode: str, log: logging.Logger,
) -> Path:
    """Post-process audio: replace or overlay --audio into the rendered mp4.

    Uses imageio_ffmpeg's bundled binary so we don't depend on the
    user's PATH. Atomically replaces `rendered` with the mixed version.
    """
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp_out = rendered.with_suffix(".mixed.mp4")

    if mode == "replace":
        cmd = [
            ffmpeg, "-y",
            "-i", str(rendered),
            "-i", str(audio_source),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac",
            "-shortest",
            str(tmp_out),
        ]
    elif mode == "overlay":
        # amix mixes both; -shortest stops at the shorter clip.
        cmd = [
            ffmpeg, "-y",
            "-i", str(rendered),
            "-i", str(audio_source),
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=shortest[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac",
            str(tmp_out),
        ]
    else:
        raise ValueError(f"unknown mix-audio mode: {mode!r}")

    log.info("[Mix] ffmpeg %s: %s + %s → %s",
             mode, rendered.name, audio_source.name, tmp_out.name)
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        log.error("ffmpeg mix failed:\n%s", proc.stderr[-2000:])
        # Don't replace `rendered` — leave the un-mixed render in place.
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
        return rendered

    shutil.move(str(tmp_out), str(rendered))
    return rendered


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )
    log = logging.getLogger("semanticvibe.render")

    # ---- input validation ----
    if not args.video.exists():
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 2
    if args.lyrics and not args.lyrics.exists():
        print(f"error: lyrics not found: {args.lyrics}", file=sys.stderr)
        return 2
    if args.audio and not args.audio.exists():
        print(f"error: audio not found: {args.audio}", file=sys.stderr)
        return 2
    if args.mix_audio and not args.audio:
        print(
            "error: --mix-audio requires --audio (otherwise there's nothing to mix in).",
            file=sys.stderr,
        )
        return 2
    if not args.fonts_dir.exists():
        print(
            f"error: fonts directory not found: {args.fonts_dir}\n"
            "       see data/README.md for font installation.",
            file=sys.stderr,
        )
        return 2

    # ---- Step 1: lyrics (3-mode priority) ----
    lyrics = get_lyrics(args, log)
    if not lyrics:
        print(
            "error: no lyrics produced (Whisper found nothing or input was empty).\n"
            "       try --lyrics with a hand-written JSON, or run\n"
            "       `python scripts/preview_lyrics.py` first.",
            file=sys.stderr,
        )
        return 1
    log.info("Got %d lyric lines.", len(lyrics))

    # ---- Step 2: align → highlights ----
    log.info("Aligning via %s…", args.provider)
    highlights = align(lyrics, provider=args.provider)
    log.info("Got %d highlights:", len(highlights))
    for h in highlights:
        log.info(
            "  t=%5.1fs  strength=%.2f  tag=%-12s  text=%r",
            h.lyric_time, h.strength, h.decoration_tag or "—", h.lyric_text,
        )

    # ---- Step 3: detect person occupancy masks ----
    log.info("Detecting person masks at %.1f fps…", args.sample_fps)
    masks = detect_person_mask(args.video, sample_fps=args.sample_fps)
    log.info("Got %d sampled masks.", len(masks))

    # ---- Step 4: render canvas size ----
    from moviepy import VideoFileClip

    with VideoFileClip(str(args.video)) as src:
        src_w, src_h = src.w, src.h
    if args.preview and src_h > 720:
        scale = 720 / src_h
        canvas_size = (int(src_w * scale), 720)
        canvas_size = (
            canvas_size[0] - (canvas_size[0] % 2),
            canvas_size[1] - (canvas_size[1] % 2),
        )
    else:
        canvas_size = (src_w - (src_w % 2), src_h - (src_h % 2))
    log.info("Canvas: %dx%d", *canvas_size)

    # ---- Step 5: build Decision ----
    log.info("Building Decision…")
    decision = build_decision(
        highlights, person_masks=masks, canvas_size=canvas_size,
        fonts_dir=args.fonts_dir, seed=args.seed,
    )
    log.info(
        "Decision: %d elements (%d text, %d decoration)",
        len(decision.elements),
        sum(1 for e in decision.elements if e.type == "text"),
        sum(1 for e in decision.elements if e.type == "decoration"),
    )

    # ---- Step 6: render ----
    log.info("Rendering to %s…", args.out)
    out = render_from_decision(
        args.video, decision, args.out,
        fonts_dir=args.fonts_dir,
        assets_dir=args.assets_dir if args.assets_dir.exists() else None,
        preview=args.preview,
    )

    # ---- Step 7: optional audio mix ----
    if args.audio and args.mix_audio:
        out = _maybe_mix_audio(out, args.audio, args.mix_audio, log)

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
