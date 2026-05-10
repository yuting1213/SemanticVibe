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

import json

from semanticvibe.build_elements import build_decision
from semanticvibe.lyrics import LyricLine, load_lyrics
from semanticvibe.pose_detector import detect_person_mask
from semanticvibe.render.composite import render_from_decision
from semanticvibe.schemas.decision import Decision, GlobalStyle
from semanticvibe.semantic_align import align_lyrics


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
    p.add_argument("--provider", choices=["rule_based", "claude", "ollama"],
                   default="rule_based",
                   help="Highlight aligner. rule_based = offline keyword dict; "
                   "claude routes through .cache/alignment for repeat runs; "
                   "ollama hits a local Ollama server (no API key needed).")
    p.add_argument("--ollama-model", default="gemma3:4b",
                   help="Ollama model tag (default gemma3:4b). Run "
                   "`ollama list` to see what you have pulled.")
    p.add_argument("--ollama-host", default=None,
                   help="Ollama server URL (default http://localhost:11434).")
    p.add_argument("--beat-sync", dest="beat_sync", action="store_true", default=True,
                   help="Snap lyric/decoration timings to detected beats and "
                   "promote downbeat hits to punchier entry animations. "
                   "Default ON. Source: --audio if given, else the video's "
                   "embedded audio track.")
    p.add_argument("--no-beat-sync", dest="beat_sync", action="store_false",
                   help="Disable beat-sync (deterministic, no librosa pass).")
    from semanticvibe.style import default_style_name, style_names
    p.add_argument("--style", choices=style_names(),
                   default=default_style_name(),
                   help="Visual style preset from assets/styles.json.")
    p.add_argument("--subtitle-style", choices=["outlined", "banner", "hero"],
                   default=None,
                   help="Lyric rendering mode: 'outlined' (v10 default — "
                   "transparent thick-outlined text, no chip background), "
                   "'banner' (legacy rounded chip), 'hero' (one big centred "
                   "glyph + small per-line text). Default comes from preset.")
    p.add_argument("--song-title", type=str, default=None,
                   help="Optional song title; passed to Claude for context "
                   "and folded into the alignment cache key.")
    p.add_argument("--elements-json", type=Path, default=None,
                   help="Skip alignment + build_decision and load a pre-built "
                   "Decision JSON (or a flat list of element dicts) directly. "
                   "Useful for hand-editing what build_elements_from_lyrics "
                   "produced.")
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

    # ---- Short-circuit: --elements-json bypasses lyrics + alignment ----
    if args.elements_json:
        if not args.elements_json.exists():
            print(f"error: elements json not found: {args.elements_json}",
                  file=sys.stderr)
            return 2
        log.info("[elements-json] loading pre-built Decision: %s",
                 args.elements_json)
        raw = json.loads(args.elements_json.read_text(encoding="utf-8"))
        # Accept either {"elements": [...], "global_style": {...}} or a bare list.
        if isinstance(raw, list):
            raw = {
                "elements": raw,
                "global_style": {
                    "color_palette": ["#FF6B9D", "#E63946", "#FFFFFF"],
                    "vibe": "v6 elements-json mode",
                },
            }
        decision = Decision.model_validate(raw)
        return _render_decision_path(args, decision, log)

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
    log.info("Aligning via %s%s…", args.provider,
             f" (song={args.song_title!r})" if args.song_title else "")
    align_result = align_lyrics(
        lyrics, provider=args.provider, song_title=args.song_title,
        ollama_model=args.ollama_model, ollama_host=args.ollama_host,
    )
    highlights = align_result.highlights
    log.info("Got %d highlights (%d hooks, %d non-hooks):",
             len(highlights),
             sum(1 for h in highlights if h.is_hook),
             len(align_result.non_hooks))
    for h in highlights:
        log.info(
            "  t=%5.1fs  hook=%-5s  tags=%-22s  text=%r",
            h.time, str(h.is_hook), str(h.tags), h.text,
        )

    # ---- Step 3: detect person occupancy masks ----
    log.info("Detecting person masks at %.1f fps…", args.sample_fps)
    masks = detect_person_mask(args.video, sample_fps=args.sample_fps)
    log.info("Got %d sampled masks.", len(masks))

    canvas_size = _measure_canvas(args, log)

    # ---- Step 5: build Decision ----
    log.info("Building Decision (style=%s, subtitle=%s, beat_sync=%s)…",
             args.style, args.subtitle_style or "<preset default>",
             args.beat_sync)
    # Beat-detection audio source: --audio if given (preferred — usually
    # cleaner), else the video's embedded track.
    beat_audio = args.audio if args.audio else args.video
    decision = build_decision(
        highlights, person_masks=masks, canvas_size=canvas_size,
        fonts_dir=args.fonts_dir, seed=args.seed,
        style=args.style, subtitle_style=args.subtitle_style,
        audio_path=beat_audio if args.beat_sync else None,
        beat_sync=args.beat_sync,
    )
    log.info(
        "Decision: %d elements (%d text, %d outlined, %d banner, %d decoration, %d hero)",
        len(decision.elements),
        sum(1 for e in decision.elements if e.type == "text"),
        sum(1 for e in decision.elements if e.type == "subtitle_outlined"),
        sum(1 for e in decision.elements if e.type == "subtitle_banner"),
        sum(1 for e in decision.elements if e.type == "decoration"),
        sum(1 for e in decision.elements if e.type == "hero_text"),
    )

    return _render_decision_path(args, decision, log)


def _measure_canvas(args, log) -> tuple[int, int]:
    """Compute the render canvas size, respecting --preview."""
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
    return canvas_size


def _render_decision_path(args, decision: Decision, log: logging.Logger) -> int:
    """Encode + optional audio mix. Shared by the lyrics-driven and
    --elements-json branches."""
    log.info("Rendering to %s…", args.out)
    out = render_from_decision(
        args.video, decision, args.out,
        fonts_dir=args.fonts_dir,
        assets_dir=args.assets_dir if args.assets_dir.exists() else None,
        preview=args.preview,
    )
    if args.audio and args.mix_audio:
        out = _maybe_mix_audio(out, args.audio, args.mix_audio, log)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
