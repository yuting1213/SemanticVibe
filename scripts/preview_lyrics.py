"""Whisper preview: peek at the lyrics for a video / audio + cache the result.

Usage:
    # From a video's embedded audio
    python scripts/preview_lyrics.py --video samples/dance.mp4

    # From an independent audio file
    python scripts/preview_lyrics.py --audio samples/song.mp3

What happens:
1. Runs the same Whisper subprocess that `render` uses.
2. Prints each line to the console.
3. Caches the result at `.cache/lyrics/<sha1>.json` (keyed by source path
   + model + language) so re-running for the same input is instant.
4. Also writes a fresh copy to `samples/auto_lyrics.json` for quick
   editing — fix typos / timings, then pass it back via:

       python -m semanticvibe.render \\
           --video samples/dance.mp4 \\
           --lyrics samples/auto_lyrics.json \\
           --out outputs/edited.mp4

If you need to re-run Whisper on a freshly-edited audio file, delete
`.cache/lyrics/<sha1>.json` first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "lyrics"
DEFAULT_AUTO_LYRICS = REPO_ROOT / "samples" / "auto_lyrics.json"


def _cache_key(media: Path, model: str, language: str | None) -> str:
    """Stable hash of (resolved path + model + language)."""
    h = hashlib.sha1()
    h.update(str(media.resolve()).encode("utf-8"))
    h.update(b"|")
    h.update(model.encode("utf-8"))
    h.update(b"|")
    h.update((language or "auto").encode("utf-8"))
    # Bust cache when source file mtime changes.
    try:
        h.update(str(media.stat().st_mtime).encode("utf-8"))
    except FileNotFoundError:
        pass
    return h.hexdigest()[:16]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="preview_lyrics")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, help="Video file (uses embedded audio).")
    src.add_argument("--audio", type=Path, help="Standalone audio file.")
    p.add_argument("--whisper-model", default="large-v3")
    p.add_argument("--language", default=None)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
        help=f"Where to cache JSON results. Default: {DEFAULT_CACHE_DIR}",
    )
    p.add_argument(
        "--auto-lyrics", type=Path, default=DEFAULT_AUTO_LYRICS,
        help="Where to drop a fresh editable copy. Default: samples/auto_lyrics.json",
    )
    p.add_argument(
        "--no-write", action="store_true",
        help="Don't write samples/auto_lyrics.json — print only.",
    )
    p.add_argument("--force", action="store_true",
                   help="Ignore cache and re-run Whisper.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )
    log = logging.getLogger("preview_lyrics")

    media: Path = args.video if args.video else args.audio
    if not media.exists():
        print(f"error: file not found: {media}", file=sys.stderr)
        return 2

    # ---- cache lookup ----
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(media, args.whisper_model, args.language)
    cache_file = args.cache_dir / f"{key}.json"

    if cache_file.exists() and not args.force:
        log.info("[Cache] hit → %s", cache_file)
        lyrics_dicts = json.loads(cache_file.read_text(encoding="utf-8"))
    else:
        log.info("[Whisper] %s on %s (model=%s, lang=%s)…",
                 "cache miss" if not cache_file.exists() else "force re-run",
                 media.name, args.whisper_model, args.language or "auto")
        from semanticvibe.preprocess.whisper_asr import transcribe_audio

        segments = transcribe_audio(
            media,
            model_size=args.whisper_model,
            language=args.language,
            device=args.device,
        )
        lyrics_dicts = [{"time": s.time, "text": s.text} for s in segments]
        cache_file.write_text(
            json.dumps(lyrics_dicts, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log.info("[Cache] wrote → %s", cache_file)

    # ---- console pretty-print ----
    if not lyrics_dicts:
        print("(Whisper produced no segments. Try --whisper-model medium, "
              "or --language auto, or hand-author lyrics in --lyrics JSON.)")
        return 1
    print()
    print(f"=== {len(lyrics_dicts)} lines from {media.name} ===")
    for line in lyrics_dicts:
        print(f"  {line['time']:6.2f}s  {line['text']}")
    print()

    # ---- editable copy ----
    if not args.no_write:
        args.auto_lyrics.parent.mkdir(parents=True, exist_ok=True)
        args.auto_lyrics.write_text(
            json.dumps(lyrics_dicts, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log.info("[Editable] wrote → %s", args.auto_lyrics)
        print(f"Edit `{args.auto_lyrics.relative_to(REPO_ROOT)}` and rerun render with:")
        print(f"  python -m semanticvibe.render \\")
        print(f"      --video {args.video or '<your video>'} \\")
        print(f"      --lyrics {args.auto_lyrics.relative_to(REPO_ROOT)} \\")
        print(f"      --out outputs/edited.mp4")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
