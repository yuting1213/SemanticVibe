"""v5 entry point: lyrics + video → mp4.

Pipeline:

    lyrics.json  ──┐
                   ├─ semantic_align.align()      → list[Highlight]
    video       ──┤
                   ├─ pose_detector.detect_person_mask() → time → bool mask
                   ├─ build_elements.build_decision()    → Decision
                   └─ render.composite.render_from_decision() → mp4

Nothing is hardcoded. Text content comes from lyrics, positions come
from pose masks, decorations come from tag vocabulary matching.

Usage:
    python -m semanticvibe.render \\
        --video samples/dance.mp4 \\
        --lyrics samples/lyrics_mosimosi.json \\
        --provider rule_based \\
        --out output_v5.mp4
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from semanticvibe.build_elements import build_decision
from semanticvibe.pose_detector import detect_person_mask
from semanticvibe.render.composite import render_from_decision
from semanticvibe.semantic_align import align, load_lyrics


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="semanticvibe.render")
    p.add_argument("--video", type=Path, required=True, help="Input video file.")
    p.add_argument("--lyrics", type=Path, required=True,
                   help="JSON list of {time: float, text: str} lyric lines.")
    p.add_argument("--out", type=Path, required=True, help="Output mp4 path.")
    p.add_argument("--provider", choices=["rule_based", "claude", "openai"],
                   default="rule_based",
                   help="Highlight aligner. rule_based is offline + free; "
                        "claude/openai need API keys and produce richer picks.")
    p.add_argument("--fonts-dir", type=Path, default=Path("data/fonts"))
    p.add_argument("--assets-dir", type=Path, default=Path("data/assets_lib"))
    p.add_argument("--preview", action="store_true",
                   help="720p re-encode for fast iteration.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for animation assignment.")
    p.add_argument("--sample-fps", type=float, default=2.0,
                   help="Pose detection sample rate (default 2 fps).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )
    log = logging.getLogger("semanticvibe.render")

    if not args.video.exists():
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 2
    if not args.lyrics.exists():
        print(f"error: lyrics not found: {args.lyrics}", file=sys.stderr)
        return 2
    if not args.fonts_dir.exists():
        print(
            f"error: fonts directory not found: {args.fonts_dir}\n"
            "       see data/README.md for font installation.",
            file=sys.stderr,
        )
        return 2

    # ------ Step 1: lyrics → highlights ------
    log.info("Loading lyrics from %s", args.lyrics)
    lyrics = load_lyrics(args.lyrics)
    log.info("Aligning %d lyric lines via %s", len(lyrics), args.provider)
    highlights = align(lyrics, provider=args.provider)
    log.info("Got %d highlights", len(highlights))
    for h in highlights:
        log.info(
            "  t=%5.1fs  strength=%.2f  tag=%-12s  text=%r",
            h.lyric_time, h.strength, h.decoration_tag or "—", h.lyric_text,
        )

    # ------ Step 2: detect person occupancy masks ------
    log.info("Detecting person masks at %.1f fps…", args.sample_fps)
    masks = detect_person_mask(args.video, sample_fps=args.sample_fps)
    log.info("Got %d sampled masks", len(masks))

    # ------ Step 3: get final canvas size (matches what render uses) ------
    from moviepy import VideoFileClip

    with VideoFileClip(str(args.video)) as src:
        src_w, src_h = src.w, src.h
    if args.preview and src_h > 720:
        scale = 720 / src_h
        canvas_size = (int(src_w * scale), 720)
        # Round to even — same logic as composite.py
        canvas_size = (
            canvas_size[0] - (canvas_size[0] % 2),
            canvas_size[1] - (canvas_size[1] % 2),
        )
    else:
        canvas_size = (
            src_w - (src_w % 2),
            src_h - (src_h % 2),
        )
    log.info("Render canvas size: %dx%d", *canvas_size)

    # ------ Step 4: build the Decision ------
    log.info("Building Decision from highlights + masks…")
    decision = build_decision(
        highlights,
        person_masks=masks,
        canvas_size=canvas_size,
        fonts_dir=args.fonts_dir,
        seed=args.seed,
    )
    log.info(
        "Decision: %d elements (%d text, %d decoration)",
        len(decision.elements),
        sum(1 for e in decision.elements if e.type == "text"),
        sum(1 for e in decision.elements if e.type == "decoration"),
    )

    # ------ Step 5: render ------
    log.info("Rendering to %s…", args.out)
    out = render_from_decision(
        args.video,
        decision,
        args.out,
        fonts_dir=args.fonts_dir,
        assets_dir=args.assets_dir if args.assets_dir.exists() else None,
        preview=args.preview,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
