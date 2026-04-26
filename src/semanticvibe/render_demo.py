"""Week 1 deliverable: render an mp4 from a video + hand-written Decision JSON.

    uv run python -m semanticvibe.render_demo \\
        --video data/test_videos/sample_30s.mp4 \\
        --json examples/hand_written_decision.json \\
        --output outputs/week1_demo.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from semanticvibe.render.composite import render_from_decision
from semanticvibe.schemas.decision import Decision


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="semanticvibe-render-demo")
    p.add_argument("--video", type=Path, required=True, help="Input video file.")
    p.add_argument(
        "--json", type=Path, required=True, dest="decision_json", help="Decision JSON file."
    )
    p.add_argument("--output", type=Path, required=True, help="Output .mp4 path.")
    p.add_argument(
        "--fonts-dir",
        type=Path,
        default=Path("data/fonts"),
        help="Directory containing the .ttf fonts referenced in the Decision.",
    )
    p.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("data/assets_lib"),
        help="Decoration asset library root. Decorations are skipped if absent.",
    )
    p.add_argument(
        "--preview",
        action="store_true",
        help="Downscale to 720p for faster iteration (spec §10 mitigation).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.video.exists():
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 2
    if not args.decision_json.exists():
        print(f"error: decision JSON not found: {args.decision_json}", file=sys.stderr)
        return 2
    if not args.fonts_dir.exists():
        print(
            f"error: fonts directory not found: {args.fonts_dir}\n"
            "       see data/README.md for the font download instructions.",
            file=sys.stderr,
        )
        return 2

    with args.decision_json.open(encoding="utf-8") as f:
        decision = Decision.model_validate(json.load(f))

    out = render_from_decision(
        args.video,
        decision,
        args.output,
        fonts_dir=args.fonts_dir,
        assets_dir=args.assets_dir if args.assets_dir.exists() else None,
        preview=args.preview,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
