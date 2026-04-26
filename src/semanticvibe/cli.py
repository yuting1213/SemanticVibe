"""End-to-end CLI: video → mp4 across all 5 stages.

    uv run python -m semanticvibe.cli \\
        --video data/test_videos/demo.mp4 \\
        --output outputs/demo_overlay.mp4 \\
        --style warm_handdrawn \\
        --preview

Use `--keep-intermediates DIR` to dump FeatureSummary / Decision JSON for
inspection or re-rendering via `render_demo`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from semanticvibe.config import STYLE_PRESETS, LLMProvider
from semanticvibe.pipeline import run


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="semanticvibe")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--style",
        default="warm_handdrawn",
        choices=sorted(STYLE_PRESETS.keys()),
        help="Style preset (controls palette + vibe descriptor for the LLM).",
    )
    p.add_argument(
        "--provider",
        choices=["claude", "openai"],
        default=None,
        help="Override the LLM provider; defaults to settings.llm_provider.",
    )
    p.add_argument("--fonts-dir", type=Path, default=Path("data/fonts"))
    p.add_argument("--assets-dir", type=Path, default=Path("data/assets_lib"))
    p.add_argument("--preview", action="store_true", help="720p re-encode for fast iteration.")
    p.add_argument(
        "--keep-intermediates",
        type=Path,
        default=None,
        help="Directory to dump FeatureSummary + Decision JSON.",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    if not args.video.exists():
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 2

    out = run(
        args.video,
        args.output,
        style_preset=args.style,
        fonts_dir=args.fonts_dir,
        assets_dir=args.assets_dir,
        provider=args.provider,  # type: ignore[arg-type]
        preview=args.preview,
        intermediate_dir=args.keep_intermediates,
        device=args.device,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
