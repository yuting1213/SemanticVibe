"""Generate placeholder decoration PNGs into data/assets_lib/.

Week 1 needs *some* assets for `near_text_id` decorations to render. The real
hand-curated / SDXL-generated 200-300-asset library is Stage 4 (Week 4) work.
These four procedurally-drawn shapes are enough to validate the decoration
render path end-to-end.

Run from project root:
    uv run python scripts/generate_placeholder_assets.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

ASSETS_DIR = Path(__file__).resolve().parent.parent / "data" / "assets_lib"
SIZE = 128


def draw_sparkle(path: Path, color: str = "#FFE066") -> None:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = SIZE // 2
    long_r, short_r = 56, 12
    pts = []
    for i in range(8):
        angle = i * math.pi / 4
        r = long_r if i % 2 == 0 else short_r
        pts.append(
            (cx + r * math.cos(angle - math.pi / 2), cy + r * math.sin(angle - math.pi / 2))
        )
    d.polygon(pts, fill=color, outline="#264653", width=3)
    img.save(path)


def draw_heart(path: Path, color: str = "#E76F51") -> None:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pts = []
    for i in range(60):
        t = i / 59 * 2 * math.pi
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((SIZE / 2 + x * 3.2, SIZE / 2 + y * 3.2))
    d.polygon(pts, fill=color, outline="#264653", width=3)
    img.save(path)


def draw_star5(path: Path, color: str = "#2A9D8F") -> None:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = SIZE // 2
    long_r, short_r = 54, 22
    pts = []
    for i in range(10):
        angle = i * math.pi / 5 - math.pi / 2
        r = long_r if i % 2 == 0 else short_r
        pts.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    d.polygon(pts, fill=color, outline="#264653", width=3)
    img.save(path)


def draw_dot(path: Path, color: str = "#F4A261") -> None:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((24, 24, SIZE - 24, SIZE - 24), fill=color, outline="#264653", width=4)
    img.save(path)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("sparkle.png", draw_sparkle, ["sparkle", "celebration", "shine"]),
        ("heart.png", draw_heart, ["heart", "love"]),
        ("star.png", draw_star5, ["star", "musical-note"]),
        ("dot.png", draw_dot, ["dot", "circle"]),
    ]
    metadata = []
    for filename, draw_fn, tags in jobs:
        p = ASSETS_DIR / filename
        draw_fn(p)
        metadata.append(
            {
                "filename": filename,
                "tags": tags,
                "license": "self-drawn (procedural)",
                "source": "internal placeholder for Week 1",
            }
        )
        print(f"  {filename}  ({p.stat().st_size:,} bytes)")

    meta_path = ASSETS_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {meta_path} with {len(metadata)} entries")


if __name__ == "__main__":
    main()
