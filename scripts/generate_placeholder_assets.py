"""Generate placeholder decoration PNGs into data/assets_lib/.

The first batch of four (sparkle / heart / star / dot) covers the decoration
render path end-to-end. The second batch adds the cute / playful primitives
seen in baseline reference videos: mini-heart for confetti scatters, burst
lines for emphasis, scribble arrows, fire spark, exclaim burst.

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


# ----------------------------------------------------------------------------
# Second batch — cute / playful primitives for confetti and emphasis
# ----------------------------------------------------------------------------


def draw_mini_heart(path: Path, color: str = "#FF66A8") -> None:
    """Smaller, outline-only heart for scatter confetti. The composite stage
    will tint these into palette colours, so the source colour barely matters
    — we just need a clean shape with a transparent background.
    """
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    pts = []
    for i in range(80):
        t = i / 79 * 2 * math.pi
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((SIZE / 2 + x * 2.3, SIZE / 2 + y * 2.3 - 4))
    d.polygon(pts, fill=color, outline="#FFFFFF", width=4)
    img.save(path)


def draw_burst(path: Path, color: str = "#FFE066") -> None:
    """Manga-style emphasis burst — radiating lines from a central point.
    Place near a text to add 'this just landed' energy.
    """
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = SIZE // 2
    n_lines = 14
    for i in range(n_lines):
        angle = (i / n_lines) * 2 * math.pi
        # Tapered line: short inner + long outer, so it reads as a starburst.
        inner_r = 14
        outer_r = 56 if i % 2 == 0 else 42
        x1 = cx + inner_r * math.cos(angle)
        y1 = cy + inner_r * math.sin(angle)
        x2 = cx + outer_r * math.cos(angle)
        y2 = cy + outer_r * math.sin(angle)
        d.line([(x1, y1), (x2, y2)], fill=color, width=5)
        # White inner stroke for the glow effect.
        d.line([(x1, y1), (x2, y2)], fill="#FFFFFF", width=2)
    img.save(path)


def draw_arrow(path: Path, color: str = "#FF80AB") -> None:
    """Scribbled arrow — wobbly hand-drawn pointing-down-right arrow.
    Useful for indicating a subject ("look here!" energy).
    """
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Wobbly shaft using a sequence of slightly offset segments.
    pts = []
    for i in range(20):
        t = i / 19
        x = 18 + 80 * t + math.sin(t * math.pi * 4) * 3
        y = 18 + 80 * t + math.cos(t * math.pi * 3) * 4
        pts.append((x, y))
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        d.line([(x1, y1), (x2, y2)], fill=color, width=6)
    # Arrowhead — two strokes off the tip.
    tip = pts[-1]
    head_len = 22
    for angle_off in (math.radians(150), math.radians(-150 - 40)):
        a = math.atan2(pts[-1][1] - pts[-3][1], pts[-1][0] - pts[-3][0]) + angle_off
        d.line(
            [tip, (tip[0] + head_len * math.cos(a), tip[1] + head_len * math.sin(a))],
            fill=color,
            width=6,
        )
    img.save(path)


def draw_fire(path: Path, color: str = "#FF6B35") -> None:
    """Tear-drop / flame shape — simplified fire emoji."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Outer flame: tear-drop bezier-ish polygon.
    pts = []
    for i in range(40):
        t = i / 39
        angle = t * 2 * math.pi
        # Stretched on the bottom, pointed on top.
        r_x = 28 + 6 * math.sin(angle * 2)
        r_y = 38 if angle > math.pi else 24
        x = SIZE / 2 + r_x * math.sin(angle)
        y = SIZE / 2 + r_y * math.cos(angle) * -1
        pts.append((x, y))
    d.polygon(pts, fill=color, outline="#B23A1A", width=3)
    # Inner core — yellow.
    inner_pts = [(SIZE / 2 + (px - SIZE / 2) * 0.55, SIZE / 2 + (py - SIZE / 2) * 0.55 + 6) for px, py in pts]
    d.polygon(inner_pts, fill="#FFE066")
    img.save(path)


def draw_exclaim(path: Path, color: str = "#E63946") -> None:
    """Comic-book impact burst — jagged star with !? in the middle space.
    We draw just the burst here; text is layered separately.
    """
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = SIZE // 2
    # 12-point star with high point variance for jagged look.
    n_pts = 12
    long_r, short_r = 56, 30
    pts = []
    for i in range(n_pts * 2):
        angle = i * math.pi / n_pts - math.pi / 2
        r = long_r if i % 2 == 0 else short_r
        # Add slight per-point jitter for hand-drawn feel.
        r_jit = r + (-3 if i % 4 == 0 else 2 if i % 3 == 0 else 0)
        pts.append((cx + r_jit * math.cos(angle), cy + r_jit * math.sin(angle)))
    d.polygon(pts, fill=color, outline="#FFFFFF", width=4)
    img.save(path)


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("sparkle.png", draw_sparkle, ["sparkle", "celebration", "shine"]),
        ("heart.png", draw_heart, ["heart", "love"]),
        ("star.png", draw_star5, ["star", "musical-note"]),
        ("dot.png", draw_dot, ["dot", "circle"]),
        ("mini_heart.png", draw_mini_heart, ["mini-heart", "confetti", "small-heart"]),
        ("burst.png", draw_burst, ["burst", "emphasis", "starburst", "pop"]),
        ("arrow.png", draw_arrow, ["arrow", "pointer", "look-here"]),
        ("fire.png", draw_fire, ["fire", "flame", "hot", "spicy"]),
        ("exclaim.png", draw_exclaim, ["exclaim", "impact", "bam", "shock"]),
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
