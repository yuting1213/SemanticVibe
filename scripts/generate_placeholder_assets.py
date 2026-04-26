"""Generate hand-drawn-style decoration PNGs into data/assets_lib/.

Aesthetic target (per user's baseline reference):
- Outline-only, no solid fills — internals stay transparent.
- Each contour point is jittered ±jitter px so the line wobbles like a
  shaky hand drew it.
- Two overlapping passes per shape with slightly different jitter, so
  it reads as "drawn over" rather than as a clean polygon.
- Stroke width varies a bit between segments (the human hand never
  produces uniform pressure).

The earlier geometric versions (filled hearts, polygon stars) shipped as
`*_geometric.png` if you want the cleaner look back.

Run from project root:
    uv run python scripts/generate_placeholder_assets.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

ASSETS_DIR = Path(__file__).resolve().parent.parent / "data" / "assets_lib"
SIZE = 128


# ---------------------------------------------------------------------------
# Hand-drawn primitives
# ---------------------------------------------------------------------------


def _wobble_point(p: tuple[float, float], rng: random.Random, jitter: float) -> tuple[float, float]:
    return (p[0] + rng.uniform(-jitter, jitter), p[1] + rng.uniform(-jitter, jitter))


def _draw_wobbly_path(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: str,
    *,
    base_width: int,
    rng: random.Random,
    jitter: float,
    closed: bool = True,
) -> None:
    """Draw a path point-by-point, jittering each endpoint and varying width.

    `points` is the IDEAL contour — we walk it once with one jitter seed,
    then a second time with a different seed so the eye reads it as two
    overlapping pencil strokes. closed=True draws a return segment to
    points[0].
    """
    iter_points = points + ([points[0]] if closed else [])
    for pass_idx in range(2):
        sub_rng = random.Random(rng.random())
        prev = _wobble_point(iter_points[0], sub_rng, jitter)
        for raw in iter_points[1:]:
            curr = _wobble_point(raw, sub_rng, jitter)
            # Segment width wobbles too — never wider than base_width + 1.
            seg_w = max(1, base_width + sub_rng.choice([-1, 0, 0, 1]))
            draw.line([prev, curr], fill=color, width=seg_w)
            prev = curr


def _heart_contour(scale: float, n_points: int = 60) -> list[tuple[float, float]]:
    """Classic heart parametric curve, centred at (0, 0)."""
    pts = []
    for i in range(n_points):
        t = i / (n_points - 1) * 2 * math.pi
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((x * scale, y * scale))
    return pts


def draw_handdrawn_heart(
    path: Path,
    color: str = "#FF6B9D",
    *,
    size: int = SIZE,
    scale: float = 3.2,
    stroke_width: int = 4,
    jitter: float = 3.0,
    seed: int = 0,
) -> None:
    """Outline-only heart with wobbling double-stroked contour."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rng = random.Random(seed)

    contour = [
        (size / 2 + x, size / 2 + y - 4)
        for x, y in _heart_contour(scale)
    ]
    _draw_wobbly_path(d, contour, color, base_width=stroke_width, rng=rng, jitter=jitter)
    img.save(path)


def draw_handdrawn_mini_heart(path: Path, color: str = "#FF6B9D") -> None:
    """Smaller scale of the hand-drawn heart for confetti use."""
    draw_handdrawn_heart(path, color, scale=2.1, stroke_width=3, jitter=2.0, seed=1)


def draw_handdrawn_sparkle(
    path: Path, color: str = "#FF6B9D", *, jitter: float = 2.5
) -> None:
    """Four-point burst as wobbly outlined diamonds."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rng = random.Random(2)
    cx = cy = SIZE // 2
    long_r, short_r = 52, 14
    contour = []
    for i in range(8):
        angle = i * math.pi / 4 - math.pi / 2
        r = long_r if i % 2 == 0 else short_r
        contour.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    _draw_wobbly_path(d, contour, color, base_width=4, rng=rng, jitter=jitter)
    img.save(path)


def draw_handdrawn_star(
    path: Path, color: str = "#E63946", *, jitter: float = 2.5
) -> None:
    """5-point star, outline-only with wobble."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rng = random.Random(3)
    cx = cy = SIZE // 2
    long_r, short_r = 50, 22
    contour = []
    for i in range(10):
        angle = i * math.pi / 5 - math.pi / 2
        r = long_r if i % 2 == 0 else short_r
        contour.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    _draw_wobbly_path(d, contour, color, base_width=4, rng=rng, jitter=jitter)
    img.save(path)


def draw_handdrawn_dot(path: Path, color: str = "#FF6B9D") -> None:
    """Small wobbly circle outline."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rng = random.Random(4)
    cx = cy = SIZE // 2
    r = 36
    contour = [(cx + r * math.cos(t * 2 * math.pi / 40), cy + r * math.sin(t * 2 * math.pi / 40))
               for t in range(40)]
    _draw_wobbly_path(d, contour, color, base_width=4, rng=rng, jitter=2.0)
    img.save(path)


def draw_handdrawn_burst(
    path: Path, color: str = "#FF6B9D", *, jitter: float = 2.0
) -> None:
    """Radiating wobbly lines from a centre point — hand-drawn manga emphasis."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rng = random.Random(5)
    cx = cy = SIZE // 2
    n_lines = 12
    for i in range(n_lines):
        angle = (i / n_lines) * 2 * math.pi
        inner_r = 12 + rng.uniform(-2, 2)
        outer_r = 54 + rng.uniform(-6, 6)
        x1 = cx + inner_r * math.cos(angle) + rng.uniform(-jitter, jitter)
        y1 = cy + inner_r * math.sin(angle) + rng.uniform(-jitter, jitter)
        x2 = cx + outer_r * math.cos(angle) + rng.uniform(-jitter, jitter)
        y2 = cy + outer_r * math.sin(angle) + rng.uniform(-jitter, jitter)
        # Two overlapping passes per ray.
        for _ in range(2):
            jx = rng.uniform(-1.5, 1.5)
            jy = rng.uniform(-1.5, 1.5)
            d.line([(x1 + jx, y1 + jy), (x2 + jx, y2 + jy)], fill=color, width=4)
    img.save(path)


def draw_handdrawn_arrow(
    path: Path, color: str = "#FF6B9D", *, jitter: float = 2.0
) -> None:
    """Wobbly hand-drawn arrow pointing down-right."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rng = random.Random(6)
    pts = []
    for i in range(20):
        t = i / 19
        x = 18 + 80 * t + math.sin(t * math.pi * 4) * 3
        y = 18 + 80 * t + math.cos(t * math.pi * 3) * 4
        pts.append((x, y))
    for pass_idx in range(2):
        sub_rng = random.Random(rng.random())
        prev = _wobble_point(pts[0], sub_rng, jitter)
        for raw in pts[1:]:
            curr = _wobble_point(raw, sub_rng, jitter)
            d.line([prev, curr], fill=color, width=max(3, 5 + sub_rng.choice([-1, 0, 0, 1])))
            prev = curr
    # Arrowhead: two short strokes off the tip.
    tip = pts[-1]
    head_len = 22
    for angle_off in (math.radians(150), math.radians(-150 - 40)):
        a = math.atan2(pts[-1][1] - pts[-3][1], pts[-1][0] - pts[-3][0]) + angle_off
        d.line(
            [tip, (tip[0] + head_len * math.cos(a), tip[1] + head_len * math.sin(a))],
            fill=color,
            width=5,
        )
    img.save(path)


def draw_handdrawn_fire(path: Path, color: str = "#E63946") -> None:
    """Small hand-drawn flame outline."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rng = random.Random(7)
    contour = []
    for i in range(40):
        t = i / 39
        angle = t * 2 * math.pi
        r_x = 28 + 6 * math.sin(angle * 2)
        r_y = 38 if angle > math.pi else 24
        x = SIZE / 2 + r_x * math.sin(angle)
        y = SIZE / 2 + r_y * math.cos(angle) * -1
        contour.append((x, y))
    _draw_wobbly_path(d, contour, color, base_width=4, rng=rng, jitter=2.0)
    img.save(path)


def draw_handdrawn_exclaim(path: Path, color: str = "#E63946") -> None:
    """Jagged star outline (impact burst)."""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rng = random.Random(8)
    cx = cy = SIZE // 2
    n_pts = 12
    long_r, short_r = 54, 28
    contour = []
    for i in range(n_pts * 2):
        angle = i * math.pi / n_pts - math.pi / 2
        r = long_r if i % 2 == 0 else short_r
        r_jit = r + rng.uniform(-3, 3)
        contour.append((cx + r_jit * math.cos(angle), cy + r_jit * math.sin(angle)))
    _draw_wobbly_path(d, contour, color, base_width=4, rng=rng, jitter=2.5)
    img.save(path)


# ---------------------------------------------------------------------------


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Callable[[Path], None], list[str]]] = [
        ("heart.png",       draw_handdrawn_heart,       ["heart", "love"]),
        ("mini_heart.png",  draw_handdrawn_mini_heart,  ["mini-heart", "confetti", "small-heart"]),
        ("sparkle.png",     draw_handdrawn_sparkle,     ["sparkle", "celebration", "shine"]),
        ("star.png",        draw_handdrawn_star,        ["star", "musical-note"]),
        ("dot.png",         draw_handdrawn_dot,         ["dot", "circle"]),
        ("burst.png",       draw_handdrawn_burst,       ["burst", "emphasis", "starburst", "pop"]),
        ("arrow.png",       draw_handdrawn_arrow,       ["arrow", "pointer", "look-here"]),
        ("fire.png",        draw_handdrawn_fire,        ["fire", "flame", "hot", "spicy"]),
        ("exclaim.png",     draw_handdrawn_exclaim,     ["exclaim", "impact", "bam", "shock"]),
    ]
    metadata = []
    for filename, draw_fn, tags in jobs:
        p = ASSETS_DIR / filename
        draw_fn(p)
        metadata.append(
            {
                "filename": filename,
                "tags": tags,
                "license": "self-drawn (procedural, hand-drawn jitter)",
                "source": "internal",
            }
        )
        print(f"  {filename}  ({p.stat().st_size:,} bytes)")

    meta_path = ASSETS_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {meta_path} with {len(metadata)} entries")


if __name__ == "__main__":
    main()
