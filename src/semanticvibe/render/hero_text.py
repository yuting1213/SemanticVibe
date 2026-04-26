"""Render `HeroTextElement` — huge chalk/crayon-style centred glyph.

Distinct from the playful TextElement renderer because the look is
intentionally calmer:

- Multi-pass blur halos around the glyph build a soft "powder around chalk"
  edge instead of a sharp manga-style outline.
- The fill itself is overlaid with small white dots / short strokes
  (`grain=True`) to imply chalk dust on the page.
- Animation is a slow fade-in/out with optional gentle scale breathing
  (sin-modulated ±2% scale around 1.0).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from semanticvibe.render.text_render import _load_font, _resolve_font_file
from semanticvibe.schemas.decision import HeroPosition, HeroTextElement


def hero_breathing_scale(now: float, start: float) -> float:
    """0.97 .. 1.03 sinusoidal scale modulation, periodic on ~3.5s."""
    phase = (now - start) * (2 * math.pi / 3.5)
    return 1.0 + 0.03 * math.sin(phase)


def _hero_alpha(now: float, start: float, end: float, fade_in: float = 1.2,
                fade_out: float = 1.0) -> float:
    if now < start or now > end:
        return 0.0
    if now - start < fade_in:
        return (now - start) / fade_in
    if end - now < fade_out:
        return max(0.0, (end - now) / fade_out)
    return 1.0


def _resolve_hero_position(
    pos: HeroPosition | tuple[int, int],
    tile_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> tuple[int, int]:
    canvas_w, canvas_h = canvas_size
    tile_w, tile_h = tile_size
    if isinstance(pos, tuple):
        return (pos[0] - tile_w // 2, pos[1] - tile_h // 2)
    cx = canvas_w // 2
    if pos == "center":
        return (cx - tile_w // 2, canvas_h // 2 - tile_h // 2)
    if pos == "center_lower":
        return (cx - tile_w // 2, int(canvas_h * 0.66) - tile_h // 2)
    if pos == "upper_left":
        return (int(canvas_w * 0.18), int(canvas_h * 0.18))
    if pos == "upper_right":
        return (int(canvas_w * 0.82) - tile_w, int(canvas_h * 0.18))
    # default: center_upper
    return (cx - tile_w // 2, int(canvas_h * 0.28) - tile_h // 2)


def render_hero(
    element: HeroTextElement,
    *,
    now: float,
    fonts_dir: Path,
    canvas_size: tuple[int, int],
) -> tuple[Image.Image, tuple[int, int]] | None:
    """Render the hero glyph at time `now`. Returns (RGBA tile, top-left
    pixel) or None when fully invisible at this timestamp.
    """
    alpha = _hero_alpha(now, element.start_time, element.end_time)
    if alpha <= 0:
        return None

    font_path = str(_resolve_font_file(element.font, fonts_dir))
    font = _load_font(font_path, element.size)

    bbox = font.getbbox(element.content)
    pad = max(40, element.size // 4)  # roomy pad — blur halos need bleed space
    width = (bbox[2] - bbox[0]) + 2 * pad
    height = (bbox[3] - bbox[1]) + 2 * pad

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    glyph_xy = (pad - bbox[0], pad - bbox[1])

    if element.style == "chalk":
        # Build the halo on a separate layer so we can blur and tint it
        # independently of the crisp fill on top.
        halo = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        halo_draw = ImageDraw.Draw(halo)
        halo_color = ImageColor.getrgb(element.halo_color)
        # Three blur passes from soft-and-wide to tight-and-bright.
        for blur_radius, halo_alpha in [(28, 70), (16, 100), (8, 140)]:
            layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            ImageDraw.Draw(layer).text(
                glyph_xy, element.content, font=font,
                fill=halo_color + (halo_alpha,),
            )
            layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            halo = Image.alpha_composite(halo, layer)
        img = Image.alpha_composite(img, halo)
        draw = ImageDraw.Draw(img)

    # Crisp fill on top.
    draw.text(glyph_xy, element.content, font=font, fill=element.color)

    if element.style == "chalk" and element.grain:
        # Chalk dust: ~80 small white dots and short strokes inside the glyph
        # bounding box. Deterministic per-element so frames are stable.
        rng = random.Random(hash(element.content) & 0xFFFFFFFF)
        gx0, gy0 = glyph_xy
        gx1 = gx0 + (bbox[2] - bbox[0])
        gy1 = gy0 + (bbox[3] - bbox[1])
        for _ in range(80):
            x = rng.randint(gx0, gx1)
            y = rng.randint(gy0, gy1)
            kind = rng.random()
            if kind < 0.6:
                # tiny dot
                r = rng.choice([1, 1, 2])
                draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 200))
            else:
                # short stroke
                dx = rng.choice([-3, -2, 2, 3])
                draw.line([(x, y), (x + dx, y + dx)], fill=(255, 255, 255, 180), width=1)

    if alpha < 1.0:
        a = img.getchannel("A")
        a = a.point(lambda px: int(px * alpha))
        img.putalpha(a)

    if element.breathing:
        scale = hero_breathing_scale(now, element.start_time)
        if scale != 1.0:
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            img = img.resize((new_w, new_h), Image.LANCZOS)

    top_left = _resolve_hero_position(element.pos, img.size, canvas_size)
    return img, top_left
