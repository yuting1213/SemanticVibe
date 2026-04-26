"""CJK-safe text rendering via Pillow.

Spec §5.5.1 mandates Pillow over MoviePy.TextClip — TextClip's ImageMagick
path mangles CJK glyphs and breaks on Windows. Pillow's `truetype` + `text`
with `stroke_width` handles double-outline rendering correctly.

Convention: every render returns RGBA. Transparent pixels stay transparent
when composited. Anchor coordinates are top-left of the rendered string.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from semanticvibe.render.animations import AnimationState
from semanticvibe.schemas.decision import TextElement


def _resolve_font_file(font_name: str, fonts_dir: Path) -> Path:
    """Map a font *name* to an actual file under `fonts_dir`.

    Accepts: bare name ("KleeOne-Regular"), filename ("KleeOne-Regular.ttf"),
    or absolute path. Falls back to NotoSansTC-Regular.ttf for CJK coverage.
    """
    candidate = Path(font_name)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    for suffix in ("", ".ttf", ".otf"):
        p = fonts_dir / f"{font_name}{suffix}"
        if p.exists():
            return p

    fallback = fonts_dir / "NotoSansTC-Regular.ttf"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"font {font_name!r} not found under {fonts_dir} and no NotoSansTC-Regular fallback"
    )


@lru_cache(maxsize=64)
def _load_font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size=size)


def measure_text(element: TextElement, fonts_dir: Path) -> tuple[int, int]:
    """Return (width, height) of the rendered string in pixels, including outline."""
    font_path = str(_resolve_font_file(element.font, fonts_dir))
    font = _load_font(font_path, element.size)
    bbox = font.getbbox(
        element.content,
        stroke_width=element.outline_width,
    )
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def fit_to_canvas(
    element: TextElement,
    fonts_dir: Path,
    canvas_size: tuple[int, int],
    *,
    margin: int = 16,
    min_size: int = 24,
) -> TextElement:
    """Return a copy of `element` with `size` shrunk so the rendered tile
    fits inside `canvas_size` minus `margin` on each side.

    Portrait phone clips downscaled to a 404-wide preview canvas can't fit
    7+ Chinese characters at the LLM-suggested size of 72-96 px. Rather
    than clip the text or let it bleed off the edge, we proportionally
    shrink. `min_size` floors the shrink so we don't end up with unreadable
    text on degenerate inputs.

    The shrink is iterative because outline width doesn't scale with font
    size and glyph spacing isn't perfectly linear, so a single proportional
    estimate can land short.
    """
    canvas_w, canvas_h = canvas_size
    max_w = max(min_size, canvas_w - 2 * margin)
    max_h = max(min_size, canvas_h - 2 * margin)

    text_w, text_h = measure_text(element, fonts_dir)
    if text_w <= max_w and text_h <= max_h:
        return element

    current = element
    for _ in range(8):  # bounded so a degenerate input can't loop forever
        text_w, text_h = measure_text(current, fonts_dir)
        if (text_w <= max_w and text_h <= max_h) or current.size <= min_size:
            break
        # Shrink by the worse-fitting axis, with a small extra factor (0.95)
        # so we converge instead of plateauing one pixel above the bound.
        scale = min(max_w / text_w, max_h / text_h) * 0.95
        new_size = max(min_size, int(current.size * scale))
        if new_size == current.size:
            new_size = max(min_size, current.size - 1)
        current = current.model_copy(update={"size": new_size})

    return current if current.size != element.size else element


def render_text(
    element: TextElement,
    state: AnimationState,
    fonts_dir: Path,
) -> Image.Image:
    """Render a single TextElement frame as an RGBA Pillow image.

    Returns the smallest tile that contains the glyphs; the caller is
    responsible for placing it at the element's anchor + state.dx/dy on the
    final canvas. Outline is drawn once via Pillow's `stroke_width`.
    """
    if state.alpha <= 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    font_path = str(_resolve_font_file(element.font, fonts_dir))
    font = _load_font(font_path, element.size)

    # Honour typewriter / draw_in reveal: trim the displayed string.
    visible = element.content
    if state.reveal_fraction < 1.0:
        n = max(1, int(round(len(element.content) * state.reveal_fraction)))
        visible = element.content[:n]

    # Sizing: use the *full* string's bbox so the tile size doesn't jitter as
    # the typewriter advances.
    full_bbox = font.getbbox(element.content, stroke_width=element.outline_width)
    pad = element.outline_width + 2
    width = (full_bbox[2] - full_bbox[0]) + 2 * pad
    height = (full_bbox[3] - full_bbox[1]) + 2 * pad

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.text(
        (pad - full_bbox[0], pad - full_bbox[1]),
        visible,
        font=font,
        fill=element.color,
        stroke_width=element.outline_width,
        stroke_fill=element.outline_color,
    )

    if state.alpha < 1.0:
        # Multiply the alpha channel by `state.alpha`.
        alpha_channel = img.getchannel("A")
        alpha_channel = alpha_channel.point(lambda px: int(px * state.alpha))
        img.putalpha(alpha_channel)

    if state.scale != 1.0:
        new_w = max(1, int(round(width * state.scale)))
        new_h = max(1, int(round(height * state.scale)))
        img = img.resize((new_w, new_h), Image.LANCZOS)

    if state.rotation_deg:
        img = img.rotate(state.rotation_deg, resample=Image.BICUBIC, expand=True)

    return img
