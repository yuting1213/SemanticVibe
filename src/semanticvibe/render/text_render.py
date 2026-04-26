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


def _total_outline_width(element: TextElement) -> int:
    """Sum of the primary outline + every additional outline_layers width."""
    return element.outline_width + sum(layer.width for layer in element.outline_layers)


def measure_text(element: TextElement, fonts_dir: Path) -> tuple[int, int]:
    """Return (width, height) of the rendered string in pixels, including all outlines."""
    font_path = str(_resolve_font_file(element.font, fonts_dir))
    font = _load_font(font_path, element.size)
    bbox = font.getbbox(
        element.content,
        stroke_width=_total_outline_width(element),
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
    final canvas.

    Outline rendering: starts from the OUTERMOST layer (the last entry of
    `outline_layers`, drawn first so it sits behind everything) and works
    inward to the primary `outline_color`/`outline_width`, then the fill on
    top. Pillow's `stroke_width` draws a stroke OUTSIDE the glyph at the
    given width, so each pass uses the cumulative width up to that layer.

    A drop-shadow layer (if `shadow_offset` is set) is drawn first of all,
    underneath the outline stack.
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

    # Sizing: use the *full* string's bbox at the largest stroke width so
    # the tile doesn't jitter as the typewriter advances and accommodates
    # the outermost outline layer plus any shadow offset.
    total_outline = _total_outline_width(element)
    full_bbox = font.getbbox(element.content, stroke_width=total_outline)
    pad = total_outline + 4
    sx, sy = element.shadow_offset or (0, 0)
    extra_x = abs(sx)
    extra_y = abs(sy)
    width = (full_bbox[2] - full_bbox[0]) + 2 * pad + extra_x
    height = (full_bbox[3] - full_bbox[1]) + 2 * pad + extra_y

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Anchor inside the tile: leave room for outermost outline + shadow drift.
    base_xy = (
        pad - full_bbox[0] + (sx if sx > 0 else 0),
        pad - full_bbox[1] + (sy if sy > 0 else 0),
    )

    # Drop shadow (drawn first → bottom of stack).
    if element.shadow_offset is not None:
        draw.text(
            (base_xy[0] + sx, base_xy[1] + sy),
            visible,
            font=font,
            fill="#00000080",
            stroke_width=total_outline,
            stroke_fill="#00000080",
        )

    # Outline layers, OUTERMOST first (largest stroke), working inward.
    # Each pass renders the glyph WITH the cumulative stroke width using
    # that layer's colour as both fill and stroke. Subsequent passes draw
    # a smaller stroke on top, so the rings appear concentric.
    cumulative = total_outline
    for layer in element.outline_layers:
        draw.text(
            base_xy,
            visible,
            font=font,
            fill=layer.color,
            stroke_width=cumulative,
            stroke_fill=layer.color,
        )
        cumulative -= layer.width

    # Primary outline (the innermost outline before the fill).
    if element.outline_width > 0:
        draw.text(
            base_xy,
            visible,
            font=font,
            fill=element.outline_color,
            stroke_width=element.outline_width,
            stroke_fill=element.outline_color,
        )

    # Fill on top.
    draw.text(
        base_xy,
        visible,
        font=font,
        fill=element.color,
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
