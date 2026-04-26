"""Render-stage tests.

The full render path needs a real video, real fonts, and ffmpeg — those are
covered by integration tests behind the `integration` marker. Here we cover
what we can without those: text-tile rendering against a synthetic font.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PIL import Image, ImageFont

from semanticvibe.render.animations import AnimationState
from semanticvibe.render.text_render import (
    _resolve_font_file,
    fit_to_canvas,
    measure_text,
    render_text,
)
from semanticvibe.schemas.decision import TextElement


def _find_system_truetype_font() -> Path | None:
    """Look for any TrueType font we can use as a stand-in for tests.

    Tests don't ship the project's own fonts (those are in `data/fonts/`,
    which is gitignored). On Windows we can usually pick up Arial.
    """
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/Library/Fonts/Arial.ttf"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: ship a tiny font via Pillow's bundled default? Pillow's load_default
    # doesn't return a TrueType font, so it doesn't satisfy stroke_width — skip.
    return None


@pytest.fixture
def fonts_dir(tmp_path: Path) -> Path:
    """Build a per-test fonts dir with one TrueType font copied in."""
    src = _find_system_truetype_font()
    if src is None:
        pytest.skip("no system TrueType font available for render tests")
    dest_dir = tmp_path / "fonts"
    dest_dir.mkdir()
    # Copy under a stable name so the TextElement.font field can reference it.
    shutil.copy(src, dest_dir / "TestFont.ttf")
    # Also copy as the CJK fallback name so the resolver can find it that way.
    shutil.copy(src, dest_dir / "NotoSansTC-Regular.ttf")
    return dest_dir


def _make_text_element(font_name: str = "TestFont", content: str = "Hi") -> TextElement:
    return TextElement(
        content=content,
        start_time=0.0,
        end_time=2.0,
        font=font_name,
        size=48,
        color="#FFFFFF",
        outline_color="#000000",
        outline_width=3,
        animation="fade",
        reasoning="test",
    )


def test_resolve_font_file_finds_bare_name(fonts_dir: Path):
    p = _resolve_font_file("TestFont", fonts_dir)
    assert p.name == "TestFont.ttf"


def test_resolve_font_file_falls_back_to_noto(fonts_dir: Path):
    # Asking for a name that doesn't exist should hit the NotoSansTC fallback.
    p = _resolve_font_file("DoesNotExist", fonts_dir)
    assert p.name == "NotoSansTC-Regular.ttf"


def test_resolve_font_file_raises_when_no_fallback(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        _resolve_font_file("Whatever", empty)


def test_render_text_returns_rgba_with_content(fonts_dir: Path):
    el = _make_text_element()
    state = AnimationState(alpha=1.0)
    img = render_text(el, state, fonts_dir)
    assert img.mode == "RGBA"
    assert img.width > 1 and img.height > 1
    # Some pixel must be opaque — the glyph itself.
    assert img.getextrema()[3][1] > 0


def test_render_text_invisible_returns_tiny_transparent_tile(fonts_dir: Path):
    el = _make_text_element()
    img = render_text(el, AnimationState(alpha=0.0), fonts_dir)
    assert img.size == (1, 1)
    assert img.getpixel((0, 0))[3] == 0


def test_render_text_partial_alpha_attenuates(fonts_dir: Path):
    el = _make_text_element()
    full = render_text(el, AnimationState(alpha=1.0), fonts_dir)
    half = render_text(el, AnimationState(alpha=0.5), fonts_dir)
    # Same tile dimensions, but max alpha should be roughly halved.
    assert full.size == half.size
    full_max_a = full.getextrema()[3][1]
    half_max_a = half.getextrema()[3][1]
    assert half_max_a < full_max_a


def test_render_text_typewriter_reveal_shorter_than_full(fonts_dir: Path):
    el = _make_text_element(content="ABCDEFGH")
    full = render_text(el, AnimationState(alpha=1.0, reveal_fraction=1.0), fonts_dir)
    half = render_text(el, AnimationState(alpha=1.0, reveal_fraction=0.5), fonts_dir)
    # Tile dimensions are stable (full string bbox), but the half-revealed
    # tile must have FEWER opaque pixels.
    full_opaque = sum(1 for px in full.getdata() if px[3] > 0)
    half_opaque = sum(1 for px in half.getdata() if px[3] > 0)
    assert half_opaque < full_opaque


def test_render_text_scale_grows_image(fonts_dir: Path):
    el = _make_text_element()
    base = render_text(el, AnimationState(alpha=1.0, scale=1.0), fonts_dir)
    bigger = render_text(el, AnimationState(alpha=1.0, scale=1.5), fonts_dir)
    assert bigger.width > base.width
    assert bigger.height > base.height


def test_fit_to_canvas_shrinks_oversized_text(fonts_dir: Path):
    """A string that overflows the canvas at the requested size must shrink."""
    el = _make_text_element(content="ABCDEFGHIJKLMNOP")
    big = el.model_copy(update={"size": 96})
    fitted = fit_to_canvas(big, fonts_dir, canvas_size=(200, 720), margin=8)
    assert fitted.size < big.size


def test_fit_to_canvas_actually_fits_when_possible(fonts_dir: Path):
    """When the shrunk-down size still fits above the min_size floor, the
    rendered tile must end up inside the canvas minus margins."""
    el = _make_text_element(content="ABCDEF")
    big = el.model_copy(update={"size": 96})
    canvas = (300, 720)
    margin = 16
    fitted = fit_to_canvas(big, fonts_dir, canvas_size=canvas, margin=margin)
    w, _h = measure_text(fitted, fonts_dir)
    assert w <= canvas[0] - 2 * margin


def test_fit_to_canvas_passthrough_when_fits(fonts_dir: Path):
    el = _make_text_element(content="Hi")
    fitted = fit_to_canvas(el, fonts_dir, canvas_size=(1280, 720))
    # Same instance — element already fit, no copy needed.
    assert fitted is el


def test_fit_to_canvas_floors_at_min_size(fonts_dir: Path):
    el = _make_text_element(content="ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    big = el.model_copy(update={"size": 200})
    # Tiny canvas + min_size floor — must clamp at min_size, not go below.
    fitted = fit_to_canvas(big, fonts_dir, canvas_size=(80, 80), min_size=24)
    assert fitted.size == 24
