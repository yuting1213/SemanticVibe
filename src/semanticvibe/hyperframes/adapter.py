"""Decision → Hyperframes composition (HTML + GSAP timeline).

Translates the v10 Decision schema into a self-contained HTML document
that can be screen-captured frame-by-frame. The page is transparent
(background: transparent) — `omitBackground: true` in Puppeteer
preserves the alpha channel, so the rendered video composites cleanly
over the original dance video downstream.

Animation translation: each element's `animation` (entry) + `idle_animation`
fields map to GSAP tween calls. The mapping table follows the spec's
"animation correspondence" table, including the v9 beat-sync
`beat_period_sec` for tempo-locked pulse.

Returns the path to the generated composition.html, plus the duration
(seconds) so the caller knows how many frames to capture.
"""

from __future__ import annotations

import html
import json
import logging
import shutil
import textwrap
from dataclasses import dataclass
from pathlib import Path

from semanticvibe.schemas.decision import (
    Decision,
    DecorationElement,
    HeroTextElement,
    SubtitleBannerElement,
    SubtitleOutlinedElement,
    TextElement,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompositionResult:
    html_path: Path
    duration_sec: float
    canvas_size: tuple[int, int]
    assets_dir: Path
    fps: int


# ---------------------------------------------------------------------------
# Entry-animation → GSAP tween mapping
# ---------------------------------------------------------------------------
#
# Each entry returns a `tl.from(selector, {...}, start_time)` call. We use
# `from()` so the element appears at its final position at end-of-entry
# without manual reset.

_ENTRY_DUR = 0.5  # default entry-animation length in seconds


def _gate_visibility(selector: str, start_time: float, end_time: float) -> list[str]:
    """Produce the GSAP lines that hide an element before its start_time
    and after its end_time. Without these, every element is visible the
    entire composition duration (data-start/data-duration are *metadata*,
    not actual visibility gates).

    `autoAlpha` ties opacity + visibility together so the element doesn't
    consume pixels when 0.
    """
    s = json.dumps(selector)
    t0 = round(start_time, 3)
    t1 = round(end_time, 3)
    return [
        # Start hidden at t=0.
        f'tl.set({s}, {{ autoAlpha: 0 }}, 0);',
        # Show at start_time. The entry animation that fires here will
        # tween FROM its `opacity: 0` back to this revealed state.
        f'tl.set({s}, {{ autoAlpha: 1 }}, {t0});',
        # Hide again at end_time.
        f'tl.set({s}, {{ autoAlpha: 0 }}, {t1});',
    ]


def _entry_gsap(animation: str, selector: str, start_time: float) -> str:
    """Render one `tl.from(...)` JS line for the given entry animation."""
    s = json.dumps(selector)
    t = round(start_time, 3)
    if animation == "fade":
        return f'tl.from({s}, {{ opacity: 0, duration: {_ENTRY_DUR}, ease: "power2.out" }}, {t});'
    if animation == "bounce_in":
        return f'tl.from({s}, {{ scale: 0.5, opacity: 0, duration: 0.6, ease: "bounce.out" }}, {t});'
    if animation == "typewriter":
        # GSAP TextPlugin is on the CDN bundle — but text reveal via clip-path is simpler + free.
        return (
            f'tl.from({s}, {{ clipPath: "inset(0 100% 0 0)", duration: 0.7, ease: "none" }}, {t});'
        )
    if animation == "draw_in":
        return (
            f'tl.from({s}, {{ clipPath: "inset(0 0 100% 0)", duration: 0.6, ease: "power1.inOut" }}, {t});'
        )
    if animation == "wiggle":
        return (
            f'tl.from({s}, {{ rotation: -10, opacity: 0, duration: 0.5, ease: "elastic.out(1, 0.55)" }}, {t});'
        )
    if animation == "scale_pop":
        return f'tl.from({s}, {{ scale: 0.3, opacity: 0, duration: 0.5, ease: "back.out(2.2)" }}, {t});'
    if animation == "drop_in":
        return f'tl.from({s}, {{ y: -200, opacity: 0, duration: 0.7, ease: "bounce.out" }}, {t});'
    if animation == "slide_in_left":
        return f'tl.from({s}, {{ x: -300, opacity: 0, duration: 0.5, ease: "power3.out" }}, {t});'
    if animation == "slide_in_right":
        return f'tl.from({s}, {{ x: 300, opacity: 0, duration: 0.5, ease: "power3.out" }}, {t});'
    if animation == "slide_in_top":
        return f'tl.from({s}, {{ y: -300, opacity: 0, duration: 0.5, ease: "power3.out" }}, {t});'
    if animation == "slide_in_bottom":
        return f'tl.from({s}, {{ y: 300, opacity: 0, duration: 0.5, ease: "power3.out" }}, {t});'
    if animation == "stamp":
        # elastic settle from 1.6x with a tiny rotation shake.
        return (
            f'tl.from({s}, {{ scale: 1.6, opacity: 0, duration: 0.35, ease: "elastic.out(1, 0.5)" }}, {t});\n'
            f'tl.to({s}, {{ rotation: 5, duration: 0.08, repeat: 3, yoyo: true, ease: "power1.inOut" }}, {t});'
        )
    if animation == "wobble_in":
        return (
            f'tl.from({s}, {{ rotation: -15, scale: 0.7, opacity: 0, duration: 0.6, ease: "elastic.out(1, 0.6)" }}, {t});'
        )
    if animation == "spin_in":
        return (
            f'tl.from({s}, {{ rotation: 360, scale: 0, opacity: 0, duration: 0.6, ease: "back.out(2)" }}, {t});'
        )
    # Unknown → fall back to fade.
    log.warning("hyperframes adapter: unknown entry animation %r, using fade", animation)
    return f'tl.from({s}, {{ opacity: 0, duration: {_ENTRY_DUR}, ease: "power2.out" }}, {t});'


# ---------------------------------------------------------------------------
# Idle-animation → GSAP tween mapping (continuous, runs after entry settles)
# ---------------------------------------------------------------------------


def _idle_gsap(
    idle: str,
    selector: str,
    start_time: float,
    seed: int = 0,
    beat_period_sec: float | None = None,
) -> list[str]:
    """Return zero or more `tl.to(...)` lines implementing the idle loop.

    Idle starts after the entry settles (start_time + 0.5s by convention)
    and runs `repeat: -1` for the rest of the element's lifetime.
    """
    if idle in (None, "none", ""):
        return []
    s = json.dumps(selector)
    idle_start = round(start_time + _ENTRY_DUR, 3)
    if idle == "pulse":
        period = beat_period_sec or 1.5
        half = round(period / 2, 3)
        return [
            f'tl.to({s}, {{ scale: 1.08, duration: {half}, repeat: -1, yoyo: true, ease: "sine.inOut" }}, {idle_start});'
        ]
    if idle == "wiggle":
        return [
            f'tl.to({s}, {{ rotation: 4, x: "+=6", duration: 0.25, repeat: -1, yoyo: true, ease: "sine.inOut" }}, {idle_start});'
        ]
    if idle == "drift":
        # Slow figure-eight via two-axis offsets with different periods.
        return [
            f'tl.to({s}, {{ x: "+=15", duration: 1.6, repeat: -1, yoyo: true, ease: "sine.inOut" }}, {idle_start});',
            f'tl.to({s}, {{ y: "-=8", duration: 1.1, repeat: -1, yoyo: true, ease: "sine.inOut" }}, {idle_start});',
        ]
    if idle == "rotate_slow":
        return [
            f'tl.to({s}, {{ rotation: "+=360", duration: 8, repeat: -1, ease: "none" }}, {idle_start});'
        ]
    if idle == "shimmer":
        return [
            f'tl.to({s}, {{ opacity: 0.6, duration: 0.4, repeat: -1, yoyo: true, ease: "sine.inOut" }}, {idle_start});'
        ]
    log.warning("hyperframes adapter: unknown idle animation %r — skipped", idle)
    return []


# ---------------------------------------------------------------------------
# Element-by-element HTML/CSS generation
# ---------------------------------------------------------------------------


def _color_alpha(hex_color: str, alpha: int) -> str:
    """`'#RRGGBB' + 0..255` → 'rgba(r, g, b, a)'."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha / 255:.3f})"


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _wrap_cjk_or_latin(text: str, max_chars_per_line: int = 8) -> list[str]:
    """Greedy wrap mirroring the v10 Pillow wrap. CJK breaks anywhere
    (no spaces); Latin breaks at the last space within the prefix.

    `max_chars_per_line` is the practical visual budget for a narrow
    portrait canvas. The CSS-side max-width also applies; this just
    pre-splits for SVG which doesn't auto-wrap.
    """
    if len(text) <= max_chars_per_line:
        return [text]
    lines: list[str] = []
    remaining = text
    while remaining and len(lines) < 3:
        if len(remaining) <= max_chars_per_line:
            lines.append(remaining)
            break
        cut = max_chars_per_line
        # Latin word-break preference: walk back to last space within prefix.
        if cut < len(remaining) and remaining[cut].isascii() and remaining[cut - 1].isascii():
            space_idx = remaining.rfind(" ", 0, cut)
            if space_idx > 0:
                cut = space_idx + 1
        lines.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    return lines


def _render_subtitle_outlined(
    el: SubtitleOutlinedElement, idx: int, canvas_size: tuple[int, int],
) -> tuple[str, str]:
    """Return (html_markup, css_block) for a SubtitleOutlinedElement.

    v11 Phase 2: render via inline SVG `<text>` with `paint-order:
    stroke fill` for crisp anti-aliased outlines on CJK glyphs.
    `-webkit-text-stroke` was aliased + double-stroked at line breaks.
    """
    cw, ch = canvas_size
    # Vertical anchor by position keyword.
    if el.position == "top_banner":
        y_css = f"top: {el.margin}px;"
    elif el.position == "bottom_banner":
        y_css = f"bottom: {el.margin}px;"
    else:
        y_css = "top: 50%;"

    # Conservative wrap budget. We can't trust upstream el.size because
    # v10's Pillow fitter uses Noto Sans TC metrics whereas Chrome falls
    # back to Yu Gothic on Windows — glyph widths differ enough to push
    # the rendered text off-canvas. Recompute size + wrap here against
    # the actual canvas width.
    #
    # Strategy: pick the largest font size such that
    #   max(1, ceil(N / max_lines)) × size_per_char ≤ usable_width
    # where size_per_char ≈ size × 0.95 for CJK / 0.55 for Latin.
    is_cjk_heavy = any("　" <= ch <= "鿿" for ch in el.content)
    glyph_ratio = 0.95 if is_cjk_heavy else 0.55
    usable_w = cw * el.max_width_ratio - el.outline_width * 2 - 8
    # Try each size from preset down to 24; pick the smallest size that
    # allows the text to wrap inside `max_lines` AND fit `usable_w`.
    size = max(24, int(el.size))
    max_lines_budget = el.max_lines
    chosen_size = size
    chosen_lines: list[str] = [el.content]
    while True:
        chars_per_line = max(1, int(usable_w / (size * glyph_ratio)))
        wrapped = _wrap_cjk_or_latin(el.content, max_chars_per_line=chars_per_line)
        wrapped = wrapped[:max_lines_budget]
        widest = max(len(line) for line in wrapped)
        if widest * size * glyph_ratio <= usable_w:
            chosen_size, chosen_lines = size, wrapped
            break
        if size <= 24:
            chosen_size, chosen_lines = size, wrapped
            break
        size = max(24, size - 4)
    lines = chosen_lines
    size = chosen_size

    # Build SVG. Width spans canvas; each <text> line drawn centred at
    # x=50%, dy = font_size × line_spacing.
    svg_w = cw
    line_h = int(size * el.line_spacing)
    svg_h = line_h * len(lines) + el.outline_width * 2 + 8
    tspans = []
    for i, line in enumerate(lines):
        y = el.outline_width + size + i * line_h
        tspans.append(
            f'<text x="50%" y="{y}" text-anchor="middle" '
            f'font-family="\'Noto Sans TC\', \'Yu Gothic\', sans-serif" '
            f'font-size="{size}" font-weight="900" '
            f'fill="{el.text_color}" stroke="{el.outline_color}" '
            f'stroke-width="{el.outline_width}" stroke-linejoin="round" '
            f'paint-order="stroke fill">{_esc(line)}</text>'
        )
    svg_inner = "".join(tspans)

    shadow_filter = ""
    if el.shadow_offset > 0 and el.shadow_alpha > 0:
        shadow_filter = (
            f'filter="drop-shadow({el.shadow_offset}px {el.shadow_offset}px 0 '
            f'{_color_alpha(el.shadow_color, el.shadow_alpha)})"'
        )

    svg = (
        f'<svg width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg" '
        f'{shadow_filter}>{svg_inner}</svg>'
    )

    css = textwrap.dedent(f"""
        #el{idx} {{
          position: absolute;
          left: 0;
          {y_css}
          width: {svg_w}px;
          height: {svg_h}px;
          text-align: center;
        }}
    """).strip()
    return (
        f'<div id="el{idx}" class="clip" data-start="{el.start_time}" '
        f'data-duration="{el.end_time - el.start_time:.3f}" '
        f'data-track-index="2">{svg}</div>',
        css,
    )


def _render_text(el: TextElement, idx: int) -> tuple[str, str]:
    """A v5-style floating TextElement. Pixel-anchored."""
    if isinstance(el.anchor, tuple):
        ax, ay = el.anchor
    else:
        ax, ay = 16, 16  # auto fallback
    shadow_css = ""
    if el.shadow_offset:
        sx, sy = el.shadow_offset
        shadow_css = (
            f"text-shadow: {sx}px {sy}px 0 rgba(0, 0, 0, 0.5);"
        )
    css = textwrap.dedent(f"""
        #el{idx} {{
          position: absolute;
          left: {ax}px;
          top: {ay}px;
          font-family: 'Noto Sans TC', 'Yu Gothic', sans-serif;
          font-size: {el.size}px;
          font-weight: 900;
          color: {el.color};
          -webkit-text-stroke: {el.outline_width}px {el.outline_color};
          {shadow_css}
        }}
    """).strip()
    return (
        f'<div id="el{idx}" class="clip" data-start="{el.start_time}" '
        f'data-duration="{el.end_time - el.start_time:.3f}" '
        f'data-track-index="2">{_esc(el.content)}</div>',
        css,
    )


def _render_hero(el: HeroTextElement, idx: int, canvas_size: tuple[int, int]) -> tuple[str, str]:
    cw, ch = canvas_size
    if el.pos == "center":
        pos_css = "left: 50%; top: 50%; transform: translate(-50%, -50%);"
    elif el.pos == "center_upper":
        pos_css = "left: 50%; top: 25%; transform: translate(-50%, -50%);"
    elif el.pos == "center_lower":
        pos_css = "left: 50%; top: 75%; transform: translate(-50%, -50%);"
    elif el.pos == "upper_left":
        pos_css = "left: 15%; top: 25%; transform: translate(-50%, -50%);"
    elif el.pos == "upper_right":
        pos_css = "left: 85%; top: 25%; transform: translate(-50%, -50%);"
    elif isinstance(el.pos, tuple):
        pos_css = f"left: {el.pos[0]}px; top: {el.pos[1]}px;"
    else:
        pos_css = "left: 50%; top: 25%; transform: translate(-50%, -50%);"
    glow = (
        f"text-shadow: 0 0 40px {el.halo_color}, 0 0 80px {el.halo_color}, "
        f"0 8px 24px rgba(0, 0, 0, 0.6);"
    )
    css = textwrap.dedent(f"""
        #el{idx} {{
          position: absolute;
          {pos_css}
          font-family: 'Noto Sans TC', 'Yu Gothic', sans-serif;
          font-size: {el.size}px;
          font-weight: 900;
          color: {el.color};
          {glow}
        }}
    """).strip()
    return (
        f'<div id="el{idx}" class="clip" data-start="{el.start_time}" '
        f'data-duration="{el.end_time - el.start_time:.3f}" '
        f'data-track-index="3">{_esc(el.content)}</div>',
        css,
    )


def _render_decoration(
    el: DecorationElement, idx: int, asset_src: Path | None,
) -> tuple[str, str]:
    """Single-anchor decoration (pixel_anchor or near_text resolved)."""
    if el.pixel_anchor is not None:
        ax, ay = el.pixel_anchor
    else:
        # No anchor; default to top-right-ish corner so it shows up somewhere.
        ax, ay = 24, 24
    size = el.base_size or 80
    src_str = f"./assets/{asset_src.name}" if asset_src else ""
    css = textwrap.dedent(f"""
        #el{idx} {{
          position: absolute;
          left: {ax}px;
          top: {ay}px;
          width: {size}px;
          height: {size}px;
        }}
        #el{idx} img {{ width: 100%; height: 100%; pointer-events: none; }}
    """).strip()
    if src_str:
        img_html = f'<img src="{_esc(src_str)}" alt="" />'
    else:
        img_html = ""  # missing asset → invisible div, renderer skips silently
    return (
        f'<div id="el{idx}" class="clip" data-start="{el.start_time}" '
        f'data-duration="{el.end_time - el.start_time:.3f}" '
        f'data-track-index="1">{img_html}</div>',
        css,
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def build_composition(
    decision: Decision,
    *,
    canvas_size: tuple[int, int],
    out_dir: Path,
    fps: int = 30,
    asset_lookup: callable | None = None,
) -> CompositionResult:
    """Generate composition.html + an assets/ folder under `out_dir`.

    `asset_lookup(tag) -> Path | None` resolves a DecorationElement's
    asset_tag to a PNG on disk. Falls back to None (decoration silently
    skipped) when not provided or the tag has no asset.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = out_dir / "assets"
    assets_dir.mkdir(exist_ok=True)

    # Beat-locked pulse period (from v9). None when beat-sync was off.
    beat_period = (
        decision.global_style.beat_period_sec if decision.global_style else None
    )

    canvas_w, canvas_h = canvas_size
    duration = max(
        (el.end_time for el in decision.elements), default=10.0,
    ) + 0.2  # tail buffer so the last frame includes the exit animation

    body_chunks: list[str] = []
    css_chunks: list[str] = []
    gsap_lines: list[str] = []

    # ---- v6: lazy-import the asset retriever ONCE here so we share the
    # same recently-used dedup across the whole composition.
    if asset_lookup is None:
        try:
            from semanticvibe.asset_retrieval import AssetRetriever
            _ret = AssetRetriever()
            def _lookup(tag: str, color_bucket: str | None = None) -> Path | None:
                rec = _ret.get(tag, prefer_color_bucket=color_bucket)
                if rec is None:
                    return None
                return _ret.repo_root / rec["file"]
            asset_lookup = _lookup
        except Exception as exc:  # noqa: BLE001
            log.warning("AssetRetriever init failed (%s); decorations will be empty.", exc)
            asset_lookup = lambda tag, color_bucket=None: None  # noqa: E731

    for idx, el in enumerate(decision.elements):
        selector = f"#el{idx}"
        try:
            # Gate visibility for every element (skipped for the discriminator
            # check; concrete branches below add the entry + idle tweens).
            gsap_lines.extend(_gate_visibility(selector, el.start_time, el.end_time))

            if isinstance(el, SubtitleOutlinedElement):
                markup, css = _render_subtitle_outlined(el, idx, canvas_size)
                body_chunks.append(markup); css_chunks.append(css)
                gsap_lines.append(_entry_gsap("fade", selector, el.start_time))
                # Subtitles use a soft fade-in instead of registry animation.
            elif isinstance(el, SubtitleBannerElement):
                # Treat banner as outlined for now (chip background is a v9
                # legacy and the user explicitly wanted plain outlined).
                fake = SubtitleOutlinedElement(
                    content=el.content,
                    start_time=el.start_time,
                    end_time=el.end_time,
                    position=el.position,
                    font=el.font,
                    size=el.size,
                    text_color=el.text_color,
                    outline_color=el.outline_color,
                    outline_width=el.outline_width,
                    margin=el.margin,
                    reasoning=el.reasoning,
                )
                markup, css = _render_subtitle_outlined(fake, idx, canvas_size)
                body_chunks.append(markup); css_chunks.append(css)
                gsap_lines.append(_entry_gsap("fade", selector, el.start_time))
            elif isinstance(el, HeroTextElement):
                markup, css = _render_hero(el, idx, canvas_size)
                body_chunks.append(markup); css_chunks.append(css)
                gsap_lines.append(_entry_gsap("fade", selector, el.start_time))
                if el.breathing:
                    gsap_lines.extend(_idle_gsap(
                        "pulse", selector, el.start_time, beat_period_sec=beat_period,
                    ))
            elif isinstance(el, TextElement):
                markup, css = _render_text(el, idx)
                body_chunks.append(markup); css_chunks.append(css)
                gsap_lines.append(_entry_gsap(el.animation, selector, el.start_time))
                gsap_lines.extend(_idle_gsap(
                    el.idle_animation, selector, el.start_time,
                    seed=idx, beat_period_sec=beat_period,
                ))
            elif isinstance(el, DecorationElement):
                src_path = asset_lookup(
                    el.asset_tag, getattr(el, "prefer_color_bucket", None),
                ) if asset_lookup else None
                copied_src: Path | None = None
                if src_path is not None and src_path.exists():
                    copied_src = assets_dir / f"el{idx}_{src_path.name}"
                    shutil.copy2(src_path, copied_src)
                markup, css = _render_decoration(el, idx, copied_src)
                body_chunks.append(markup); css_chunks.append(css)
                gsap_lines.append(_entry_gsap(el.animation, selector, el.start_time))
                gsap_lines.extend(_idle_gsap(
                    el.idle_animation, selector, el.start_time,
                    seed=idx, beat_period_sec=beat_period,
                ))
        except Exception as exc:  # noqa: BLE001 — never block whole render on one element
            log.warning("hyperframes adapter: element %d skipped (%s)", idx, exc)

    body_html = "\n      ".join(body_chunks) if body_chunks else "<!-- empty -->"
    css_block = "\n".join(css_chunks)
    gsap_block = "\n      ".join(gsap_lines) if gsap_lines else "// no animations"

    composition_html = textwrap.dedent(f"""\
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="UTF-8" />
            <meta name="viewport" content="width={canvas_w}, height={canvas_h}" />
            <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
            <style>
              * {{ margin: 0; padding: 0; box-sizing: border-box; }}
              html, body {{
                width: {canvas_w}px; height: {canvas_h}px;
                overflow: hidden;
                background: transparent;
                font-family: 'Noto Sans TC', 'Yu Gothic', 'Segoe UI', sans-serif;
              }}
              #root {{ position: relative; width: 100%; height: 100%; background: transparent; }}
              .clip {{ position: absolute; }}
              {css_block}
            </style>
          </head>
          <body>
            <div id="root"
                 data-composition-id="main"
                 data-start="0"
                 data-duration="{duration:.3f}"
                 data-width="{canvas_w}"
                 data-height="{canvas_h}">
              {body_html}
            </div>
            <script>
              window.__timelines = window.__timelines || {{}};
              const tl = gsap.timeline({{ paused: true }});
              {gsap_block}
              window.__timelines["main"] = tl;
              window.__duration = {duration:.3f};
              window.__fps = {fps};
              window.__ready = true;
            </script>
          </body>
        </html>
    """)
    html_path = out_dir / "composition.html"
    html_path.write_text(composition_html, encoding="utf-8")

    log.info(
        "[hf-adapter] wrote %s (%.2fs, %d elements, %d gsap tweens)",
        html_path, duration, len(decision.elements), len(gsap_lines),
    )
    return CompositionResult(
        html_path=html_path,
        duration_sec=float(duration),
        canvas_size=canvas_size,
        assets_dir=assets_dir,
        fps=fps,
    )
