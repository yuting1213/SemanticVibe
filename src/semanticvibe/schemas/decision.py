"""Stage 2 → Stages 3–5 contract.

Discriminated union on `type` so the LLM emits a flat list and downstream
stages dispatch on `Element.type` without isinstance trees.

Every element carries a `reasoning` field — spec §5.2.2 mandates chain-of-thought.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Entry animation set — keep in sync with semanticvibe.render.animations.REGISTRY.
# The first 5 are the legacy v1 set (committed example JSONs reference them);
# the rest are the v4 IG-Reels-style expansion.
AnimationName = Literal[
    "bounce_in", "typewriter", "wiggle", "draw_in", "fade",
    "scale_pop", "drop_in",
    "slide_in_left", "slide_in_right", "slide_in_top", "slide_in_bottom",
    "stamp", "wobble_in", "spin_in",
]

# Idle (steady-state) animation — layered on top of the entry envelope between
# entry-end and exit-start. "none" = stay still after entry settles.
# Keep in sync with semanticvibe.render.idle_animations.REGISTRY.
IdleAnimationName = Literal[
    "none", "pulse", "wiggle", "drift", "rotate_slow", "shimmer",
]

# (x, y) in pixel coordinates of the output frame, top-left origin.
PixelAnchor = tuple[int, int]


class _ElementBase(BaseModel):
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    reasoning: str = Field(
        min_length=1,
        description="Chain-of-thought rationale from the LLM for placing this element. "
        "Required by spec §5.2.2 — do not strip when serialising for downstream stages.",
    )

    @model_validator(mode="after")
    def _times_consistent(self) -> "_ElementBase":
        if self.end_time <= self.start_time:
            raise ValueError(f"end_time ({self.end_time}) must be > start_time ({self.start_time})")
        return self


class OutlineLayer(BaseModel):
    """One layer of the rendered text's outline stack.

    Multiple layers stack from outermost (drawn first) to innermost (drawn
    last, on top). Mimics the manga / sticker look where a glyph has e.g.
    a thick white halo + a thin coloured outline + the fill.
    """

    color: str
    width: int = Field(ge=0)


class TextElement(_ElementBase):
    type: Literal["text"] = "text"
    content: str = Field(min_length=1)
    anchor: Literal["auto"] | PixelAnchor = Field(
        default="auto",
        description='"auto" defers to the layout stage; a pixel tuple pins the element.',
    )
    font: str = Field(description="Font family name; resolved against data/fonts/.")
    size: int = Field(gt=0, description="Pixel size of the rendered glyph height.")
    color: str = Field(description="Fill colour — any Pillow-acceptable CSS string or hex.")
    outline_color: str
    outline_width: int = Field(ge=0)
    outline_layers: list[OutlineLayer] = Field(
        default_factory=list,
        description="Optional extra outline layers stacked OUTSIDE the primary "
        "outline_color/outline_width. Each layer adds its width to the previous "
        "ones, so [{'color':'#fff','width':4}] produces a white halo around "
        "the existing outline. Empty list = single outline (the legacy default).",
    )
    animation: AnimationName
    idle_animation: IdleAnimationName = Field(
        default="none",
        description="Steady-state modulation layered on top of the entry "
        "envelope between entry-end and exit-start. Lets a settled element "
        "keep moving (pulse / drift / shimmer / etc.) without re-triggering "
        "the entry animation.",
    )
    rotation_jitter: float = Field(
        default=0.0, description="Max rotation in degrees applied as random jitter."
    )
    shadow_offset: tuple[int, int] | None = Field(
        default=None,
        description="(dx, dy) pixel offset of an additional drop-shadow layer. "
        "None = no shadow.",
    )

    @field_validator("anchor", mode="before")
    @classmethod
    def _anchor_from_list(cls, v):
        # JSON has no tuple type — accept lists of length 2 as anchors.
        if isinstance(v, list) and len(v) == 2:
            return tuple(v)
        return v

    @field_validator("shadow_offset", mode="before")
    @classmethod
    def _shadow_from_list(cls, v):
        if isinstance(v, list) and len(v) == 2:
            return tuple(v)
        return v


class DecorationElement(_ElementBase):
    type: Literal["decoration"] = "decoration"
    asset_tag: str = Field(
        min_length=1,
        description="Semantic tag the asset library will resolve via CLIP search.",
    )
    near_text_id: int | None = Field(
        default=None,
        description="If set, layout stage clusters this decoration near that "
        "TextElement (index into Decision.elements).",
    )
    scale_jitter: float = Field(default=0.0, ge=0)
    rotation_jitter: float = Field(default=0.0)
    count: int = Field(
        default=1,
        ge=1,
        le=64,
        description="Number of copies to render. Use with scatter=True for "
        "confetti-style spreads (10-15 hearts across the frame).",
    )
    scatter: bool = Field(
        default=False,
        description="If True, the `count` copies are placed at deterministic "
        "pseudo-random positions across the frame (avoiding subjects via the "
        "occupancy map). If False, copies stack at the resolved anchor.",
    )
    color_tint: list[str] = Field(
        default_factory=list,
        description="Optional per-copy colour tints applied to the asset. "
        "Cycled if shorter than `count`. Empty list = no tinting.",
    )
    base_size: int | None = Field(
        default=None,
        gt=0,
        description="Override the asset's natural pixel size. None = use the "
        "asset PNG's native dimensions.",
    )
    scatter_zone: tuple[int, int, int, int] | None = Field(
        default=None,
        description="(x1, y1, x2, y2) in canvas pixel coordinates. When set + "
        "scatter=True, all copies land inside this bbox instead of the full "
        "frame — produces clustered placements (e.g. 'all in the upper-left').",
    )
    size_steps: list[int] | None = Field(
        default=None,
        description="Per-copy base_size values, cycled if shorter than count. "
        "Lets one decoration emit big+medium+small siblings (e.g. [200, 80, "
        "80, 80, 40, 40, 40, 40] for the 'one large + a few medium + a few "
        "small' baseline cluster look).",
    )
    wiggle_amp: float = Field(
        default=0.0,
        ge=0,
        description="Steady-state position wiggle amplitude (pixels) — legacy "
        "field, prefer the unified `idle_animation='wiggle'` going forward. "
        "Both compose additively so authored JSONs that set wiggle_amp keep "
        "working.",
    )
    prefer_color_bucket: str | None = Field(
        default=None,
        description="Optional colour-bucket hint passed to AssetRetriever "
        "(e.g. 'pink', 'green'). Narrows the candidate pool for this "
        "decoration's PNG; falls back to the default colour-balanced pick "
        "when no PNG in the requested bucket exists.",
    )
    pixel_anchor: PixelAnchor | None = Field(
        default=None,
        description="v10: explicit (x, y) top-left pixel position chosen "
        "by the forbidden-map layout. When set the renderer uses this "
        "directly and skips the near_text_id heuristic.",
    )

    @field_validator("pixel_anchor", mode="before")
    @classmethod
    def _pixel_anchor_from_list(cls, v):
        if isinstance(v, list) and len(v) == 2:
            return tuple(v)
        return v
    animation: AnimationName = Field(
        default="fade",
        description="Entry animation. Defaults to 'fade' to match the v1 "
        "decoration behaviour where this field didn't exist.",
    )
    idle_animation: IdleAnimationName = Field(
        default="none",
        description="Steady-state modulation, same semantics as TextElement's.",
    )

    @field_validator("scatter_zone", mode="before")
    @classmethod
    def _zone_from_list(cls, v):
        if isinstance(v, list) and len(v) == 4:
            return tuple(v)
        return v


HeroPosition = Literal[
    "center_upper", "center", "center_lower", "upper_left", "upper_right"
]


class HeroTextElement(_ElementBase):
    """Single huge centred glyph (or short phrase) drawn in chalk style.

    Distinct from TextElement because:
    - Position is keyword-bucketed, not pixel-precise.
    - Render path uses multi-blur halos + grain dots for the chalk look,
      not stacked outlines.
    - Animation envelope is a long slow fade with subtle scale breathing,
      not the playful bounce/typewriter set.
    """

    type: Literal["hero_text"] = "hero_text"
    content: str = Field(min_length=1, max_length=8)
    pos: HeroPosition | PixelAnchor = Field(default="center_upper")
    size: int = Field(default=350, gt=0)
    color: str = Field(default="#FFFFFF")
    style: Literal["chalk", "outline"] = "chalk"
    breathing: bool = Field(
        default=True,
        description="Subtle scale oscillation throughout the visible window — "
        "reads as a quietly breathing object rather than static text.",
    )
    font: str = Field(default="KleeOne-SemiBold")
    halo_color: str = Field(
        default="#FFFFFF",
        description="Colour of the soft outer halo blur (chalk style only).",
    )
    grain: bool = Field(
        default=True,
        description="If True, scatter small white dots/lines on top of the "
        "fill to simulate chalk dust.",
    )
    idle_animation: IdleAnimationName = Field(
        default="none",
        description="Optional idle modulation. Note: HeroTextElement already "
        "supports `breathing` (a built-in scale pulse on a fixed period). "
        "Setting idle_animation='pulse' is roughly equivalent; pick one to "
        "avoid double-modulating.",
    )

    @field_validator("pos", mode="before")
    @classmethod
    def _pos_from_list(cls, v):
        if isinstance(v, list) and len(v) == 2:
            return tuple(v)
        return v


SubtitlePosition = Literal["top_banner", "bottom_banner", "center"]


class SubtitleOutlinedElement(_ElementBase):
    """v10 subtitle: pure text with thick outline + optional shadow.

    No background chip — the lyric reads as overlay text against the
    raw video. Uses a circular thick-outline implementation (multiple
    offset draws within a radius) for the けんぱ-style cute look.

    The optional `outline_color_alt` lets per-instance composition
    alternate two outline colours (caller picks which one to use; the
    renderer just draws whatever `outline_color` is set to).
    """

    type: Literal["subtitle_outlined"] = "subtitle_outlined"
    content: str = Field(min_length=1)
    position: SubtitlePosition = "top_banner"
    font: str = Field(default="KleeOne-SemiBold")
    size: int = Field(default=64, gt=0)
    text_color: str = "#FFFFFF"
    outline_color: str = "#FF6B9D"
    outline_width: int = Field(default=6, ge=0)
    shadow_offset: int = Field(default=2, ge=0,
        description="Drop-shadow offset in pixels. 0 = no shadow.")
    shadow_alpha: int = Field(default=120, ge=0, le=255)
    shadow_color: str = "#000000"
    margin: int = Field(default=16, ge=0)
    max_width_ratio: float = Field(default=0.85, gt=0, le=1.0)
    wrap_lines: bool = Field(
        default=True,
        description="When the rendered tile would exceed canvas_w * "
        "max_width_ratio, attempt to break the text into up to `max_lines` "
        "lines BEFORE shrinking the font. Mandarin/Japanese/Korean wrap "
        "anywhere; Latin/Cyrillic wraps at spaces only.",
    )
    max_lines: int = Field(default=2, ge=1, le=4)
    line_spacing: float = Field(default=1.15, gt=0,
        description="Multiplier on the font's natural line height.")


class SubtitleBannerElement(_ElementBase):
    """One full-line lyric chip — the baseline3-vibe rendering for v7.

    Distinct from TextElement because:
    - Background is a semi-transparent rounded rectangle behind the glyph row.
    - Position is keyword-bucketed (top_banner / bottom_banner / center),
      not a pixel anchor — the banner stretches to the canvas width minus
      a configurable margin.
    - Style fields collapse the multi-layer outline stack into a single
      colour + width; outline + halo come from the style preset, not
      authored per-element.
    """

    type: Literal["subtitle_banner"] = "subtitle_banner"
    content: str = Field(min_length=1)
    position: SubtitlePosition = "top_banner"
    font: str = Field(default="KleeOne-SemiBold")
    size: int = Field(default=42, gt=0)
    text_color: str = "#FFFFFF"
    outline_color: str = "#A0E847"
    outline_width: int = Field(default=4, ge=0)
    bg_color: str = "#A0E847"
    bg_alpha: int = Field(default=140, ge=0, le=255)
    corner_radius: int = Field(default=16, ge=0)
    padding: int = Field(default=18, ge=0)
    margin: int = Field(default=16, ge=0,
                        description="Min pixels between banner and canvas edge.")


Element = Annotated[
    Union[
        TextElement,
        DecorationElement,
        HeroTextElement,
        SubtitleBannerElement,
        SubtitleOutlinedElement,
    ],
    Field(discriminator="type"),
]


class GlobalStyle(BaseModel):
    color_palette: list[str] = Field(min_length=1)
    vibe: str = Field(min_length=1)
    # v9: when set, the renderer drives the `pulse` idle animation off this
    # period (in seconds) instead of the default 1.5s — synchronises the
    # on-screen breathing with the music's tempo.
    beat_period_sec: float | None = Field(default=None, gt=0)


class Decision(BaseModel):
    elements: list[Element]
    global_style: GlobalStyle

    def text_elements(self) -> list[TextElement]:
        return [e for e in self.elements if isinstance(e, TextElement)]

    def decoration_elements(self) -> list[DecorationElement]:
        return [e for e in self.elements if isinstance(e, DecorationElement)]
