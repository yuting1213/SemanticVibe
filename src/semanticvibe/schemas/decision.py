"""Stage 2 → Stages 3–5 contract.

Discriminated union on `type` so the LLM emits a flat list and downstream
stages dispatch on `Element.type` without isinstance trees.

Every element carries a `reasoning` field — spec §5.2.2 mandates chain-of-thought.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

# Animation set — keep in sync with semanticvibe.render.animations dispatch.
AnimationName = Literal["bounce_in", "typewriter", "wiggle", "draw_in", "fade"]

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


Element = Annotated[Union[TextElement, DecorationElement], Field(discriminator="type")]


class GlobalStyle(BaseModel):
    color_palette: list[str] = Field(min_length=1)
    vibe: str = Field(min_length=1)


class Decision(BaseModel):
    elements: list[Element]
    global_style: GlobalStyle

    def text_elements(self) -> list[TextElement]:
        return [e for e in self.elements if isinstance(e, TextElement)]

    def decoration_elements(self) -> list[DecorationElement]:
        return [e for e in self.elements if isinstance(e, DecorationElement)]
