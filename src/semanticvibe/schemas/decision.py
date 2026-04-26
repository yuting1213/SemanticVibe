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
    animation: AnimationName
    rotation_jitter: float = Field(
        default=0.0, description="Max rotation in degrees applied as random jitter."
    )

    @field_validator("anchor", mode="before")
    @classmethod
    def _anchor_from_list(cls, v):
        # JSON has no tuple type — accept lists of length 2 as anchors.
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
