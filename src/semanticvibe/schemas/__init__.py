"""Schemas — the narrow waist between stages.

`FeatureSummary` is the contract from Stage 1 (preprocess) into Stage 2 (LLM).
`Decision` is the contract from Stage 2 into Stages 3–5 (assets, layout, render).

Once a schema field is committed and consumed by a stage, treat it as a stable
API. Backwards-incompatible changes require an explicit version bump.
"""

from semanticvibe.schemas.decision import (
    Decision,
    DecorationElement,
    Element,
    GlobalStyle,
    TextElement,
)
from semanticvibe.schemas.feature_summary import FeatureSummary, LyricSegment

__all__ = [
    "FeatureSummary",
    "LyricSegment",
    "Decision",
    "TextElement",
    "DecorationElement",
    "Element",
    "GlobalStyle",
]
