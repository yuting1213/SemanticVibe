"""SemanticVibe: 5-stage semantic-vibe overlay pipeline.

Public surface re-exports the schemas (the narrow waist) and the top-level
orchestrator. Stage-internal modules are intentionally not re-exported here.
"""

from semanticvibe.schemas.decision import (
    Decision,
    DecorationElement,
    Element,
    GlobalStyle,
    HeroTextElement,
    TextElement,
)
from semanticvibe.schemas.feature_summary import FeatureSummary, LyricSegment

__all__ = [
    "FeatureSummary",
    "LyricSegment",
    "Decision",
    "TextElement",
    "DecorationElement",
    "HeroTextElement",
    "Element",
    "GlobalStyle",
]

__version__ = "0.1.0"
