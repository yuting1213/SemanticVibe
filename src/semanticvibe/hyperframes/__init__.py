"""SemanticVibe × Hyperframes hybrid pipeline (v11).

Three layers:
  - adapter.py: Decision → composition.html + GSAP timeline
  - overlay_renderer.py: composition.html → transparent WebM (Puppeteer)
  - compositor.py: base video + overlay WebM → final mp4 (ffmpeg)

Plus orchestrator:
  - pipeline.render_from_decision_hyperframes(...) — top-level entry,
    drop-in replacement for render.composite.render_from_decision.

AI brains (Whisper / LLM / pose / beat) stay in semanticvibe; this
package only handles rendering.
"""

from semanticvibe.hyperframes.adapter import build_composition
from semanticvibe.hyperframes.compositor import composite_overlay
from semanticvibe.hyperframes.overlay_renderer import render_overlay_webm
from semanticvibe.hyperframes.pipeline import render_from_decision_hyperframes

__all__ = [
    "build_composition",
    "render_overlay_webm",
    "composite_overlay",
    "render_from_decision_hyperframes",
]
