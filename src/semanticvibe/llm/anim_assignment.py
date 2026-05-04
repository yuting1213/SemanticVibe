"""Beat-driven random animation assignment.

Used when a Decision JSON has no explicit per-element animation choice
(e.g. emitted by the heuristic with no LLM API key). Walks the elements,
classifies each `start_time` against beat info, and assigns:

- entry animation from `pool_strong` (downbeat) or `pool_normal` (normal beat)
- idle animation from `pool_idle`

The assignment is deterministic per `seed`, so the same FeatureSummary
+ same beats always yields the same Decision JSON across runs.
"""

from __future__ import annotations

import random
from dataclasses import replace

from semanticvibe.preprocess.beat_sync import BeatInfo, classify_beat, snap_to_beat
from semanticvibe.schemas.decision import (
    Decision,
    DecorationElement,
    HeroTextElement,
    TextElement,
)


# Pools tuned for the IG-Reels-style "punchy on the beat / calm between" feel.
POOL_STRONG_ENTRY = ["scale_pop", "stamp", "drop_in", "spin_in"]
POOL_NORMAL_ENTRY = ["fade", "slide_in_left", "slide_in_right", "wobble_in"]
POOL_IDLE = ["pulse", "wiggle", "drift", "rotate_slow", "shimmer"]


def assign_random_animations(
    decision: Decision,
    beat_info: BeatInfo,
    *,
    seed: int = 42,
    snap: bool = True,
) -> Decision:
    """Return a new Decision with randomized + beat-synced animations.

    Mutation rules:
    - If `snap=True`, each element's `start_time` is gently nudged to the
      nearest beat (max 0.15 s offset).
    - `animation` is replaced from POOL_STRONG_ENTRY (on a downbeat) or
      POOL_NORMAL_ENTRY (otherwise). Existing element.animation is overwritten
      to ensure variety — same goes for idle_animation.
    - HeroTextElement is left alone (its envelope is bespoke).
    """
    rng = random.Random(seed)
    new_elements = []
    for el in decision.elements:
        if isinstance(el, HeroTextElement):
            new_elements.append(el)
            continue

        new_start = el.start_time
        if snap and beat_info.beats:
            new_start = snap_to_beat(el.start_time, beat_info.beats)

        label = classify_beat(new_start, beat_info)
        pool = POOL_STRONG_ENTRY if label == "downbeat" else POOL_NORMAL_ENTRY
        anim = rng.choice(pool)
        idle = rng.choice(POOL_IDLE)

        # Preserve `end_time - start_time` duration when snapping.
        delta = new_start - el.start_time
        if isinstance(el, TextElement):
            new_elements.append(
                el.model_copy(update={
                    "animation": anim,
                    "idle_animation": idle,
                    "start_time": new_start,
                    "end_time": el.end_time + delta,
                })
            )
        elif isinstance(el, DecorationElement):
            new_elements.append(
                el.model_copy(update={
                    "animation": anim,
                    "idle_animation": idle,
                    "start_time": new_start,
                    "end_time": el.end_time + delta,
                })
            )
        else:
            new_elements.append(el)

    return replace(decision, elements=new_elements) if False else decision.model_copy(
        update={"elements": new_elements}
    )
