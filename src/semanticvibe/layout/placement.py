"""Resolve `anchor="auto"` elements into pixel coordinates.

Top-level entry point for Stage 4. Combines an occupancy map (from MediaPipe
subject detections) with the bin-packing algorithm to find a non-colliding
placement for each text element. Decorations follow as `near_text_id`
references, so they do not need a separate pack pass.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

from semanticvibe.layout.bin_packing import pack_rects
from semanticvibe.layout.occupancy import build_occupancy, subjects_in_window
from semanticvibe.preprocess.mediapipe_pose import SubjectBox
from semanticvibe.render.text_render import fit_to_canvas, measure_text
from semanticvibe.schemas.decision import Decision, TextElement

log = logging.getLogger(__name__)


def resolve_anchors(
    decision: Decision,
    *,
    video_path: Path,
    frame_size: tuple[int, int],
    fonts_dir: Path,
    subjects: list[SubjectBox] | None = None,
) -> Decision:
    """Return a copy of `decision` with every "auto" text anchor resolved.

    Args:
        subjects: Pre-computed subject boxes. If omitted, MediaPipe is run
            here. Pass through if you already paid the detection cost.
    """
    if subjects is None:
        from semanticvibe.preprocess.mediapipe_pose import detect_subjects

        subjects = detect_subjects(video_path)

    width, height = frame_size
    new_decision = deepcopy(decision)

    # Shrink any text element whose tile is wider than the frame BEFORE
    # measuring — otherwise bin-packing rejects them as unplaceable and
    # they fall through to the "auto" fallback in render.
    for i, el in enumerate(new_decision.elements):
        if isinstance(el, TextElement) and el.anchor == "auto":
            new_decision.elements[i] = fit_to_canvas(el, fonts_dir, frame_size)

    auto_indices: list[int] = []
    auto_sizes: list[tuple[int, int]] = []
    for i, el in enumerate(new_decision.elements):
        if isinstance(el, TextElement) and el.anchor == "auto":
            tw, th = measure_text(el, fonts_dir)
            auto_indices.append(i)
            auto_sizes.append((tw, th))

    if not auto_indices:
        return new_decision

    # Average over the union of windows for all auto-anchored elements. This is
    # a coarse approximation — for tighter placement, run pack per-element with
    # its own time window. The motion of subjects across a 5-10s clip is
    # usually small enough that one occupancy pass suffices.
    in_window: list[SubjectBox] = []
    for i in auto_indices:
        el = new_decision.elements[i]
        in_window.extend(subjects_in_window(subjects, el.start_time, el.end_time))
    occupancy = build_occupancy(width, height, in_window, padding_px=24)

    placed = pack_rects(occupancy, auto_sizes, grid_step=12)

    for idx, rect in zip(auto_indices, placed):
        el = new_decision.elements[idx]
        if rect is None:
            log.warning(
                "Could not pack text element %d (%r); leaving as 'auto' for fallback.",
                idx,
                el.content,
            )
            continue
        # Replace the element with one that has a concrete anchor.
        new_decision.elements[idx] = el.model_copy(update={"anchor": (rect.x, rect.y)})

    return new_decision
