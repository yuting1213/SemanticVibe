"""Dual-zone forbidden-area layout (v10).

Combines the MediaPipe person mask with the rendered subtitle bboxes
into a single per-time "where can I NOT put a decoration?" boolean
grid. Decorations then call `find_free_position(target_size)` to land
in the remaining empty area, with optional zone preferences (corners).

Design notes:
- The mask lives at canvas-pixel resolution. Caller scales the
  pose mask to canvas size before adding it.
- Person mask is dilated by `padding_iters` to give the silhouette
  some breathing room — uses scipy.ndimage.binary_dilation.
- Subtitle bboxes get a hard-coded 15 px padding (small — subtitles
  are already centred, no need to push decorations farther away).
- find_free_position samples the canvas every 20 px (configurable).
  Yes that's coarse, but the resulting spacing is exactly right for
  hand-drawn-sticker placement and saves us from O(W·H) work per
  decoration.
- Returns the BEST candidate (highest score) but with a top-K random
  pick so consecutive frames don't choose the same exact pixel.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class ForbiddenMap:
    """A single-time bool grid of forbidden pixels for one canvas frame."""

    def __init__(self, video_width: int, video_height: int):
        self.W = int(video_width)
        self.H = int(video_height)
        self.mask: np.ndarray = np.zeros((self.H, self.W), dtype=bool)

    def add_rect(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        *,
        padding: int = 10,
    ) -> None:
        """Mark a padded axis-aligned rectangle as forbidden."""
        x0 = max(0, int(x) - padding)
        y0 = max(0, int(y) - padding)
        x1 = min(self.W, int(x) + int(w) + padding)
        y1 = min(self.H, int(y) + int(h) + padding)
        if x1 > x0 and y1 > y0:
            self.mask[y0:y1, x0:x1] = True

    def add_person_mask(
        self,
        person_mask: np.ndarray,
        *,
        padding_iters: int = 10,
    ) -> None:
        """Union an MediaPipe-derived person mask, dilated by `padding_iters`.

        `person_mask` must already be at canvas resolution (caller's
        responsibility — pose_detector returns video-resolution masks
        and the renderer scales them with cv2.INTER_NEAREST).
        """
        if person_mask is None:
            return
        if person_mask.shape != self.mask.shape:
            log.warning(
                "ForbiddenMap.add_person_mask: shape mismatch %r vs %r — skipping",
                person_mask.shape, self.mask.shape,
            )
            return
        if padding_iters > 0:
            try:
                from scipy.ndimage import binary_dilation
                dilated = binary_dilation(person_mask, iterations=padding_iters)
            except Exception as exc:  # noqa: BLE001
                log.warning("scipy dilation failed (%s); using raw mask", exc)
                dilated = person_mask
        else:
            dilated = person_mask
        self.mask |= dilated.astype(bool)

    def coverage_pct(self) -> float:
        """Fraction of the canvas that is currently forbidden — diagnostic."""
        return float(self.mask.mean())

    def find_free_position(
        self,
        target_size: tuple[int, int],
        *,
        prefer_zone: Optional[tuple[int, int, int, int]] = None,
        rng: Optional[random.Random] = None,
        step: int = 20,
        top_k: int = 8,
    ) -> Optional[tuple[int, int]]:
        """Return a (top-left x, y) where a `target_size` rectangle fits
        entirely outside the forbidden mask, or None if no such position
        exists.

        Args:
            target_size: (width, height) of the rectangle to place.
            prefer_zone: optional (x0, y0, x1, y1) — candidates inside
                this zone get +100 to their score so they outrank
                neutral candidates.
            rng: a `random.Random` for deterministic top-K randomisation.
                If None, the highest-scored candidate wins outright.
            step: pixel stride for the grid sweep. Smaller = more
                positions but slower. 20 px is plenty for sticker
                placement.
            top_k: pick uniformly from the K highest-scored candidates
                so consecutive frames don't always lock to the same pixel.
        """
        tw, th = int(target_size[0]), int(target_size[1])
        if tw >= self.W or th >= self.H:
            return None

        candidates: list[tuple[float, int, int]] = []
        for x in range(0, self.W - tw, max(1, step)):
            for y in range(0, self.H - th, max(1, step)):
                region = self.mask[y : y + th, x : x + tw]
                if region.size == 0 or region.any():
                    continue
                score = 0.0
                if prefer_zone is not None:
                    px0, py0, px1, py1 = prefer_zone
                    if px0 <= x <= px1 and py0 <= y <= py1:
                        score += 100.0
                # Slight preference for non-edge positions so decorations
                # don't bleed against the canvas frame.
                edge_distance = min(x, y, self.W - x - tw, self.H - y - th)
                score += min(edge_distance, 50) * 0.5
                candidates.append((score, x, y))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        pool = candidates[: max(1, min(top_k, len(candidates)))]
        if rng is None:
            return pool[0][1], pool[0][2]
        choice = rng.choice(pool)
        return choice[1], choice[2]


def build_forbidden_map_at_time(
    t: float,
    *,
    person_masks: dict[float, np.ndarray] | None,
    subtitle_rects: list[tuple[float, float, int, int, int, int]],
    canvas_size: tuple[int, int],
    person_padding_iters: int = 10,
    subtitle_padding: int = 15,
    pose_match_window_sec: float = 1.0,
) -> ForbiddenMap:
    """Construct the forbidden map active at time `t`.

    Args:
        t: the absolute video time the map represents.
        person_masks: time → bool ndarray (canvas-resolution OR
            video-resolution; caller scales to canvas first if needed).
            Only samples within `pose_match_window_sec` of `t` are used.
        subtitle_rects: pre-computed list of
            (start_time, end_time, x, y, w, h) for every subtitle/banner
            that COULD be active. We filter by t ∈ [start, end] here.
        canvas_size: (width, height) of the render canvas.
        person_padding_iters: scipy binary-dilation iterations for the
            silhouette breathing room.
        subtitle_padding: pixels of padding around each active subtitle
            rect.
        pose_match_window_sec: max gap between sampled mask time and `t`.
            Beyond that we ignore the mask (better to forbid nothing
            than to forbid the wrong region).
    """
    fmap = ForbiddenMap(canvas_size[0], canvas_size[1])
    if person_masks:
        nearest_t = min(person_masks.keys(), key=lambda k: abs(k - t))
        if abs(nearest_t - t) < pose_match_window_sec:
            fmap.add_person_mask(
                person_masks[nearest_t], padding_iters=person_padding_iters,
            )
    for st, et, x, y, w, h in subtitle_rects:
        if st <= t <= et:
            fmap.add_rect(int(x), int(y), int(w), int(h), padding=subtitle_padding)
    return fmap
