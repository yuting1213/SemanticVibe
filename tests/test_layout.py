"""Layout tests — occupancy + bin-packing math, no GPU."""

from __future__ import annotations

from semanticvibe.layout.bin_packing import pack_rects
from semanticvibe.layout.occupancy import build_occupancy, subjects_in_window
from semanticvibe.preprocess.mediapipe_pose import SubjectBox


def test_occupancy_unions_padded_boxes():
    boxes = [SubjectBox(frame_time=0.0, x=100, y=100, w=50, h=50)]
    occ = build_occupancy(640, 360, boxes, padding_px=10)
    # Padded box: (90, 90) to (160, 160).
    assert occ.mask[120, 120] == 1, "centre of subject should be marked occupied"
    assert occ.mask[0, 0] == 0, "far corner should still be free"
    assert occ.mask[200, 200] == 0


def test_occupancy_clips_to_frame():
    boxes = [SubjectBox(frame_time=0.0, x=600, y=300, w=200, h=200)]
    occ = build_occupancy(640, 360, boxes, padding_px=20)
    # Box should be clipped at the right/bottom edges, not over-run.
    assert occ.mask.shape == (360, 640)


def test_subjects_in_window_filters_by_time():
    boxes = [
        SubjectBox(frame_time=0.0, x=0, y=0, w=10, h=10),
        SubjectBox(frame_time=5.0, x=0, y=0, w=10, h=10),
        SubjectBox(frame_time=10.0, x=0, y=0, w=10, h=10),
    ]
    selected = subjects_in_window(boxes, start=4.0, end=6.0)
    times = sorted(s.frame_time for s in selected)
    assert times == [5.0]


def test_pack_rects_avoids_occupancy():
    occ = build_occupancy(640, 360, [SubjectBox(0.0, 100, 50, 200, 250)], padding_px=10)
    placements = pack_rects(occ, [(120, 60)], grid_step=8, edge_margin=8)
    rect = placements[0]
    assert rect is not None
    # Rect must not overlap the padded subject region.
    occupied_pixels = occ.mask[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
    assert (occupied_pixels == 0).all()


def test_pack_rects_returns_none_when_too_big():
    occ = build_occupancy(100, 60, [], padding_px=0)
    placements = pack_rects(occ, [(200, 100)])
    assert placements == [None]


def test_pack_rects_avoids_already_placed():
    occ = build_occupancy(640, 360, [], padding_px=0)
    placements = pack_rects(occ, [(120, 60), (120, 60)], grid_step=8)
    a, b = placements
    assert a is not None and b is not None
    # The two rects must not overlap each other.
    overlap_x = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    overlap_y = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    assert overlap_x == 0 or overlap_y == 0


def test_pack_rects_lower_band_bias():
    """Empty canvas → lower-band bias should put the rect in the bottom 30%."""
    occ = build_occupancy(640, 360, [], padding_px=0)
    placements = pack_rects(occ, [(120, 60)], grid_step=8, bias="lower-band")
    rect = placements[0]
    assert rect is not None
    centre_y = rect.y + rect.h / 2
    # Target is at 0.78 * 360 = 280.8; allow a generous tolerance.
    assert centre_y > 360 * 0.55
