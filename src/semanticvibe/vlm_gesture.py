"""v13 — VLM-driven gesture anchoring.

Pipeline: for each motion peak (from v12), send the corresponding frame
to a local VLM (qwen2.5vl:7b via Ollama), parse the gesture label
against a closed vocabulary (`assets/gesture_vocabulary.json`), and
return a list of GestureEvents the renderer can drop in as first-class
decorations anchored to the dancer's actual action.

Closed-vocab pattern mirrors v6 (`semantic_align.py`): the prompt lists
the legal labels, the parser drops anything outside the list. Disk
cache pattern mirrors `semantic_align._load_cache` / `_save_cache` so
re-renders on the same video skip the ~2.5 min VLM pass.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import TypedDict

import cv2
from pydantic import BaseModel

log = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_VOCAB_PATH = _REPO_ROOT / "assets" / "gesture_vocabulary.json"
_CACHE_DIR = _REPO_ROOT / ".cache" / "vlm_gestures"

_OLLAMA_DEFAULT_HOST = "http://localhost:11434"
_OLLAMA_DEFAULT_MODEL = "qwen2.5vl:7b"
_REQUEST_TIMEOUT_SEC = 120


# ---------------------------------------------------------------------------
# Vocab loading
# ---------------------------------------------------------------------------


def _load_vocab() -> tuple[list[dict], str, str]:
    """Returns (gestures, fallback_id, vocab_fingerprint)."""
    if not _VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"gesture vocab not found at {_VOCAB_PATH}; checkout incomplete?"
        )
    with _VOCAB_PATH.open(encoding="utf-8") as f:
        raw = json.load(f)
    gestures = list(raw["gestures"])
    fallback = str(raw.get("fallback_gesture", "none"))
    # Fingerprint busts the cache when the vocab is edited.
    fp = hashlib.md5(
        json.dumps(gestures, sort_keys=True, ensure_ascii=False).encode("utf-8"),
    ).hexdigest()[:8]
    return gestures, fallback, fp


_GESTURES, _FALLBACK_GESTURE, _VOCAB_FINGERPRINT = _load_vocab()
_VALID_GESTURE_IDS: set[str] = {g["id"] for g in _GESTURES}
_GESTURE_TAG_MAP: dict[str, str | None] = {g["id"]: g.get("tag") for g in _GESTURES}
_GESTURE_ANIM_MAP: dict[str, str | None] = {g["id"]: g.get("animation") for g in _GESTURES}
# v14: gesture → anchor-symbol (e.g. "right_index", "mid_wrists_above")
# resolved against MediaPipe landmark indices in build_elements.
_GESTURE_ANCHOR_MAP: dict[str, str | None] = {
    g["id"]: g.get("anchor_landmark") for g in _GESTURES
}


def gesture_anchor_symbol(gesture: str) -> str | None:
    """Look up the anchor symbol (e.g. 'right_index') for a gesture id."""
    return _GESTURE_ANCHOR_MAP.get(gesture)


# Defensive: every non-null tag must exist in the closed tag vocab.
def _validate_vocab_against_tags() -> None:
    try:
        from semanticvibe.semantic_align import VALID_TAGS
    except Exception:  # pragma: no cover — circular-import edge case
        return
    used = {t for t in _GESTURE_TAG_MAP.values() if t is not None}
    bad = used - VALID_TAGS
    if bad:
        raise RuntimeError(
            f"gesture_vocabulary.json references tags not in tag_vocabulary.json: "
            f"{sorted(bad)}"
        )


_validate_vocab_against_tags()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


_ZONE_VALUES = {
    "top_left", "top_right", "bottom_left", "bottom_right", "none",
}

# Animation names the renderer schema accepts. Must stay in sync with
# AnimationName in schemas/decision.py. We validate VLM-reported
# animations against this set and silently fall through to the gesture-
# vocab default when the VLM emits something else (e.g. "none").
_VALID_ANIMATIONS = {
    "bounce_in", "typewriter", "wiggle", "draw_in", "fade",
    "scale_pop", "drop_in",
    "slide_in_left", "slide_in_right", "slide_in_top", "slide_in_bottom",
    "stamp", "wobble_in", "spin_in",
}


class GestureEvent(BaseModel):
    time: float
    gesture: str
    tag: str | None
    animation: str | None
    confidence: float = 0.7
    # v13.1: VLM-reported per-frame metadata. action is free text (debug);
    # emotion is an enum (future v14 may drive idle animation by emotion);
    # zone tells the renderer where the empty space is — caller maps to a
    # pixel anchor via _zone_to_anchor.
    action: str | None = None
    emotion: str | None = None
    zone: str | None = None
    # v14: optional per-peak landmark snapshot, normalised [0, 1] of
    # source frame. Shape (N_LANDMARKS, 2). Set by detect_gestures when
    # the caller passes `landmarks_by_time`; consumed by build_elements
    # for hand/face-anchored placement. None falls back to zone-based.
    landmarks_normalised: list[list[float]] | None = None


class GestureInfo(TypedDict):
    events: list[GestureEvent]
    model: str
    cache_hit: bool


# ---------------------------------------------------------------------------
# Cache helpers (mirror semantic_align pattern)
# ---------------------------------------------------------------------------


def _cache_key(
    video_path: str,
    peak_times: list[float],
    model: str,
) -> str:
    p = Path(video_path).resolve()
    mtime = p.stat().st_mtime if p.exists() else 0
    payload = {
        "path": str(p),
        "mtime": mtime,
        "model": model,
        "vocab_fp": _VOCAB_FINGERPRINT,
        "peaks": [round(t, 3) for t in peak_times],
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _load_cache(key: str) -> list[GestureEvent] | None:
    f = _CACHE_DIR / f"{key}.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return [GestureEvent.model_validate(e) for e in data]
    except Exception as exc:  # noqa: BLE001
        log.warning("vlm_gesture cache miss (corrupt %s): %s", f.name, exc)
        return None


def _save_cache(key: str, events: list[GestureEvent]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = [json.loads(e.model_dump_json()) for e in events]
    (_CACHE_DIR / f"{key}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Frame extraction + VLM call
# ---------------------------------------------------------------------------


def _build_prompt() -> str:
    """v13.1 structured-JSON prompt. Asks for action description + emotion +
    composition zone + gesture tag/animation/confidence in one pass.

    Key wins over the v13 single-label prompt:
    - `format: "json"` (set in _ask_vlm) forces valid JSON
    - confidence < 0.45 → caller drops the event (no more "VLM guessed")
    - best_empty_zone → renderer places decoration in the actual empty
      spot the VLM sees, instead of computing it geometrically
    - action / emotion are free text for debug + future v14 hooks
    """
    gesture_ids = [g["id"] for g in _GESTURES]
    return f"""Analyse this dance-video frame. Respond with STRICT JSON only.

{{
  "action":     "<one sentence in 繁體中文, very specific — e.g. '右手舉起比V字、左手叉腰'>",
  "emotion":    "<one of: excited|shy|intense|calm|playful|serious>",
  "composition": {{
    "subject_main_zone": "<one of: top|center|bottom|left|right>",
    "best_empty_zone":   "<one of: top_left|top_right|bottom_left|bottom_right|none>"
  }},
  "gesture":    "<one of: {' | '.join(gesture_ids)}>",
  "animation":  "<one of: stamp|fade|spin_in|scale_pop|drop_in|slide_in_left>",
  "confidence": <0.0 to 1.0 — how sure you are about the gesture>
}}

Rules:
- If you genuinely cannot identify a clear, deliberate gesture (e.g. she
  is just walking or the frame is motion-blurred mid-step), set
  gesture="none" and confidence<0.4. Do NOT guess.
- "point_at_camera" requires SINGLE arm extended forward with finger
  visible. Both arms down at sides ≠ point_at_camera.
- "arms_raised" requires BOTH arms clearly above the shoulder line.
- "heart_hands" requires hands forming a heart shape, not just framing
  the face.
- "peace_sign" requires V-sign with index + middle finger near face/head.

Output ONLY the JSON object. No prose, no markdown fences."""


def _extract_frame_jpeg_b64(
    video_path: str, t_sec: float, max_width: int = 512,
) -> str | None:
    """Grab the frame closest to t_sec and return base64 JPEG."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    # JPEG is smaller than PNG → fewer tokens for the VLM context.
    ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok2:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _ask_vlm(
    image_b64: str,
    *,
    model: str,
    host: str,
    prompt: str,
) -> dict | None:
    """v13.1: returns the parsed JSON dict (or None on failure).

    Uses Ollama's `format: "json"` flag so we get a syntactically valid
    JSON object back even when the VLM goes off-script.
    """
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2, "num_predict": 256},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SEC) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama unreachable at {host}: {exc}") from exc
    except TimeoutError as exc:
        log.warning("VLM timed out on one frame (%s); skipping", exc)
        return None
    text = (payload.get("response") or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # Tolerate stray prose surrounding the JSON.
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e == -1:
            log.debug("VLM returned non-JSON: %r", text[:200])
            return None
        try:
            return json.loads(text[s : e + 1])
        except json.JSONDecodeError:
            log.debug("Could not salvage JSON from %r: %s", text[:200], exc)
            return None


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


_MIN_CONFIDENCE = 0.45


def detect_gestures(
    video_path: str,
    peak_times: list[float],
    *,
    model: str = _OLLAMA_DEFAULT_MODEL,
    host: str | None = None,
    use_cache: bool = True,
    min_confidence: float = _MIN_CONFIDENCE,
    landmarks_by_time: dict[float, "np.ndarray | None"] | None = None,
) -> GestureInfo:
    """v13.1: structured-JSON gesture detection.

    Each VLM call returns {action, emotion, composition.{subject_main_zone,
    best_empty_zone}, gesture, animation, confidence}. Events below
    `min_confidence` are dropped — VLM admitting "I'm not sure" is far
    better than the v13 "VLM always guesses a label" failure mode.
    """
    host = host or _OLLAMA_DEFAULT_HOST

    if not peak_times:
        return GestureInfo(events=[], model=model, cache_hit=False)

    key = _cache_key(video_path, peak_times, model)
    if use_cache:
        cached = _load_cache(key)
        if cached is not None:
            log.info(
                "[vlm_gesture] cache hit %s: %d events from %d peaks (model=%s)",
                key, len(cached), len(peak_times), model,
            )
            return GestureInfo(events=cached, model=model, cache_hit=True)

    prompt = _build_prompt()
    events: list[GestureEvent] = []
    dropped_unknown = 0
    dropped_null = 0
    dropped_low_conf = 0
    for t in peak_times:
        b64 = _extract_frame_jpeg_b64(video_path, t)
        if b64 is None:
            continue
        result = _ask_vlm(b64, model=model, host=host, prompt=prompt)
        if result is None:
            continue

        gesture = str(result.get("gesture", "")).strip().lower()
        if gesture not in _VALID_GESTURE_IDS:
            dropped_unknown += 1
            log.debug("[vlm_gesture] t=%.2f: unknown gesture %r", t, gesture)
            continue

        # Confidence floor.
        try:
            conf = float(result.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        if conf < min_confidence:
            dropped_low_conf += 1
            log.debug("[vlm_gesture] t=%.2f: low confidence %.2f for %r",
                      t, conf, gesture)
            continue

        # Non-actionable gestures (pose_static / lean_or_sway / none) drop.
        tag = _GESTURE_TAG_MAP[gesture]
        if tag is None:
            dropped_null += 1
            continue

        # VLM can override the default animation, but only if it picks
        # a known one. "none" or junk → fall back to gesture-vocab default.
        vlm_anim = str(result.get("animation", "")).strip().lower()
        if vlm_anim in _VALID_ANIMATIONS:
            animation = vlm_anim
        else:
            animation = _GESTURE_ANIM_MAP[gesture]

        # Composition zone — validated against the known enum.
        comp = result.get("composition") or {}
        zone = str(comp.get("best_empty_zone", "")).strip().lower() or None
        if zone not in _ZONE_VALUES:
            zone = None
        if zone == "none":
            zone = None  # treat 'none' as "no preference"

        # v14: attach the matching landmark snapshot if the caller threaded
        # them through. Stored as list[list[float]] so the JSON disk
        # cache round-trips cleanly.
        lm_arr = None
        if landmarks_by_time is not None:
            arr = landmarks_by_time.get(float(t))
            if arr is None:
                # Try the nearest float key — small float-precision drift
                # between motion_detector and our dict lookup can lose
                # the exact match.
                close = [k for k in landmarks_by_time if abs(k - float(t)) < 0.05]
                if close:
                    arr = landmarks_by_time[close[0]]
            if arr is not None:
                lm_arr = [[float(x), float(y)] for x, y in arr.tolist()]

        events.append(GestureEvent(
            time=float(t),
            gesture=gesture,
            tag=tag,
            animation=animation,
            confidence=conf,
            action=str(result.get("action", "")).strip()[:120] or None,
            emotion=str(result.get("emotion", "")).strip().lower() or None,
            zone=zone,
            landmarks_normalised=lm_arr,
        ))

    log.info(
        "[vlm_gesture] %d/%d peaks → %d valid events "
        "(unknown=%d, low_conf=%d, non-actionable=%d, model=%s, cache=False)",
        len(events), len(peak_times), len(events),
        dropped_unknown, dropped_low_conf, dropped_null, model,
    )

    if use_cache and events:
        try:
            _save_cache(key, events)
        except OSError as exc:
            log.warning("vlm_gesture cache write failed (%s)", exc)

    return GestureInfo(events=events, model=model, cache_hit=False)
