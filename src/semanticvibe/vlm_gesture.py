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


class GestureEvent(BaseModel):
    time: float
    gesture: str
    tag: str | None
    animation: str | None
    confidence: float = 0.7


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


_PROMPT_HEADER = (
    "Look at this image of a young woman who is dancing or vlogging.\n"
    "What single gesture or action is she most clearly doing right now?\n\n"
    "Pick EXACTLY ONE label from this list:\n"
)
_PROMPT_FOOTER = (
    "\nRespond with ONLY the label, lowercase, nothing else. No explanation."
)


def _build_prompt() -> str:
    lines = [_PROMPT_HEADER]
    for g in _GESTURES:
        lines.append(f"- {g['id']}")
    lines.append(_PROMPT_FOOTER)
    return "\n".join(lines)


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
) -> str | None:
    """Returns the raw, lowercased first-line answer, or None on failure."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 32},
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
    text = (payload.get("response") or "").strip().lower()
    if not text:
        return None
    return text.split("\n")[0].strip().rstrip(".,!?")


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def detect_gestures(
    video_path: str,
    peak_times: list[float],
    *,
    model: str = _OLLAMA_DEFAULT_MODEL,
    host: str | None = None,
    use_cache: bool = True,
) -> GestureInfo:
    """Per-peak gesture detection. See module docstring.

    Returns `GestureInfo` with an `events` list of `GestureEvent` records,
    one per peak that produced a valid in-vocabulary gesture (peaks
    rejected by the closed-vocab filter or with `tag: null` mappings
    are silently dropped — they're "recognised but not actionable").
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
    for t in peak_times:
        b64 = _extract_frame_jpeg_b64(video_path, t)
        if b64 is None:
            continue
        raw = _ask_vlm(b64, model=model, host=host, prompt=prompt)
        if raw is None:
            continue
        # Tolerate trailing punctuation / leading "label:" prefix.
        label = raw.replace("label:", "").strip().rstrip(".,!?").strip()
        if label not in _VALID_GESTURE_IDS:
            dropped_unknown += 1
            log.debug("[vlm_gesture] t=%.2f: unknown label %r, dropping", t, raw)
            continue
        tag = _GESTURE_TAG_MAP[label]
        anim = _GESTURE_ANIM_MAP[label]
        if tag is None:
            dropped_null += 1
            log.debug("[vlm_gesture] t=%.2f: label %s is non-actionable", t, label)
            continue
        events.append(GestureEvent(
            time=float(t), gesture=label, tag=tag, animation=anim,
        ))

    log.info(
        "[vlm_gesture] %d/%d peaks → %d valid events "
        "(unknown=%d, non-actionable=%d, model=%s, cache=False)",
        len(events), len(peak_times), len(events),
        dropped_unknown, dropped_null, model,
    )

    if use_cache and events:
        try:
            _save_cache(key, events)
        except OSError as exc:
            log.warning("vlm_gesture cache write failed (%s)", exc)

    return GestureInfo(events=events, model=model, cache_hit=False)
