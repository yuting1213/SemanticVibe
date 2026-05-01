"""BLIP captioning on selected keyframes.

We use Salesforce/blip-image-captioning-large (~990 MB, ~2 GB VRAM) — quality
is solid on natural-scene captions and it leaves enough VRAM headroom on a
12 GB card for Whisper to coexist.

transformers 5.x retired the "image-to-text" pipeline name, so we drive the
model + processor directly rather than via the high-level `pipeline()`
factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

from semanticvibe.preprocess.keyframes import frame_time


@dataclass(frozen=True)
class FrameCaption:
    frame_time: float
    caption: str


@lru_cache(maxsize=1)
def _load_blip(model_name: str, device: str):
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name, low_cpu_mem_usage=False)
    model.to(device)
    model.eval()
    return processor, model, torch


def caption_keyframes(
    keyframes: list[Path],
    *,
    model_name: str = "Salesforce/blip-image-captioning-large",
    device: str = "cuda",
    max_new_tokens: int = 40,
) -> list[FrameCaption]:
    """Run BLIP captioning over each keyframe; return one caption per frame."""
    if not keyframes:
        return []

    processor, model, torch = _load_blip(model_name, device)

    out: list[FrameCaption] = []
    for kf in keyframes:
        img = Image.open(kf).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = processor.decode(ids[0], skip_special_tokens=True).strip()
        if text:
            out.append(FrameCaption(frame_time=frame_time(kf), caption=text))
    return out


def condense_captions(captions: list[FrameCaption], *, max_chars: int = 600) -> str:
    """Collapse per-frame captions into one description paragraph.

    No LLM call here — just dedupe near-identical captions and join. The
    Stage 2 LLM will read this directly as `FeatureSummary.video_description`.
    """
    if not captions:
        return "An untitled video clip with no caption available."

    # Dedupe by lowercased caption while preserving order.
    seen: set[str] = set()
    unique: list[FrameCaption] = []
    for c in captions:
        key = c.caption.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    parts = [f"At {c.frame_time:.1f}s: {c.caption}" for c in unique]
    text = " ".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text
