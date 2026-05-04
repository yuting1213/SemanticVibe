"""generate_stickers.py — batch sticker generation for the SemanticVibe asset library.

Pipeline:
1. Read `stickers_config.json` (categories + variants).
2. For each variant × N seeds: run SDXL (or DALL-E fallback) with the
   hand-drawn doodle prompt template.
3. Auto-remove background via rembg.
4. Crop to content + 20px padding.
5. Save to `assets/stickers/{category}/{name}_{seed}.png`.
6. Append metadata to `assets/stickers/index.json`.

Failures don't kill the run — they're appended to `failed.log` and
processing continues.

Usage:
    uv run python scripts/generate_stickers.py                       # auto-pick backend
    uv run python scripts/generate_stickers.py --backend dalle       # force DALL-E
    uv run python scripts/generate_stickers.py --dry-run             # print prompts only
    uv run python scripts/generate_stickers.py --variants-per-name 2 # fewer seeds

Prerequisites:
    uv sync --extra sdxl   # diffusers + accelerate + rembg
    # Drop any doodle LoRA .safetensors files into ./loras/ (auto-loaded).
    # For DALL-E fallback: export OPENAI_API_KEY=...
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import random
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("generate_stickers")


# ---- Constants --------------------------------------------------------------

PROMPT_TEMPLATE = (
    "hand-drawn doodle {category}, {color_hint}, chunky marker outline, "
    "slightly wobbly imperfect lines, white background, sticker style, "
    "kawaii Japanese zine aesthetic, flat illustration, single isolated "
    "object, centered, no shadow"
)

NEGATIVE_PROMPT = (
    "3d, realistic, photo, gradient, shadow, multiple objects, text, "
    "signature, watermark, busy background, color background"
)

DEFAULT_VARIANTS_PER_NAME = 4
DEFAULT_PADDING_PX = 20
DEFAULT_OUTPUT_ROOT = Path("assets/stickers")
DEFAULT_CONFIG = Path("stickers_config.json")
DEFAULT_LORAS_DIR = Path("loras")
DEFAULT_FAILED_LOG = Path("failed.log")

SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_INFERENCE_STEPS = 30
SDXL_GUIDANCE_SCALE = 7.5
SDXL_HEIGHT = 1024
SDXL_WIDTH = 1024
SDXL_LORA_SCALE = 0.7

# Z-Image Turbo: distilled DiT, 8 NFE, low CFG (it was distilled away).
# 6B params at bf16 ≈ 12 GB, so cpu_offload is required on a 12 GB card.
ZIMAGE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_INFERENCE_STEPS = 8
ZIMAGE_GUIDANCE_SCALE = 1.0
# 512 instead of 1024 — at 768 we observed 12 GB VRAM pinned at 96 % and
# silent paging (70 s / step, leak across images). 512 cuts activations
# another ~2.25 × so the GPU has actual headroom. Stickers get cropped to
# content + 20 px padding so 512 inputs typically yield 300-450 px outputs,
# which is still oversized vs how SemanticVibe uses them on a 720 p canvas.
ZIMAGE_HEIGHT = 512
ZIMAGE_WIDTH = 512

DALLE_MODEL = "dall-e-3"
DALLE_SIZE = "1024x1024"


def _safe_adapter_name(idx: int, lora_path: Path) -> str:
    """diffusers requires adapter names to be valid Python identifiers."""
    sanitised = re.sub(r"[^a-zA-Z0-9]", "_", lora_path.stem)[:32].strip("_") or "lora"
    return f"adapter_{idx}_{sanitised}"


# ---- Config dataclasses -----------------------------------------------------


@dataclass
class Variant:
    name: str
    color_hint: str


@dataclass
class CategorySpec:
    category: str
    variants: list[Variant]


def load_config(path: Path) -> list[CategorySpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: list[CategorySpec] = []
    for row in raw:
        out.append(
            CategorySpec(
                category=row["category"],
                variants=[Variant(name=v["name"], color_hint=v["color_hint"]) for v in row["variants"]],
            )
        )
    return out


def build_prompt(category: str, color_hint: str) -> str:
    # Underscores in category names (e.g. "speech_bubble") confuse SDXL's
    # tokenizer — it reads them as a single oddball token. Replace with
    # spaces so "music_note" becomes "music note" in the prompt.
    readable = category.replace("_", " ")
    return PROMPT_TEMPLATE.format(category=readable, color_hint=color_hint)


# ---- Backend: SDXL ----------------------------------------------------------


def cuda_available() -> bool:
    """True if torch is importable and a CUDA device is visible."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 — torch missing is one of several failure modes
        return False


def load_sdxl_pipeline(loras_dir: Path, lora_scale: float):
    """Load SDXL base pipeline + every LoRA found under `loras_dir`.

    Lazy import — diffusers + torch are slow to import and only needed on
    the SDXL code path.
    """
    import torch
    from diffusers import StableDiffusionXLPipeline

    log.info("Loading SDXL base pipeline (%s)…", SDXL_MODEL_ID)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        SDXL_MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    # CPU offload swaps the largest blocks back to CPU between calls — keeps
    # SDXL + a small LoRA + rembg coexisting on a 12 GB card.
    pipe.enable_model_cpu_offload()

    _attach_loras(pipe, loras_dir, lora_scale, label="SDXL")
    return pipe


def sdxl_generate(pipe, prompt: str, seed: int):
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=SDXL_INFERENCE_STEPS,
        guidance_scale=SDXL_GUIDANCE_SCALE,
        generator=generator,
        height=SDXL_HEIGHT,
        width=SDXL_WIDTH,
    )
    return result.images[0]


# ---- Backend: Z-Image Turbo -------------------------------------------------


def load_zimage_pipeline(loras_dir: Path, lora_scale: float, *, cpu_offload: bool = False):
    """Load Z-Image-Turbo + every LoRA found under `loras_dir`.

    Z-Image is a 6B-param DiT model. At bf16 the weights sit at ~12 GB,
    which is right at the edge of an RTX 3060 12 GB. We default to
    *no* cpu_offload because the offload path runs ~10 min per image
    (the transformer keeps shuttling between CPU and GPU per step). At
    768×768 the activations fit alongside the weights on a 12 GB card.

    cpu_offload=True is the slow-but-safe fallback if the GPU OOMs.
    """
    import torch
    from diffusers import ZImagePipeline

    log.info(
        "Loading Z-Image Turbo pipeline (%s, cpu_offload=%s)…",
        ZIMAGE_MODEL_ID,
        cpu_offload,
    )
    pipe = ZImagePipeline.from_pretrained(
        ZIMAGE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    _attach_loras(pipe, loras_dir, lora_scale, label="Z-Image")
    return pipe


def zimage_generate(pipe, prompt: str, seed: int):
    """Generate a single image via Z-Image Turbo. CFG ≈ 1.0 because the
    Turbo distillation removed CFG dependency; higher values tend to
    over-saturate.
    """
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=ZIMAGE_INFERENCE_STEPS,
            guidance_scale=ZIMAGE_GUIDANCE_SCALE,
            generator=generator,
            height=ZIMAGE_HEIGHT,
            width=ZIMAGE_WIDTH,
        )
    except TypeError:
        # Some Turbo distillations drop negative_prompt support.
        result = pipe(
            prompt=prompt,
            num_inference_steps=ZIMAGE_INFERENCE_STEPS,
            guidance_scale=ZIMAGE_GUIDANCE_SCALE,
            generator=generator,
            height=ZIMAGE_HEIGHT,
            width=ZIMAGE_WIDTH,
        )
    return result.images[0]


# ---- Shared LoRA loading ----------------------------------------------------


def _attach_loras(pipe, loras_dir: Path, lora_scale: float, *, label: str) -> None:
    """Discover every .safetensors under `loras_dir` and load them as adapters.

    Adapter names are sanitised because diffusers requires Python-identifier
    style ([a-zA-Z0-9_]). Filenames like `[ZImage.Turbo]Doodle_Redmond` would
    otherwise blow up at load time.
    """
    if not loras_dir.exists():
        log.warning("LoRA dir %s does not exist — running base %s only.", loras_dir, label)
        return
    loras = sorted(loras_dir.glob("*.safetensors"))
    if not loras:
        log.warning("No LoRA found in %s — running base %s only.", loras_dir, label)
        return
    adapter_names: list[str] = []
    for i, lora in enumerate(loras):
        adapter_name = _safe_adapter_name(i, lora)
        log.info("Loading %s LoRA %s as %s", label, lora.name, adapter_name)
        pipe.load_lora_weights(str(lora), adapter_name=adapter_name)
        adapter_names.append(adapter_name)
    pipe.set_adapters(adapter_names, adapter_weights=[lora_scale] * len(adapter_names))


# ---- Backend: DALL-E --------------------------------------------------------


def dalle_generate(prompt: str):
    """DALL-E 3 fallback — slower, costs money, no negative prompt support."""
    from openai import OpenAI
    from PIL import Image

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot use DALL-E fallback")

    client = OpenAI(api_key=api_key)
    response = client.images.generate(
        model=DALLE_MODEL,
        prompt=prompt,
        size=DALLE_SIZE,
        n=1,
    )
    img_url = response.data[0].url
    if img_url is None:
        raise RuntimeError("DALL-E returned no URL")
    with urllib.request.urlopen(img_url) as resp:  # noqa: S310 — vendor URL is trusted
        img_bytes = resp.read()
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


# ---- Post-processing --------------------------------------------------------


def remove_background(img):
    from rembg import remove

    return remove(img)  # PIL.Image RGBA


def crop_to_content(img, padding: int = DEFAULT_PADDING_PX):
    """Crop the RGBA image to its non-transparent bounding box + padding."""
    rgba = img.convert("RGBA")
    alpha = rgba.split()[-1]
    bbox = alpha.getbbox()
    if bbox is None:
        # Fully transparent → return a tiny placeholder rather than blow up.
        return rgba
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(rgba.width, x2 + padding)
    y2 = min(rgba.height, y2 + padding)
    return rgba.crop((x1, y1, x2, y2))


# ---- Metadata helpers -------------------------------------------------------


_HEX_RE = re.compile(r"#([0-9A-Fa-f]{6})")


def hex_from_hint(color_hint: str) -> str | None:
    m = _HEX_RE.search(color_hint)
    return f"#{m.group(1).upper()}" if m else None


_STYLE_KEYWORDS = [
    ("outline", "outline"),
    ("double", "double-traced"),
    ("solid", "solid"),
    ("filled", "solid"),
    ("dashed", "dashed"),
    ("flat", "flat"),
]


def style_from_hint(color_hint: str) -> str:
    hint = color_hint.lower()
    for needle, label in _STYLE_KEYWORDS:
        if needle in hint:
            return label
    return "default"


# ---- Main loop --------------------------------------------------------------


def _record_failure(failed_log: Path, key: str, exc: BaseException) -> None:
    msg = f"{key}: {type(exc).__name__}: {exc}\n"
    with failed_log.open("a", encoding="utf-8") as f:
        f.write(msg)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="generate_stickers")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--loras-dir", type=Path, default=DEFAULT_LORAS_DIR)
    p.add_argument("--lora-scale", type=float, default=SDXL_LORA_SCALE)
    p.add_argument("--failed-log", type=Path, default=DEFAULT_FAILED_LOG)
    p.add_argument("--variants-per-name", type=int, default=DEFAULT_VARIANTS_PER_NAME)
    p.add_argument("--padding", type=int, default=DEFAULT_PADDING_PX)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument(
        "--backend",
        choices=["auto", "sdxl", "zimage", "dalle"],
        default="auto",
        help="auto = SDXL when CUDA is available, otherwise DALL-E. Use "
        "'zimage' for Z-Image Turbo (6B DiT, 8 NFE, requires zimage-tuned LoRAs).",
    )
    p.add_argument(
        "--zimage-cpu-offload",
        action="store_true",
        help="Z-Image only: enable model_cpu_offload (slow but safe on <12 GB GPUs). "
        "Default off — the script auto-falls-back to offload on CUDA OOM.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each (variant, seed) and its prompt; do not generate images.",
    )
    args = p.parse_args(argv)

    if not args.config.exists():
        log.error("Config not found: %s", args.config)
        return 2

    specs = load_config(args.config)
    total = sum(len(s.variants) * args.variants_per_name for s in specs)
    log.info(
        "Loaded %d categories / %d variants → %d total renders",
        len(specs),
        sum(len(s.variants) for s in specs),
        total,
    )

    # --- Dry run --------------------------------------------------------
    if args.dry_run:
        rng = random.Random(args.seed_base)
        for spec in specs:
            for variant in spec.variants:
                prompt = build_prompt(spec.category, variant.color_hint)
                seeds = [rng.randint(1, 999999) for _ in range(args.variants_per_name)]
                log.info(
                    "[dry-run] %s/%s × %d seeds (%s)",
                    spec.category,
                    variant.name,
                    len(seeds),
                    ", ".join(str(s) for s in seeds),
                )
                log.info("           prompt: %s", prompt)
        return 0

    # --- Backend selection ---------------------------------------------
    backend = args.backend
    if backend == "auto":
        backend = "sdxl" if cuda_available() else "dalle"
        log.info("Auto-selected backend: %s", backend)

    args.output_root.mkdir(parents=True, exist_ok=True)
    pipe = None
    if backend == "sdxl":
        pipe = load_sdxl_pipeline(args.loras_dir, args.lora_scale)
    elif backend == "zimage":
        try:
            pipe = load_zimage_pipeline(
                args.loras_dir, args.lora_scale, cpu_offload=args.zimage_cpu_offload
            )
        except Exception as exc:  # noqa: BLE001
            # Catches CUDA OOM (torch.cuda.OutOfMemoryError) + import errors.
            log.warning(
                "Z-Image load failed (%s) — retrying with cpu_offload=True", exc
            )
            pipe = load_zimage_pipeline(
                args.loras_dir, args.lora_scale, cpu_offload=True
            )
    elif backend == "dalle":
        if not os.environ.get("OPENAI_API_KEY"):
            log.error("DALL-E backend requires OPENAI_API_KEY in the environment.")
            return 2

    # --- Generation loop -----------------------------------------------
    try:
        from tqdm import tqdm
    except ImportError:
        # tqdm should be in deps but graceful fallback so the script doesn't die.
        def tqdm(iterable=None, total=None, desc=None):  # type: ignore[no-redef]
            return iterable if iterable is not None else range(total or 0)

    index: list[dict] = []
    failed_count = 0
    rng = random.Random(args.seed_base)
    progress = tqdm(total=total, desc="generating")

    for spec in specs:
        category_dir = args.output_root / spec.category
        category_dir.mkdir(parents=True, exist_ok=True)

        for variant in spec.variants:
            prompt = build_prompt(spec.category, variant.color_hint)
            for _v_idx in range(args.variants_per_name):
                seed = rng.randint(1, 999999)
                key = f"{spec.category}/{variant.name}@{seed}"
                out_path = category_dir / f"{variant.name}_{seed}.png"
                try:
                    if backend == "sdxl":
                        raw = sdxl_generate(pipe, prompt, seed)
                    elif backend == "zimage":
                        raw = zimage_generate(pipe, prompt, seed)
                    else:
                        raw = dalle_generate(prompt)
                    nobg = remove_background(raw)
                    cropped = crop_to_content(nobg, padding=args.padding)
                    cropped.save(out_path)
                    index.append(
                        {
                            "file": f"{spec.category}/{out_path.name}",
                            "category": spec.category,
                            "color": hex_from_hint(variant.color_hint),
                            "style": style_from_hint(variant.color_hint),
                            "size": [cropped.width, cropped.height],
                            "seed": seed,
                            "prompt": prompt,
                            "backend": backend,
                        }
                    )
                except Exception as exc:  # noqa: BLE001 — keep the loop running on any failure
                    failed_count += 1
                    _record_failure(args.failed_log, key, exc)
                    log.warning("FAILED %s — recorded to %s", key, args.failed_log)
                progress.update(1) if hasattr(progress, "update") else None
    if hasattr(progress, "close"):
        progress.close()

    index_path = args.output_root / "index.json"
    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info(
        "Wrote %d entries to %s. Failed: %d (see %s).",
        len(index),
        index_path,
        failed_count,
        args.failed_log,
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
