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

DALLE_MODEL = "dall-e-3"
DALLE_SIZE = "1024x1024"


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
    return PROMPT_TEMPLATE.format(category=category, color_hint=color_hint)


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

    if loras_dir.exists():
        loras = sorted(loras_dir.glob("*.safetensors"))
        for i, lora in enumerate(loras):
            adapter_name = f"adapter_{i}_{lora.stem}"
            log.info("Loading LoRA %s as %s", lora.name, adapter_name)
            pipe.load_lora_weights(str(lora), adapter_name=adapter_name)
        if loras:
            pipe.set_adapters([f"adapter_{i}_{lora.stem}" for i, lora in enumerate(loras)],
                              adapter_weights=[lora_scale] * len(loras))
        else:
            log.warning("No LoRA found in %s — running base SDXL only.", loras_dir)
    else:
        log.warning("LoRA dir %s does not exist — running base SDXL only.", loras_dir)
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
        choices=["auto", "sdxl", "dalle"],
        default="auto",
        help="auto = SDXL when CUDA is available, otherwise DALL-E.",
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
