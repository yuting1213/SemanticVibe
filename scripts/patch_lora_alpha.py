"""Patch Z-Image-Turbo LoRA .safetensors files to add missing `.alpha` keys.

The artificialguybr Z-Image LoRAs (Doodle.Redmond / Stickers.Redmond)
ship without per-layer `alpha` tensors. diffusers' Z-Image LoRA converter
calls `state_dict.pop("...alpha").item()` and dies with KeyError on the
first lookup.

Standard LoRA convention: `alpha` is a scalar that scales the lora_B
output. Most trainers set `alpha = rank` (so the effective scale is
alpha/rank = 1.0) and rely on the loader's `lora_scale` for any further
adjustment. We follow that — for every `<prefix>.lora_A.weight` we add
`<prefix>.alpha` as a scalar tensor of value `rank`, where `rank` is
lora_A's first dim.

Usage:
    uv run python scripts/patch_lora_alpha.py loras/Foo.safetensors
    # → writes loras/Foo_alpha.safetensors next to it (original untouched)

Or batch the entire loras/ dir:
    uv run python scripts/patch_lora_alpha.py loras/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def patch_one(src: Path) -> Path:
    if "_alpha.safetensors" in src.name:
        print(f"  skip {src.name} (already patched)")
        return src

    out = src.with_name(src.stem + "_alpha.safetensors")
    if out.exists():
        print(f"  skip {src.name} → {out.name} already exists")
        return out

    tensors: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        # safe_open exposes metadata via .metadata()
        meta = f.metadata() or {}
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
        metadata.update(meta)

    added = 0
    for key in list(tensors.keys()):
        if not key.endswith(".lora_A.weight"):
            continue
        prefix = key[: -len(".lora_A.weight")]
        alpha_key = f"{prefix}.alpha"
        if alpha_key in tensors:
            continue
        rank = tensors[key].shape[0]  # lora_A: [rank, in_features]
        # Match dtype of lora_A so loader doesn't have a dtype mismatch.
        tensors[alpha_key] = torch.tensor(float(rank), dtype=tensors[key].dtype)
        added += 1

    save_file(tensors, str(out), metadata=metadata)
    print(f"  patched {src.name} → {out.name} (+{added} alpha keys, total {len(tensors)})")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path, help="A .safetensors file or a directory of them.")
    args = p.parse_args(argv)

    targets: list[Path] = []
    if args.path.is_file():
        targets = [args.path]
    elif args.path.is_dir():
        targets = sorted(p for p in args.path.glob("*.safetensors") if "_alpha" not in p.name)
    else:
        print(f"error: {args.path} is neither a file nor a directory", file=sys.stderr)
        return 2

    if not targets:
        print(f"no .safetensors found under {args.path}")
        return 1

    print(f"patching {len(targets)} file(s):")
    for src in targets:
        patch_one(src)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
