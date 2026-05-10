"""Classify a directory of unsorted PNGs into closed-vocab tag directories.

For each PNG in `--src`:
1. Encode with open-clip ViT-B-32 image encoder.
2. Score against text prompts built from every tag's description in
   `assets/tag_vocabulary.json`.
3. Copy (not move) the PNG into `assets/stickers/<tag>/<prefix><name>` under
   the highest-scoring tag. PNGs whose top-tag score falls below
   `--min-score` go to `assets/_inbox/_rejected/` for manual review.

Idempotent: existing destination files are left alone (`shutil.copy2` would
overwrite, so we check first). Rerunnable as new sources arrive.

Usage:
    uv run python scripts/classify_new_assets.py \\
        --src "assets/_inbox/assets/tofu assets" \\
        --src "assets/_inbox/assets/tofu assets/new assets" \\
        --device cuda --min-score 0.20 --prefix tofu_

Stays strictly inside the closed v6 vocabulary — does NOT mutate
`assets/tag_vocabulary.json`. To add new tags, edit the vocab file by hand
(and add matching `KEYWORD_TO_TAGS` entries in `semantic_align.py`).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_FILE = REPO_ROOT / "assets" / "tag_vocabulary.json"
STICKERS_DIR = REPO_ROOT / "assets" / "stickers"
REJECT_DIR = REPO_ROOT / "assets" / "_inbox" / "_rejected"


def _load_clip(device: str):
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.to(device)
    model.eval()
    return model, preprocess, tokenizer, torch


def _encode_text_prompts(vocab, model, tokenizer, torch, device):
    """Encode one prompt per tag using the description field."""
    prompts = [
        f"{t['description']}, hand-drawn cute sticker, kawaii illustration"
        for t in vocab["tags"]
    ]
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.cpu().numpy().astype(np.float32)


def _encode_image(p: Path, model, preprocess, torch, device) -> np.ndarray | None:
    try:
        img = Image.open(p).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        print(f"  skip (open failed): {p.name} ({exc})", file=sys.stderr)
        return None
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0].astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", action="append", required=True,
                   help="Source directory of unsorted PNGs. Repeatable.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--min-score", type=float, default=0.20,
                   help="Reject PNGs whose top-tag cosine score is below this.")
    p.add_argument("--prefix", default="tofu_",
                   help="Filename prefix prepended to copied files.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print routing decisions without copying files.")
    args = p.parse_args()

    with VOCAB_FILE.open(encoding="utf-8") as f:
        vocab = json.load(f)
    tag_ids = [t["id"] for t in vocab["tags"]]

    model, preprocess, tokenizer, torch = _load_clip(args.device)
    text_feats = _encode_text_prompts(vocab, model, tokenizer, torch, args.device)
    print(f"Encoded {len(tag_ids)} tag prompts on device={args.device}.")

    src_files: list[Path] = []
    for s in args.src:
        sd = Path(s)
        if not sd.exists():
            print(f"warn: source dir missing: {sd}", file=sys.stderr)
            continue
        src_files.extend(sorted(sd.glob("*.png")))

    print(f"Classifying {len(src_files)} PNG(s)…")

    routed: dict[str, int] = {tag: 0 for tag in tag_ids}
    rejected = 0
    skipped_existing = 0

    for i, src in enumerate(src_files, 1):
        feat = _encode_image(src, model, preprocess, torch, args.device)
        if feat is None:
            continue
        sims = text_feats @ feat
        top_idx = int(np.argmax(sims))
        top_tag = tag_ids[top_idx]
        top_score = float(sims[top_idx])

        if top_score < args.min_score:
            target = REJECT_DIR / f"{args.prefix}{src.name}"
            tag_label = "REJECT"
        else:
            target = STICKERS_DIR / top_tag / f"{args.prefix}{src.name}"
            tag_label = top_tag

        if target.exists():
            skipped_existing += 1
            continue

        if args.dry_run:
            if i % 25 == 0 or i == len(src_files):
                print(f"  [{i}/{len(src_files)}] {src.name} -> {tag_label} ({top_score:.2f}) [dry-run]")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            if i % 50 == 0 or i == len(src_files):
                print(f"  [{i}/{len(src_files)}] {src.name} -> {tag_label} ({top_score:.2f})")

        if tag_label == "REJECT":
            rejected += 1
        else:
            routed[top_tag] += 1

    print("\nSummary:")
    print(f"  total processed: {len(src_files)}")
    print(f"  skipped (already at destination): {skipped_existing}")
    print(f"  rejected (top-tag score < {args.min_score}): {rejected}")
    print(f"  routed by tag:")
    for tag in sorted(routed):
        if routed[tag]:
            print(f"    {tag:<14} {routed[tag]}")
    print(f"\nNext: run `uv run python scripts/build_index.py` to refresh assets/index.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
