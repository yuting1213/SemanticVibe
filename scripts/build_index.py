"""Scan assets/stickers/<tag>/*.png and emit assets/index.json.

Produces a flat list of records consumed by `semanticvibe.asset_retrieval`:

    {
      "file":   "assets/stickers/heart/heart_solid_red_116740.png",
      "tag":    "heart",
      "category": "emotion",
      "size":   [797, 791],
      "weight": 1.0,
      "color":  "#E63946"     # optional, copied from the legacy index if present
    }

Tags not listed in `assets/tag_vocabulary.json` are skipped with a warning so
the closed vocabulary stays authoritative. Tags present in the vocabulary
that have zero PNGs are reported as MISS so the operator can either generate
stickers or rely on the same-category fallback at retrieval time.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
STICKERS_DIR = REPO_ROOT / "assets" / "stickers"
VOCAB_FILE = REPO_ROOT / "assets" / "tag_vocabulary.json"
INDEX_OUT = REPO_ROOT / "assets" / "index.json"
LEGACY_INDEX = STICKERS_DIR / "index.json"


def _dominant_color(png_path: Path) -> str | None:
    """Return the dominant RGB hex of opaque pixels in `png_path`.

    Mean-of-opaque is good enough for biasing decoration picks: we only
    need a coarse colour-bucket signal, not a perceptually-tuned palette.
    Skips pixels with alpha < 200 so soft anti-alias borders don't drag
    the average toward background grey.
    """
    try:
        with Image.open(png_path) as im:
            arr = np.asarray(im.convert("RGBA"))
    except Exception:
        return None
    rgb = arr[..., :3]
    a = arr[..., 3]
    mask = a > 200
    if not mask.any():
        return None
    mean = rgb[mask].mean(axis=0).astype(int)
    return f"#{mean[0]:02X}{mean[1]:02X}{mean[2]:02X}"


def _color_bucket(hex_color: str) -> str:
    """Coarse-grain a hex colour into a named bucket so the retriever can
    match against a style preset's palette without exact-RGB hits.

    Buckets: red / pink / orange / yellow / green / cyan / blue / purple /
    brown / grey / black / white. Picked by H/S/V thresholds — close enough
    for "give me a green-ish sticker" decisions.
    """
    import colorsys
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.2:
        return "black"
    if s < 0.15:
        return "white" if v > 0.85 else "grey"
    deg = h * 360
    if deg < 15 or deg >= 345:
        return "red"
    if deg < 35:
        return "orange" if v > 0.5 else "brown"
    if deg < 65:
        return "yellow"
    if deg < 165:
        return "green"
    if deg < 200:
        return "cyan"
    if deg < 260:
        return "blue"
    if deg < 320:
        return "purple"
    return "pink"


def _load_legacy_metadata() -> dict[str, dict]:
    """Read the legacy `assets/stickers/index.json` (if present) to preserve
    color / prompt / seed metadata when rebuilding the canonical index.
    """
    if not LEGACY_INDEX.exists():
        return {}
    try:
        with LEGACY_INDEX.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}
    return {entry["file"]: entry for entry in data if "file" in entry}


def main() -> int:
    if not VOCAB_FILE.exists():
        print(f"error: {VOCAB_FILE} missing — run from a complete checkout.",
              file=sys.stderr)
        return 2

    with VOCAB_FILE.open(encoding="utf-8") as f:
        vocab = json.load(f)
    valid_tags: dict[str, str] = {t["id"]: t["category"] for t in vocab["tags"]}

    legacy = _load_legacy_metadata()

    records: list[dict] = []
    counts: dict[str, int] = defaultdict(int)

    if not STICKERS_DIR.exists():
        print(f"error: {STICKERS_DIR} missing.", file=sys.stderr)
        return 2

    for tag_dir in sorted(p for p in STICKERS_DIR.iterdir() if p.is_dir()):
        tag = tag_dir.name
        if tag not in valid_tags:
            print(f"  skip: dir '{tag}/' is not in tag_vocabulary.json")
            continue
        for png in sorted(tag_dir.glob("*.png")):
            try:
                with Image.open(png) as im:
                    size = list(im.size)
            except Exception as exc:  # noqa: BLE001
                print(f"  warn: cannot open {png}: {exc}")
                continue
            rel = png.relative_to(REPO_ROOT).as_posix()
            legacy_key_a = f"{tag}/{png.name}"
            legacy_key_b = rel
            legacy_entry = legacy.get(legacy_key_a) or legacy.get(legacy_key_b) or {}
            color_hex = _dominant_color(png)
            rec = {
                "file": rel,
                "tag": tag,
                "category": valid_tags[tag],
                "size": size,
                "weight": 1.0,
                "color_dominant": color_hex,
                "color_bucket": _color_bucket(color_hex) if color_hex else None,
            }
            if "color" in legacy_entry:
                rec["color"] = legacy_entry["color"]
            if "style" in legacy_entry:
                rec["style"] = legacy_entry["style"]
            records.append(rec)
            counts[tag] += 1

    print(f"\nIndexed {len(records)} sticker(s) across {len(counts)} tag(s).")
    print("Per-tag counts (vs closed vocab):")
    for tag in sorted(valid_tags):
        n = counts.get(tag, 0)
        marker = "  " if n > 0 else "  MISS"
        print(f"  {marker:>6}  {tag:<14} {n}")

    INDEX_OUT.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {INDEX_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
