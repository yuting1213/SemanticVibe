"""tag → sticker PNG resolver, backed by `assets/index.json`.

Reads the index produced by `scripts/build_index.py`. Provides:

- `AssetRetriever.has_tag(tag)`         → bool
- `AssetRetriever.get(tag, ...)`        → one record dict (avoiding recently
                                          used files when `avoid_recent`)
- `AssetRetriever.get_image(tag, ...)`  → an open `PIL.Image.Image` ready for
                                          compositing (RGBA)
- `AssetRetriever.get_multi(tag, n)`    → N distinct record dicts (cycles when
                                          fewer than N are available)
- `AssetRetriever._find_fallback_tag(tag)` → swaps a missing tag for a
                                          same-category sibling, or for the
                                          global fallback.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class AssetRetriever:
    def __init__(
        self,
        repo_root: Path | None = None,
        *,
        index_file: Path | None = None,
        vocab_file: Path | None = None,
        seed: int = 0,
    ) -> None:
        self.repo_root = (repo_root or Path(__file__).resolve().parent.parent.parent)
        self.index_file = index_file or (self.repo_root / "assets" / "index.json")
        self.vocab_file = vocab_file or (self.repo_root / "assets" / "tag_vocabulary.json")
        self._records: list[dict[str, Any]] = []
        self._by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._by_category: dict[str, list[str]] = defaultdict(list)
        self._fallback_tag = "heart"
        self._recent: deque[str] = deque(maxlen=8)
        self._rng = random.Random(seed)
        self._load()

    # -- loading ---------------------------------------------------------

    def _load(self) -> None:
        if self.vocab_file.exists():
            with self.vocab_file.open(encoding="utf-8") as f:
                vocab = json.load(f)
            self._fallback_tag = str(vocab.get("fallback_tag", self._fallback_tag))
            for entry in vocab.get("tags", []):
                self._by_category[entry["category"]].append(entry["id"])

        if not self.index_file.exists():
            log.warning(
                "assets/index.json missing — run `python scripts/build_index.py` first."
            )
            return
        with self.index_file.open(encoding="utf-8") as f:
            self._records = json.load(f)
        for rec in self._records:
            self._by_tag[rec["tag"]].append(rec)

    # -- query helpers ---------------------------------------------------

    def has_tag(self, tag: str) -> bool:
        return bool(self._by_tag.get(tag))

    def _find_fallback_tag(self, tag: str) -> str | None:
        """Return a substitute tag when `tag` has no PNGs.

        Strategy: same-category sibling that has assets > global fallback.
        Returns None only if the global fallback also has nothing — in
        which case the caller should skip the decoration entirely.
        """
        # 1) same-category sibling
        for cat, members in self._by_category.items():
            if tag in members:
                for sib in members:
                    if sib != tag and self.has_tag(sib):
                        log.debug(
                            "asset fallback: %r → %r (same-category sibling)", tag, sib,
                        )
                        return sib
                break
        # 2) global fallback
        if self.has_tag(self._fallback_tag):
            log.debug("asset fallback: %r → %r (global fallback)", tag, self._fallback_tag)
            return self._fallback_tag
        return None

    def _resolve(self, tag: str) -> tuple[str, list[dict[str, Any]]]:
        """Return (effective_tag, candidates). Empty list = nothing usable."""
        if self.has_tag(tag):
            return tag, list(self._by_tag[tag])
        fb = self._find_fallback_tag(tag)
        if fb is None:
            return tag, []
        return fb, list(self._by_tag[fb])

    # -- get one ---------------------------------------------------------

    def get(
        self,
        tag: str,
        *,
        avoid_recent: bool = True,
        prefer_prefix: tuple[str, ...] = ("josh_",),
        prefer_color_bucket: str | None = None,
    ) -> dict[str, Any] | None:
        """Return a single record for `tag` (or its fallback). None when
        nothing is available even via fallback.

        `prefer_prefix` biases the pick toward filenames starting with one
        of the given prefixes — used to favour the hand-drawn `josh_*`
        artworks over machine-generated `tofu_*` clusters when both exist.

        `prefer_color_bucket` (e.g. "green", "pink") narrows the pool to
        records whose `color_bucket` field matches; falls back to the full
        pool if no record in that colour exists for the tag.
        """
        eff_tag, candidates = self._resolve(tag)
        if not candidates:
            return None
        pool = candidates
        if prefer_color_bucket:
            colour_pool = [
                c for c in pool if c.get("color_bucket") == prefer_color_bucket
            ]
            if colour_pool:
                pool = colour_pool
        if prefer_prefix:
            preferred = [
                c for c in pool
                if Path(c["file"]).name.startswith(prefer_prefix)
            ]
            if preferred:
                pool = preferred
        if avoid_recent and len(pool) > 1:
            filtered = [c for c in pool if c["file"] not in self._recent]
            if filtered:
                pool = filtered
        chosen = self._rng.choice(pool)
        self._recent.append(chosen["file"])
        return chosen

    def get_image(self, tag: str, *, avoid_recent: bool = True):
        """Return an open `PIL.Image` (RGBA) for the chosen sticker, or None."""
        rec = self.get(tag, avoid_recent=avoid_recent)
        if rec is None:
            return None
        from PIL import Image

        path = self.repo_root / rec["file"]
        return Image.open(path).convert("RGBA")

    # -- get many --------------------------------------------------------

    def get_multi(self, tag: str, n: int) -> list[dict[str, Any]]:
        """Return up to `n` distinct records for `tag`, cycling when the pool
        is smaller than `n`."""
        eff_tag, candidates = self._resolve(tag)
        if not candidates:
            return []
        pool = list(candidates)
        self._rng.shuffle(pool)
        out: list[dict[str, Any]] = []
        i = 0
        while len(out) < n:
            out.append(pool[i % len(pool)])
            i += 1
        return out

    # -- colour-balanced multi-pick --------------------------------------

    @staticmethod
    def hex_to_color_bucket(hex_color: str) -> str:
        """Same coarse-bucketing as `scripts/build_index.py._color_bucket`,
        re-implemented here so the renderer can map a style preset's hex
        palette ("#A0E847") to a bucket ("green") without re-importing
        the script."""
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

    def get_balanced(
        self,
        tag: str,
        n: int,
        palette_hexes: list[str],
        *,
        prefer_prefix: tuple[str, ...] = ("josh_",),
    ) -> list[dict[str, Any]]:
        """Return `n` records for `tag`, biased toward colour-bucket variety
        across `palette_hexes`. Cycles through the palette so consecutive
        copies land in different colour buckets. Falls back gracefully to
        any-colour pick when a target bucket has no matching PNG."""
        if n <= 0:
            return []
        out: list[dict[str, Any]] = []
        buckets = [self.hex_to_color_bucket(h) for h in palette_hexes] or [None]
        for i in range(n):
            target_bucket = buckets[i % len(buckets)]
            rec = self.get(
                tag,
                prefer_color_bucket=target_bucket,
                prefer_prefix=prefer_prefix,
            )
            if rec is None:
                break
            out.append(rec)
        return out

    # -- introspection ---------------------------------------------------

    def all_tags(self) -> list[str]:
        return sorted(self._by_tag.keys())

    def count(self, tag: str) -> int:
        return len(self._by_tag.get(tag, []))
