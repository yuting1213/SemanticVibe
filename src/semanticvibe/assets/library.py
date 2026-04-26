"""Asset library: a directory of PNG decorations + a JSON metadata index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssetEntry:
    path: Path
    tags: list[str]
    license: str
    source: str


class AssetLibrary:
    """In-memory view over `data/assets_lib/`. Loads metadata.json on init."""

    def __init__(self, root: Path) -> None:
        self.root = root
        meta_file = root / "metadata.json"
        if not meta_file.exists():
            self.entries: list[AssetEntry] = []
            return
        with meta_file.open(encoding="utf-8") as f:
            raw = json.load(f)
        self.entries = [
            AssetEntry(
                path=root / e["filename"],
                tags=list(e.get("tags", [])),
                license=e.get("license", "unknown"),
                source=e.get("source", "unknown"),
            )
            for e in raw
        ]

    def by_tag(self, tag: str) -> list[AssetEntry]:
        return [e for e in self.entries if tag in e.tags]
