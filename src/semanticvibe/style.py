"""Style preset loader + accessor.

Reads `assets/styles.json` (the v7 style registry) and exposes:

- `load_styles()` → dict of preset_name → preset_dict
- `get_style(name)` → preset_dict for a named preset, falling back to default
- `style_names()` → list of all preset names

Each preset carries:
- `subtitle_default`: "banner" or "hero" (the default mode for that style)
- `text`: TextElement palette (color / outline / halo)
- `subtitle_banner`: full SubtitleBannerElement style fields
- `decoration_color_palette`: target colours for decoration variety
- `ambient_tags`: tag list to top up under-tagged highlights
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STYLES_FILE = _REPO_ROOT / "assets" / "styles.json"


@lru_cache(maxsize=1)
def load_styles() -> dict[str, Any]:
    if not _STYLES_FILE.exists():
        raise FileNotFoundError(
            f"styles file not found at {_STYLES_FILE}; checkout incomplete?"
        )
    with _STYLES_FILE.open(encoding="utf-8") as f:
        return json.load(f)


def style_names() -> list[str]:
    return list(load_styles().get("presets", {}).keys())


def default_style_name() -> str:
    return str(load_styles().get("default", "pink_handdrawn"))


def get_style(name: str | None) -> dict[str, Any]:
    """Return the named preset; on miss returns the default preset."""
    styles = load_styles()
    presets = styles.get("presets", {})
    if name and name in presets:
        return presets[name]
    fallback = styles.get("default", "pink_handdrawn")
    return presets.get(fallback, {})
