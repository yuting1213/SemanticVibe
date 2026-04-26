"""CLIP-based semantic asset retrieval.

Given a tag string from a `DecorationElement`, return the best-matching asset
file from the library by cosine similarity in CLIP embedding space.
"""

from __future__ import annotations

from pathlib import Path

from semanticvibe.assets.library import AssetEntry, AssetLibrary


def find_asset(library: AssetLibrary, tag: str, *, top_k: int = 1) -> list[AssetEntry]:
    """Return up to `top_k` best-matching asset entries for `tag`.

    Tag-exact matches are prioritised; falls back to CLIP cosine similarity
    over the textual tag against pre-computed image embeddings.
    """
    exact = library.by_tag(tag)
    if exact:
        return exact[:top_k]
    raise NotImplementedError(
        "CLIP fallback retrieval lands in Week 4 (open-clip ViT-L/14 + cached embeddings)."
    )


def precompute_embeddings(library: AssetLibrary, cache_path: Path) -> None:
    """One-off: encode every asset in `library` and persist to `cache_path`."""
    raise NotImplementedError("Week 4: open-clip image-encoder pass + np.save.")
