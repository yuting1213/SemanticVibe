"""CLIP-based semantic asset retrieval.

Given a tag string from a `DecorationElement`, return the best-matching asset
file from the library by cosine similarity in CLIP embedding space.

Embeddings are computed once and cached to disk (numpy .npy alongside the
library root). Subsequent searches just load the cache and run a dot product.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from semanticvibe.assets.library import AssetEntry, AssetLibrary

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "ViT-B-32"
_DEFAULT_PRETRAINED = "openai"


@lru_cache(maxsize=1)
def _load_clip_model(model_name: str, pretrained: str, device: str):
    import open_clip
    import torch

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.to(device)
    model.eval()
    return model, preprocess, tokenizer, torch


def _cache_paths(library: AssetLibrary) -> tuple[Path, Path]:
    return library.root / "_clip_embeddings.npy", library.root / "_clip_index.json"


def precompute_embeddings(
    library: AssetLibrary,
    *,
    model_name: str = _DEFAULT_MODEL,
    pretrained: str = _DEFAULT_PRETRAINED,
    device: str = "cuda",
) -> Path:
    """Encode every asset in `library` and persist to a .npy cache.

    Re-running is idempotent: if the cache covers the current set of files
    (same filenames, same model), this is a no-op.
    """
    if not library.entries:
        log.warning("AssetLibrary at %s is empty; nothing to embed.", library.root)
        return _cache_paths(library)[0]

    cache_npy, cache_meta = _cache_paths(library)
    current_files = sorted(e.path.name for e in library.entries)

    if cache_npy.exists() and cache_meta.exists():
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
            if (
                meta.get("model") == f"{model_name}/{pretrained}"
                and meta.get("files") == current_files
            ):
                return cache_npy
        except json.JSONDecodeError:
            pass

    model, preprocess, _tokenizer, torch = _load_clip_model(model_name, pretrained, device)

    from PIL import Image

    embeds = []
    for entry in library.entries:
        img = Image.open(entry.path).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        embeds.append(feat.cpu().numpy()[0])

    arr = np.vstack(embeds).astype(np.float32)
    np.save(cache_npy, arr)
    cache_meta.write_text(
        json.dumps({"model": f"{model_name}/{pretrained}", "files": current_files}),
        encoding="utf-8",
    )
    log.info("Wrote %d embeddings to %s", len(embeds), cache_npy)
    return cache_npy


def find_asset(
    library: AssetLibrary,
    tag: str,
    *,
    top_k: int = 1,
    model_name: str = _DEFAULT_MODEL,
    pretrained: str = _DEFAULT_PRETRAINED,
    device: str = "cuda",
) -> list[AssetEntry]:
    """Return up to `top_k` best-matching asset entries for `tag`.

    Tries exact-tag matches first (cheap, no GPU); falls back to CLIP cosine
    similarity over the textual tag against pre-computed image embeddings.
    """
    exact = library.by_tag(tag)
    if exact:
        return exact[:top_k]

    if not library.entries:
        return []

    cache_npy = precompute_embeddings(
        library, model_name=model_name, pretrained=pretrained, device=device
    )
    if not cache_npy.exists():
        return []
    image_feats = np.load(cache_npy)

    model, _preprocess, tokenizer, torch = _load_clip_model(model_name, pretrained, device)
    with torch.no_grad():
        tokens = tokenizer([tag]).to(device)
        text_feat = model.encode_text(tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        text_np = text_feat.cpu().numpy()[0].astype(np.float32)

    sims = image_feats @ text_np
    top_idx = np.argsort(-sims)[:top_k]
    return [library.entries[int(i)] for i in top_idx]
