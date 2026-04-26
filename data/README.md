# `data/` — gitignored runtime assets

Almost everything under here is excluded from git. The two committed exceptions:

- `data/README.md` (this file)
- `data/assets_lib/metadata.json` (the index for the procedural assets — content reproducible by re-running the generator)

Use this README to rebuild a working environment on a fresh checkout.

## Layout

```
data/
├── README.md                       # this file
├── fonts/                          # all gitignored
│   ├── KleeOne-Regular.ttf
│   ├── KleeOne-SemiBold.ttf
│   └── NotoSansTC-Regular.ttf      # CJK fallback for missing glyphs
├── assets_lib/
│   ├── metadata.json               # committed; tags index for the PNGs
│   └── *.png                       # gitignored; regenerable via the script
└── test_videos/                    # gitignored
    └── your_clip.mp4
```

## Fonts

Both fonts ship under the SIL Open Font License — vendor them locally rather than depending on system fonts.

- **Klee One** — primary CJK display font. Spec mandates this. Designed for Japanese, also covers Traditional + Simplified Chinese characters used in lyric overlays. Download: https://fonts.google.com/specimen/Klee+One → unzip → drop `KleeOne-Regular.ttf` and `KleeOne-SemiBold.ttf` into `data/fonts/`.
- **Noto Sans TC** — fallback for any glyph Klee One doesn't cover. Download: https://fonts.google.com/noto/specimen/Noto+Sans+TC → drop the variable font in as `NotoSansTC-Regular.ttf`.

[render/text_render.py](../src/semanticvibe/render/text_render.py) `_resolve_font_file` falls back to NotoSansTC-Regular.ttf when the requested font name isn't found, so missing or misnamed fonts produce CJK output rather than blanks or crashes.

## Decoration assets (`assets_lib/`)

Currently **9 procedurally-generated hand-drawn-style PNGs** ship via [scripts/generate_placeholder_assets.py](../scripts/generate_placeholder_assets.py):

| Filename | Tags | Notes |
|---|---|---|
| `heart.png` | heart, love | Outline-only, double-stroked wobble, pink |
| `mini_heart.png` | mini-heart, confetti, small-heart | Smaller variant for `count + scatter` confetti |
| `sparkle.png` | sparkle, celebration, shine | 4-point burst |
| `star.png` | star, musical-note | 5-point outline star, red |
| `dot.png` | dot, circle | Wobbly outline circle |
| `burst.png` | burst, emphasis, starburst, pop | Radiating wobbly lines |
| `arrow.png` | arrow, pointer, look-here | Hand-drawn down-right arrow |
| `fire.png` | fire, flame, hot, spicy | Outline flame |
| `exclaim.png` | exclaim, impact, bam, shock | Jagged 12-point impact star |

All are **outline-only with per-point jitter** — internals stay transparent so they recolour cleanly via `DecorationElement.color_tint`. The geometric filled-shape predecessors are gone (see git history if you want them back).

Regenerate after a checkout (the PNGs themselves are gitignored):

```bash
uv run python scripts/generate_placeholder_assets.py
```

`metadata.json` is committed and rewritten by the same script. The `tags` field is what `Decision.elements[].asset_tag` matches against; CLIP fills any gaps via [assets/clip_search.py](../src/semanticvibe/assets/clip_search.py) (embeddings cached at `data/assets_lib/_clip_embeddings.npy` on first use).

## Adding more assets

Two paths, in order of preference:

1. **Hand-draw or curate from CC0 sources** (OpenClipart, Pixabay CC0, the Noun Project public-domain set). Drop PNGs (transparent background) into `data/assets_lib/`, append entries to `metadata.json`, and they're available immediately.
2. **SDXL pre-generation** (optional): `uv sync --extra sdxl` then run a generation script (still TBD). Background-remove with `rembg`.

For the procedural style, extend the generator script:

```python
def draw_handdrawn_yourshape(path: Path, color: str = "#FF6B9D") -> None:
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rng = random.Random(seed)
    contour = [...]  # your shape's parametric points
    _draw_wobbly_path(d, contour, color, base_width=4, rng=rng, jitter=2.5)
    img.save(path)
```

Add a row to the `jobs` list in `main()` and re-run.

## Test videos (`test_videos/`)

Drop any video file you want to render. The pipeline accepts `.mp4` / `.mov` / `.mkv`. Examples used during development (all gitignored):

- `sing.mp4` — Mandarin vocal cover, ~53s. Whisper picks up 10 lyric lines cleanly.
- `suki.mp4` — Japanese cover, ~14s. Whisper detects ja with 1.0 confidence; vocals are processed-heavy so we ship a hand-crafted [examples/suki_decision.json](../examples/suki_decision.json).
- `demo.mp4` — instrumental room footage, ~35s. Used to validate the BLIP + heuristic + render path on vocals-free input.

Self-record on phone or trim a CC0 sample. Anything works; vocal tracks give Whisper something to work with, instrumental tracks fall through to the heuristic / hand-written JSON path.
