# `data/` — gitignored runtime assets

This whole directory is excluded from git (see [.gitignore](../.gitignore)). Use this README to rebuild a working environment on a fresh checkout.

## Layout

```
data/
├── README.md                  # this file (the only thing committed)
├── fonts/
│   ├── KleeOne-Regular.ttf
│   ├── KleeOne-SemiBold.ttf
│   └── NotoSansTC-Regular.ttf  # CJK fallback for missing glyphs
├── assets_lib/
│   ├── metadata.json           # { filename, tags, license, source }[]
│   └── *.png                    # 200–300 decoration stickers
└── test_videos/
    └── sample_30s.mp4
```

## Fonts

Both fonts ship under the SIL Open Font License — vendor them locally rather than depending on system fonts.

- **Klee One** — primary CJK display font. Spec mandates this. Download: https://fonts.google.com/specimen/Klee+One → unzip → drop `KleeOne-Regular.ttf` and `KleeOne-SemiBold.ttf` into `data/fonts/`.
- **Noto Sans TC** — fallback for any glyph Klee One doesn't cover. Download: https://fonts.google.com/noto/specimen/Noto+Sans+TC.

## Decoration assets (`assets_lib/`)

200–300 PNGs with transparent backgrounds. Build via, in order of preference:

1. **CC0 sources** — OpenClipart, Pixabay (CC0 filter), the Noun Project (paid, but check the public-domain set).
2. **Self-drawn** — keep working files outside `data/` and export PNGs into `assets_lib/`.
3. **SDXL pre-generation** (optional) — `uv sync --extra sdxl` then run a generation script (lands in Week 4 alongside the asset library work). Background-remove with `rembg`.

Every entry needs a record in `metadata.json`:

```json
[
  {"filename": "sparkle_001.png", "tags": ["sparkle", "celebration"], "license": "CC0", "source": "openclipart.org/123"},
  {"filename": "heart_pink.png",  "tags": ["heart", "love"],          "license": "self-drawn", "source": "internal"}
]
```

The `tags` field is what `Decision.elements[].asset_tag` matches against; CLIP fills in any gaps in Week 4.

## Test videos (`test_videos/`)

For Week 1 you need a single 30-second clip with audible vocals — `sample_30s.mp4`. Self-record on phone, or trim a CC0 sample. Anything works as long as it has a clean audio track.
