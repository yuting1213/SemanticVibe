# SemanticVibe

A 5-stage pipeline that paints animated CJK text + hand-drawn-style decoration overlays onto a music video. See [SemanticVibe_Spec.docx](SemanticVibe_Spec.docx) for the product spec and [docs/superpowers/specs/2026-04-25-project-scaffolding-design.md](docs/superpowers/specs/2026-04-25-project-scaffolding-design.md) for the original architecture decisions.

## Pipeline

1. **Preprocess** — librosa (beats + chorus) + cv2 (scene-cut keyframes) + Whisper (lyrics, subprocess-isolated, loudnorm-amplified) + BLIP (visual captions) + MediaPipe (subject pose) → `FeatureSummary`
2. **LLM decide** — Claude / GPT-4o consume only the `FeatureSummary` text → `Decision`. Falls back to a deterministic heuristic when no API key is set.
3. **Asset retrieval** — exact-tag fast path + open-clip ViT-B-32 cosine fallback over a local PNG library
4. **Layout** — occupancy mask (from MediaPipe boxes) + greedy bin-packing with lower-band bias + auto-shrink for oversized text
5. **Render** — Pillow text + decoration tile compositing + chalk-style hero glyph + MoviePy/ffmpeg encode

The LLM never sees video frames; it consumes a text-only `FeatureSummary`. That's the central cost lever.

## Setup (Windows + RTX 3060 12GB)

```powershell
# Install uv (one-off, runs as your user, installs to ~/.local/bin)
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
$env:Path = "C:\Users\User\.local\bin;" + $env:Path

# Install Python 3.12 (mandatory — mediapipe 0.10.33 publishes wheels up to cp312 only)
uv python install 3.12

# Install all deps (PyTorch CUDA 12.1 included via tool.uv.sources)
uv sync --extra dev          # + pytest / ruff / nbstripout for development
uv sync --extra sdxl         # optional: SDXL pre-generation extras

# Copy env template, optionally fill in ANTHROPIC_API_KEY / OPENAI_API_KEY
cp .env.example .env
```

If your project path contains non-ASCII characters on Windows, see the "Environment quirk" section of [CLAUDE.md](CLAUDE.md) — you'll need a junction at an ASCII path so the venv can start.

### Fallback: plain pip / Colab

For environments without `uv` (Colab, lab-shared boxes), [requirements.txt](requirements.txt) is exported from `uv.lock` with `--extra-index-url` for the PyTorch CUDA 12.1 wheels at the top:

```bash
pip install -r requirements.txt
```

The single file is **base only** — no dev tools, no SDXL extras. Add those manually if you need them:

```bash
pip install pytest pytest-cov ruff nbstripout       # dev
pip install diffusers accelerate rembg              # SDXL pre-generation
```

Regenerate via `uv export --no-hashes --no-emit-project -o requirements.txt` after every `uv add/remove/sync --upgrade`. Re-add the `--extra-index-url` line at the top by hand.

## Quickstart (v5: lyrics-driven render)

The current entry point generates everything from the **lyrics** of the
song + the **video itself** — text content, decoration tags, positions,
and animations are all derived, nothing is hard-coded.

```powershell
# Mode 3: full auto (Whisper on the video's embedded audio)
uv run python -m semanticvibe.render `
    --video samples/dance.mp4 `
    --out outputs/auto.mp4 --preview
```

See [PROGRESS.md](PROGRESS.md) for the full architecture summary.

### Lyrics input — three modes

The system picks the highest-priority source automatically:

#### Mode 1: full auto (default)

```powershell
uv run python -m semanticvibe.render --video samples/dance.mp4 --out outputs/out.mp4
```

→ Whisper transcribes the video's embedded audio track.

#### Mode 2: independent audio file

```powershell
uv run python -m semanticvibe.render `
    --video samples/dance.mp4 --audio samples/song.mp3 `
    --mix-audio replace --out outputs/out.mp4
```

→ Whisper runs on `song.mp3` (better isolation than the video's mic
audio); `--mix-audio replace` also splices `song.mp3` into the final
mp4. `--mix-audio overlay` mixes both. Omit the flag to keep the
video's original audio with Whisper having only consulted the
independent file for ASR.

#### Mode 3: hand-edited lyrics (most precise)

When Whisper mis-transcribes (loud BGM, English/Japanese mix, slang),
preview + edit the JSON before render:

```powershell
# 1. Preview Whisper's pick + cache to .cache/lyrics/<sha>.json
uv run python scripts/preview_lyrics.py --audio samples/song.mp3

# 2. Edit samples/auto_lyrics.json — fix typos, adjust timings, optionally add `duration` field

# 3. Render from your edited JSON
uv run python -m semanticvibe.render `
    --video samples/dance.mp4 --audio samples/song.mp3 `
    --lyrics samples/auto_lyrics.json `
    --mix-audio replace --out outputs/edited.mp4
```

### Lyrics JSON schema

Whether produced by Whisper or hand-written, the format is:

```json
[
  {"time": 2.5, "text": "もしもし"},
  {"time": 5.0, "text": "電波", "duration": 0.8}
]
```

`duration` (seconds, optional) controls how long the line stays on
screen. When omitted, the renderer holds it until the next line
starts (capped at 5 s). Validated by Pydantic at load — bad input
raises a clear `ValidationError` instead of crashing 50 lines later.

### Hand-written examples (legacy)

`examples/legacy/` contains the older hand-written Decision JSONs
(`baseline_dream`, `sing_full`, `suki_decision`, ...). They remain
schema-valid and renderable via the legacy CLI:

```powershell
uv run python -m semanticvibe.render_demo `
    --video samples/dance.mp4 `
    --json examples/legacy/baseline_dream_v4.json `
    --output outputs/legacy_demo.mp4 --preview
```

You'll also need fonts in `data/fonts/` and the procedural assets — see [data/README.md](data/README.md).

## Element types (the Decision schema)

`Decision.elements` is a discriminated union on `type`:

- **`text`** — lyric / phrase overlays. Multi-layer outline stack (`outline_layers`) for glow halos; optional `shadow_offset`. Animations: `bounce_in` / `typewriter` / `wiggle` / `draw_in` / `fade`.
- **`decoration`** — asset-library PNG with `count` + `scatter` for confetti, `scatter_zone` for cluster placement, `size_steps` for size variety, `color_tint` for per-copy palette, `wiggle_amp` for ambient hand-drawn drift.
- **`hero_text`** — single huge centred chalk glyph with multi-pass Gaussian blur halos + grain dots + breathing scale.

Full field-by-field reference in [CLAUDE.md](CLAUDE.md) ("Element types" section). All schema definitions live in [src/semanticvibe/schemas/decision.py](src/semanticvibe/schemas/decision.py) — Pydantic models with field docstrings.

## Streamlit UI

```powershell
uv run streamlit run app.py
```

Upload, pick provider / cost mode / style preset / preview, render, then optionally re-render from the last saved Decision JSON without re-running upstream stages.

## Testing

```powershell
uv run pytest                    # all 134 tests
uv run pytest tests/test_render.py::test_render_text_returns_rgba_with_content
uv run ruff check && uv run ruff format
```

Heavy stages (Whisper / BLIP / MediaPipe / CLIP) are validated by end-to-end pipeline runs, not unit tests, since they need model downloads.

## Config

Copy `.env.example` → `.env`, fill in:

- `SEMANTICVIBE_LLM_PROVIDER=claude` (or `openai`) — default `claude`
- `SEMANTICVIBE_COST_MODE=dev` → Haiku 4.5 / gpt-4o-mini (default, ~1/20 prod cost). `prod` → Sonnet 4.6 / gpt-4o.
- `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` — leave blank to use the heuristic fallback.

Style presets in [src/semanticvibe/config.py](src/semanticvibe/config.py): `warm_handdrawn` (default), `soft_pastel`, `bold_pop`, `monochrome_ink`. Each defines a colour palette + vibe descriptor that the LLM and the heuristic both consume.
