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

# Install Python 3.10 (mandatory — MediaPipe has no 3.13 wheel)
uv python install 3.10

# Install all deps (PyTorch CUDA 12.1 included via tool.uv.sources)
uv sync --extra dev

# Optional: SDXL extras for asset generation
uv sync --extra sdxl

# Copy env template, optionally fill in ANTHROPIC_API_KEY / OPENAI_API_KEY
cp .env.example .env
```

If your project path contains non-ASCII characters on Windows, see the "Environment quirk" section of [CLAUDE.md](CLAUDE.md) — you'll need a junction at an ASCII path so the venv can start.

### Fallback: plain pip / Colab

For environments without `uv`, [requirements.txt](requirements.txt) is exported from `uv.lock` with `--extra-index-url` for the PyTorch CUDA 12.1 wheels:

```bash
pip install -r requirements.txt          # base
pip install -r requirements-dev.txt      # + pytest, ruff, nbstripout
pip install -r requirements-sdxl.txt     # + diffusers, accelerate, rembg
```

Regenerate via `uv export --no-hashes --no-emit-project --extra <group> -o requirements-<group>.txt` after every `uv add/remove/sync --upgrade`. Re-add the `--extra-index-url` line at the top.

## Quickstart

The project ships **9 procedural hand-drawn decoration assets** (heart / mini_heart / sparkle / star / dot / burst / arrow / fire / exclaim) and **5 reference Decision JSONs**. After `uv sync`, drop a video into `data/test_videos/` and pick an example to render:

```powershell
# Stage 5 only — render from a hand-written Decision JSON (fast):
uv run python -m semanticvibe.render_demo `
    --video data/test_videos/your_clip.mp4 `
    --json examples/sing_full.json `
    --output outputs/yours.mp4 `
    --preview                                # downscale to 720p

# Full pipeline — Stages 1–5 end-to-end (Whisper + BLIP + MediaPipe + LLM):
uv run python -m semanticvibe.cli `
    --video data/test_videos/your_clip.mp4 `
    --output outputs/auto.mp4 `
    --style warm_handdrawn `
    --language zh `
    --preview `
    --keep-intermediates outputs/intermediates
```

Without an API key, the full pipeline emits a deterministic heuristic Decision (title + chorus + outro lines, with white-halo outlines + scattered heart confetti) so the offline path still produces something watchable.

### Example JSONs

| File | What it shows |
|---|---|
| [examples/hand_written_decision.json](examples/hand_written_decision.json) | Minimal 3-element example |
| [examples/sing_full.json](examples/sing_full.json) | Full 10-line lyric overlay, mixed animations + per-line accents + heart confetti |
| [examples/suki_decision.json](examples/suki_decision.json) | Bold-pop palette with multi-outline glow + scatter + impact bursts |
| [examples/demo_chinese.json](examples/demo_chinese.json) | Warm hand-drawn manual JSON for a vocals-free clip |
| [examples/baseline_dream.json](examples/baseline_dream.json) | Hand-drawn aesthetic showcase: chalk hero 「夢」 + clustered outlined hearts (pink/red/white only) |

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
uv run pytest                    # all 65 tests
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
