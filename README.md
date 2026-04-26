# SemanticVibe

A 5-stage pipeline that adds semantic-vibe overlays (Chinese text + decoration stickers + animations) onto a music video. See [SemanticVibe_Spec.docx](SemanticVibe_Spec.docx) for the product spec and [docs/superpowers/specs/2026-04-25-project-scaffolding-design.md](docs/superpowers/specs/2026-04-25-project-scaffolding-design.md) for the architecture decisions.

## Pipeline

1. **Preprocess** — Whisper + librosa + MediaPipe + BLIP-2 → `FeatureSummary`
2. **LLM decide** — Claude / GPT-4o (text-only) → `Decision`
3. **Asset retrieval** — CLIP search over local library
4. **Layout** — bin-packing + occupancy avoidance
5. **Render** — Pillow text + MoviePy compositing

The LLM never sees video frames; it only consumes the `FeatureSummary` text contract. This is the central cost-optimisation lever.

## Setup (Windows + RTX 3060 12GB)

```powershell
# Install uv (one-off, runs as your user, installs to ~/.local/bin)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install Python 3.10 (mandatory — MediaPipe has no 3.13 wheel)
uv python install 3.10

# Install all deps (PyTorch CUDA 12.1 included via tool.uv.sources)
uv sync

# Optional: SDXL extras for asset generation
uv sync --extra sdxl

# Copy env template and fill in API keys
cp .env.example .env
```

## Week 1 deliverable

```bash
uv run python -m semanticvibe.render_demo \
    --video data/test_videos/sample_30s.mp4 \
    --json examples/hand_written_decision.json \
    --output outputs/week1_demo.mp4
```

You need to supply your own `data/test_videos/sample_30s.mp4` and the fonts in `data/fonts/`. See [data/README.md](data/README.md).

## Testing

```bash
uv run pytest                    # all
uv run pytest tests/test_render.py::test_render_from_json   # single test
uv run ruff check && uv run ruff format
```

## Config

Cost mode is controlled via `.env`:

- `SEMANTICVIBE_COST_MODE=dev` → Haiku 4.5 / gpt-4o-mini (default, ~1/20 prod cost)
- `SEMANTICVIBE_COST_MODE=prod` → Sonnet 4.6 / gpt-4o (Streamlit demo & final demo)
