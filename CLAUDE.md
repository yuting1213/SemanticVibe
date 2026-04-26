# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Source of truth

- [SemanticVibe_Spec.docx](SemanticVibe_Spec.docx) — v1.0 product/architecture spec (binary; can't be read by tools — ask the user to paste sections rather than guessing).
- [docs/superpowers/specs/2026-04-25-project-scaffolding-design.md](docs/superpowers/specs/2026-04-25-project-scaffolding-design.md) — the authoritative scaffolding design. Read this before any structural change. Section numbers below refer to this doc.

## Project: SemanticVibe

A 5-stage pipeline that adds Chinese-text + decoration overlays to a music video on a 6-week timeline. The five stages, each in its own package under [src/semanticvibe/](src/semanticvibe/):

1. [preprocess/](src/semanticvibe/preprocess/) — Whisper ASR + librosa beats + MediaPipe pose + BLIP-2 captions + keyframe selection → emits `FeatureSummary`.
2. [llm/](src/semanticvibe/llm/) — sends *only* `FeatureSummary` (never the video) to Claude/GPT-4o, gets back a `Decision`. The never-feed-video rule is the central cost lever (spec §4).
3. [assets/](src/semanticvibe/assets/) — CLIP search over a local 200–300-image library.
4. [layout/](src/semanticvibe/layout/) — occupancy map + bin-packing to place text/decorations off subjects' faces and bodies.
5. [render/](src/semanticvibe/render/) — Pillow for text (NOT MoviePy `TextClip`), MoviePy + ffmpeg for compositing.

## Implementation status (as of 2026-04-26)

- **Schemas** — done. [schemas/feature_summary.py](src/semanticvibe/schemas/feature_summary.py) and [schemas/decision.py](src/semanticvibe/schemas/decision.py) are the narrow waist; treat them as a stable API once consumed by any stage.
- **Render (Stage 5)** — Week 1 deliverable, fully wired. [render/text_render.py](src/semanticvibe/render/text_render.py), [render/animations.py](src/semanticvibe/render/animations.py), [render/composite.py](src/semanticvibe/render/composite.py), and [render_demo.py](src/semanticvibe/render_demo.py) work end-to-end given a video file + fonts in `data/fonts/`.
- **LLM client (Stage 2)** — Protocol + Claude/OpenAI class skeletons in place; `decide()` raises `NotImplementedError` until Week 3.
- **Stages 1, 3, 4** — module structure + stable signatures only; bodies are `NotImplementedError` per the design doc's per-week schedule (preprocess Week 2, assets/layout Week 4).
- **Tests** — schemas, animations, config covered. Render tests use a system TrueType font (e.g. Arial on Windows) since `data/fonts/` is gitignored; they `pytest.skip` if no system font is found.

## Locked-in decisions (don't re-litigate without updating the design doc)

- **Python 3.10, hard-pinned.** MediaPipe has no 3.13 wheel. `requires-python = ">=3.10,<3.11"` and [.python-version](.python-version) both enforce this.
- **`uv` + [pyproject.toml](pyproject.toml).** PyTorch CUDA 12.1 index is wired via `[tool.uv.sources]`. Don't switch to pip/poetry.
- **Schemas are the narrow waist.** Once a `Decision` field is consumed by Stage 3+, breaking changes need an explicit version bump and a migration of any committed example JSON.
- **`Decision` is a discriminated union** keyed on `type` (text / decoration). Every element carries a mandatory `reasoning` field — spec §5.2.2 chain-of-thought.
- **Dual LLM provider** ([llm/client.py](src/semanticvibe/llm/client.py)): `LLMClient` Protocol with `ClaudeClient` (Anthropic tool-use for structured JSON) and `OpenAIClient` (`response_format=json_schema`). Selected via `settings.llm_provider`.
- **Cost modes** in [config.py](src/semanticvibe/config.py): `dev` → Haiku 4.5 / gpt-4o-mini (default, ~1/20 prod cost); `prod` → Sonnet 4.6 / gpt-4o (Streamlit demo + final demo).
- **Claude prompt caching is on.** [llm/prompts.py](src/semanticvibe/llm/prompts.py) `SYSTEM_PROMPT` + `FEW_SHOT_EXAMPLES` are the cacheable prefix — every edit invalidates the cache.
- **Pillow renders text, not MoviePy.** `TextClip` mangles CJK; spec §5.5.1 already decided. Pillow's `stroke_width` handles the double-outline requirement.
- **`imageio-ffmpeg` bundles ffmpeg.** Don't assume system ffmpeg on PATH.
- **`faster-whisper`** over `openai-whisper` (4–5× faster, Windows wheel ships).
- **`data/` and `outputs/` are gitignored** (with carve-outs for `data/README.md`). Asset rebuild instructions live in [data/README.md](data/README.md). Don't commit test videos, fonts, or generated frames.
- **Lazy imports for heavy SDKs.** `anthropic`, `openai`, `moviepy` are imported inside the functions that need them so an `import semanticvibe` doesn't pay the cost.

## Common commands

After a one-off `uv` install:

```powershell
# Install uv (one-off, per-user)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install Python 3.10 + sync deps (CUDA 12.1 PyTorch included)
uv python install 3.10
uv sync
uv sync --extra sdxl   # optional: SDXL pre-generation extras
```

Day-to-day:

| Task | Command |
|---|---|
| Run all tests | `uv run pytest` |
| Run a single test | `uv run pytest tests/test_render.py::test_render_text_returns_rgba_with_content` |
| Lint + format | `uv run ruff check && uv run ruff format` |
| Week 1 demo | `uv run python -m semanticvibe.render_demo --video data/test_videos/sample_30s.mp4 --json examples/hand_written_decision.json --output outputs/week1_demo.mp4` |
| Streamlit (placeholder until Week 5) | `uv run streamlit run app.py` |

## When the user asks for work

- Structural / scaffolding changes → re-read the design doc first; don't reinvent the layout.
- Schema changes → high-impact, ripple to every downstream stage. Confirm with the user; update both `schemas/` files, [examples/hand_written_decision.json](examples/hand_written_decision.json), and the design doc together.
- "Implement Stage X" → its package already exists with stable signatures; fill in the `NotImplementedError` bodies and mirror the design doc's component list. Don't introduce new top-level modules.
- Anything quoting spec section numbers → the `.docx` is binary; ask the user to paste the relevant section.
- Adding dependencies → must work on Python 3.10 + Windows + CUDA 12.1. Verify Windows wheel availability before adding to [pyproject.toml](pyproject.toml).
