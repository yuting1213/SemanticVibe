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

## Implementation status (as of 2026-04-26 — all 5 stages live)

- **Schemas** — done. [schemas/feature_summary.py](src/semanticvibe/schemas/feature_summary.py) and [schemas/decision.py](src/semanticvibe/schemas/decision.py) are the narrow waist; treat them as a stable API once consumed by any stage.
- **Stage 1 (preprocess)** — done. [preprocess/pipeline.py](src/semanticvibe/preprocess/pipeline.py) orchestrates librosa → keyframes → Whisper (subprocess-isolated) → BLIP captions. Per-model files: [whisper_asr.py](src/semanticvibe/preprocess/whisper_asr.py), [librosa_beats.py](src/semanticvibe/preprocess/librosa_beats.py), [keyframes.py](src/semanticvibe/preprocess/keyframes.py), [blip2_caption.py](src/semanticvibe/preprocess/blip2_caption.py), [mediapipe_pose.py](src/semanticvibe/preprocess/mediapipe_pose.py).
- **Stage 2 (LLM decide)** — done. [llm/client.py](src/semanticvibe/llm/client.py) ships ClaudeClient (tool-use) and OpenAIClient (`response_format=json_schema`). [llm/decide.py](src/semanticvibe/llm/decide.py) wraps with tenacity retry on `ValidationError` and falls back to [llm/heuristic.py](src/semanticvibe/llm/heuristic.py) when no API key is set — so the pipeline runs offline.
- **Stage 3 (assets)** — done. [assets/clip_search.py](src/semanticvibe/assets/clip_search.py): exact-tag fast path + open-clip ViT-B-32 cosine fallback (embeddings cached at `data/assets_lib/_clip_embeddings.npy`).
- **Stage 4 (layout)** — done. [layout/placement.py](src/semanticvibe/layout/placement.py) uses MediaPipe subject boxes → [occupancy.py](src/semanticvibe/layout/occupancy.py) mask → [bin_packing.py](src/semanticvibe/layout/bin_packing.py) greedy scan with lower-band bias to resolve `auto` anchors.
- **Stage 5 (render)** — done. Pillow text + decoration tile compositing via [render/composite.py](src/semanticvibe/render/composite.py); decoration alpha follows the fade envelope, `near_text_id` snugs sticker to upper-right of the linked title.
- **Pipeline orchestrator** — [pipeline.py](src/semanticvibe/pipeline.py) `run()` runs all 5 stages end-to-end; `render_from_intermediate()` re-renders from a saved Decision JSON. CLI: [cli.py](src/semanticvibe/cli.py) (`semanticvibe-render-demo` is the legacy single-stage CLI, `semanticvibe` is the full one).
- **Streamlit UI** — [app.py](app.py) provides upload + style/provider sidebar + re-render-from-Decision button. `uv run streamlit run app.py`.
- **Tests** — 51 passing (`uv run pytest`). Coverage: schemas (round-trip + validators), animations (envelope math), config (cost modes / presets), render-tile (uses a system TrueType as the font stand-in), heuristic Decision generator, layout (occupancy + bin-packing). Heavy stages (Whisper / BLIP / MediaPipe / CLIP) are validated by the end-to-end pipeline run, not unit tests, since they need model downloads.
- **Environment** — `uv 0.11.7`, Python 3.10.20 in `.venv/`, PyTorch 2.5.1+cu121 active on the RTX 3060. moviepy 2.2.1, mediapipe 0.10.33 (uses `mp.tasks.vision.PoseLandmarker`, NOT the dropped `mp.solutions` API). Whisper runs in a subprocess to dodge the cuDNN symbol clash with PyTorch — see the file's docstring.

## Subprocess isolation for Whisper (Windows-specific gotcha)

faster-whisper bundles its own cuDNN through ctranslate2; PyTorch (used by BLIP and Open-CLIP downstream) bundles a different cuDNN build. Loading both into the same Python process on Windows produces "Could not load symbol cudnnGetLibConfig" / heap corruption (`0xC0000409` STACK_BUFFER_OVERRUN). [preprocess/whisper_asr.py](src/semanticvibe/preprocess/whisper_asr.py) runs Whisper in a `subprocess.run` with a small worker source that emits JSON on stdout. Don't refactor that back into in-process loading without solving the cuDNN conflict another way.

Stage order in [preprocess/pipeline.py](src/semanticvibe/preprocess/pipeline.py) is also load-order-sensitive: librosa (numba JIT) before any GPU model. Reordering will reintroduce the same heap-corruption class of bug.

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

## Environment quirk: project path contains Chinese characters

The project sits at `C:\Users\User\Desktop\AI人文\`. On Traditional-Chinese Windows the system locale is **cp950**, and Python 3.10's `site.py` reads `.pth` files with `encoding="locale"` — so the UTF-8-encoded path stored in `_editable_impl_semanticvibe.pth` (the editable install marker) cannot be decoded and the venv refuses to start with `UnicodeDecodeError: 'cp950' codec can't decode byte 0x96`. `PYTHONUTF8=1` doesn't help because uv invokes its internal Python with `-I`, which strips `PYTHON*` env vars.

**Workaround in place**: a directory junction `C:\sv → C:\Users\User\Desktop\AI人文`, plus the `.pth` rewritten to `C:\sv\src`. Everything works as long as you operate from `C:\sv`.

```powershell
# Daily workflow — always cd to the junction, not the original Chinese path:
Set-Location C:\sv
uv run pytest
uv run python -m semanticvibe.render_demo ...
```

If you ever wipe `.venv` or run `uv sync --reinstall`, uv will rewrite `_editable_impl_semanticvibe.pth` back to the AI人文 path and break the venv. Re-apply the fix:

```powershell
[System.IO.File]::WriteAllText("C:\sv\.venv\Lib\site-packages\_editable_impl_semanticvibe.pth", "C:\sv\src`r`n", [System.Text.Encoding]::ASCII)
```

If the user changes their mind and wants the project at an ASCII path, the cleaner fix is to move the project; the junction is an expedient.

## Common commands

After a one-off `uv` install:

```powershell
# Install uv (one-off, per-user) — adds to C:\Users\User\.local\bin
powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
$env:Path = "C:\Users\User\.local\bin;" + $env:Path

# Install Python 3.10 + sync deps (CUDA 12.1 PyTorch included)
uv python install 3.10
Set-Location C:\sv         # IMPORTANT: use the junction, not C:\Users\User\Desktop\AI人文
uv sync --extra dev
uv sync --extra sdxl       # optional: SDXL pre-generation extras
```

Day-to-day (from `C:\sv`):

| Task | Command |
|---|---|
| Run all tests | `uv run pytest` |
| Run a single test | `uv run pytest tests/test_render.py::test_render_text_returns_rgba_with_content` |
| Lint + format | `uv run ruff check && uv run ruff format` |
| Week 1 demo | `uv run python -m semanticvibe.render_demo --video data/test_videos/sample_30s.mp4 --json examples/hand_written_decision.json --output outputs/week1_demo.mp4` |
| Streamlit | `uv run streamlit run app.py` |
| Full pipeline (all 5 stages) | `uv run python -m semanticvibe.cli --video ... --output ... --style warm_handdrawn --preview --keep-intermediates outputs/intermediates` |
| Re-render from saved Decision | `uv run python -m semanticvibe.render_demo --video ... --json outputs/intermediates/decision_resolved.json --output ...` |

## When the user asks for work

- Structural / scaffolding changes → re-read the design doc first; don't reinvent the layout.
- Schema changes → high-impact, ripple to every downstream stage. Confirm with the user; update both `schemas/` files, [examples/hand_written_decision.json](examples/hand_written_decision.json), and the design doc together.
- "Implement Stage X" → its package already exists with stable signatures; fill in the `NotImplementedError` bodies and mirror the design doc's component list. Don't introduce new top-level modules.
- Anything quoting spec section numbers → the `.docx` is binary; ask the user to paste the relevant section.
- Adding dependencies → must work on Python 3.10 + Windows + CUDA 12.1. Verify Windows wheel availability before adding to [pyproject.toml](pyproject.toml).
