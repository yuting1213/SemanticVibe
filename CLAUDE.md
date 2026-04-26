# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Source of truth

- [SemanticVibe_Spec.docx](SemanticVibe_Spec.docx) — v1.0 product/architecture spec (binary; can't be read by tools — ask the user to paste sections rather than guessing).
- [docs/superpowers/specs/2026-04-25-project-scaffolding-design.md](docs/superpowers/specs/2026-04-25-project-scaffolding-design.md) — the original scaffolding design doc. Frozen historical snapshot; the codebase has since evolved (see "What changed since the design doc" below).

## Project: SemanticVibe

A 5-stage pipeline that adds CJK text + decoration overlays onto a music video. The five stages, each in its own package under [src/semanticvibe/](src/semanticvibe/):

1. [preprocess/](src/semanticvibe/preprocess/) — librosa beats + scene-cut keyframes + Whisper ASR (subprocess-isolated) + BLIP captions + MediaPipe pose → emits `FeatureSummary`.
2. [llm/](src/semanticvibe/llm/) — sends *only* `FeatureSummary` (never the video) to Claude/GPT-4o, gets back a `Decision`. Falls back to a deterministic heuristic when no API key is set.
3. [assets/](src/semanticvibe/assets/) — exact-tag fast path + open-clip ViT-B-32 cosine fallback over a local PNG library.
4. [layout/](src/semanticvibe/layout/) — occupancy map (from MediaPipe boxes) + greedy bin-packing with lower-band bias to resolve `auto` anchors.
5. [render/](src/semanticvibe/render/) — Pillow text + decoration + hero compositing, MoviePy + ffmpeg encoding.

The LLM never sees frames; it consumes `FeatureSummary` text only. That is the central cost lever.

## Implementation status (as of 2026-04-26)

All 5 stages live and end-to-end tested on real videos (`outputs/sing_overlay.mp4`, `outputs/output_v2.mp4`, etc.). Test count: **65 passing**.

- **Schemas** — [schemas/feature_summary.py](src/semanticvibe/schemas/feature_summary.py) and [schemas/decision.py](src/semanticvibe/schemas/decision.py). The Decision discriminated union now has **three** variants (`text` / `decoration` / `hero_text`) — see "Element types" below.
- **Stage 1 (preprocess)** — [preprocess/pipeline.py](src/semanticvibe/preprocess/pipeline.py) orchestrates librosa → keyframes → Whisper → BLIP. Whisper runs in a subprocess and reads a loudnorm-amplified wav (see "Whisper gotchas" below).
- **Stage 2 (LLM decide)** — [llm/client.py](src/semanticvibe/llm/client.py) ships ClaudeClient (tool-use) + OpenAIClient (`response_format=json_schema`). [llm/decide.py](src/semanticvibe/llm/decide.py) wraps with tenacity retry on `ValidationError`; falls back to [llm/heuristic.py](src/semanticvibe/llm/heuristic.py) when no API key is set.
- **Stage 3 (assets)** — [assets/clip_search.py](src/semanticvibe/assets/clip_search.py): exact-tag first, CLIP fallback with embeddings cached at `data/assets_lib/_clip_embeddings.npy`.
- **Stage 4 (layout)** — [layout/placement.py](src/semanticvibe/layout/placement.py) shrinks oversized text via `fit_to_canvas` then calls bin-packing against the MediaPipe-derived occupancy mask.
- **Stage 5 (render)** — [render/composite.py](src/semanticvibe/render/composite.py) orchestrates per-frame compositing of text tiles + decoration tiles (with scatter/cluster placement) + hero glyphs ([render/hero_text.py](src/semanticvibe/render/hero_text.py)).
- **Pipeline orchestrator** — [pipeline.py](src/semanticvibe/pipeline.py) `run()` runs all 5 stages; `render_from_intermediate()` re-renders from a saved Decision JSON without touching upstream stages.
- **Streamlit UI** — [app.py](app.py): upload + provider/style/preview sidebar + re-render-from-last-Decision shortcut.
- **Procedural assets** — [scripts/generate_placeholder_assets.py](scripts/generate_placeholder_assets.py) draws 9 outline-only hand-drawn-jitter PNGs (heart / mini_heart / sparkle / star / dot / burst / arrow / fire / exclaim) into `data/assets_lib/` plus the metadata index. Re-run anytime to regenerate.

## Element types (the Decision schema)

`Decision.elements` is a discriminated union on `type`:

### `TextElement`
Standard lyric / phrase overlay. Rendered with Pillow; outline stack drawn outermost-first.
- `outline_color` + `outline_width` — primary outline (the existing field).
- `outline_layers: list[OutlineLayer]` — extra strokes outside the primary one. Stack from outermost (drawn first) inward. The cute look uses `[{color: "#FFFFFF", width: 4}]` to add a white halo.
- `shadow_offset: (dx, dy) | None` — drop shadow (50% black) at the bottom of the stack.
- `animation: bounce_in | typewriter | wiggle | draw_in | fade`
- `rotation_jitter: float` — degrees of random tilt baked into the tile.

### `DecorationElement`
Asset-library PNG with positioning + replication.
- `asset_tag` — resolved exactly first, then CLIP fallback.
- `near_text_id: int | None` — cluster against that text element's resolved bbox.
- `count + scatter` — emit N copies. `scatter=True` spreads across the frame at deterministic pseudo-random positions; `scatter=False` stacks at the resolved anchor with small per-copy nudge.
- `scatter_zone: (x1, y1, x2, y2) | None` — when set, scatter copies land inside this canvas-pixel bbox. Author-zoned scatters skip the PERSON_BBOX push (the author is responsible for avoiding subjects).
- `size_steps: list[int] | None` — per-copy `base_size` cycle (e.g. `[200, 80, 80, 80, 40, 40, 40, 40]` for a 1-large + 3-medium + 4-small cluster).
- `color_tint: list[str]` — per-copy palette tints applied to one PNG (so a single `mini_heart.png` shows up in 6 colours without shipping 6 files).
- `base_size: int | None` — override the asset's natural pixel size (single anchor mode).
- `wiggle_amp: float` — steady-state ±N px per-copy sin-modulated drift. Reads as hand-drawn instability without re-rendering tiles.

### `HeroTextElement` (added 2026-04-26)
Single huge centred glyph, drawn separately from `TextElement` because the look is calmer.
- `pos: "center_upper" | "center" | "center_lower" | "upper_left" | "upper_right" | (x, y)`
- `style: "chalk" | "outline"` — chalk uses 3-pass Gaussian blur halos + ~80 grain dots/strokes for chalk-dust feel.
- `breathing: bool` — ±3% scale oscillation on a ~3.5s sin wave throughout the visible window.
- `halo_color: str` — colour of the soft outer blur (chalk only).
- Animation envelope is a slow fade (1.2s in / 1.0s out), not the playful TextElement set.
- See [render/hero_text.py](src/semanticvibe/render/hero_text.py).

## Render-time invariants

- **`yuv420p` + even canvas dims** — write_videofile passes `-pix_fmt yuv420p`. Source clips with odd width (e.g. 1080×1920 portrait → 405×720 preview) are rounded down to the nearest even pixel before render. Without this, libx264 falls back to yuv444p which Windows Media Player and most consumer players refuse to open.
- **`fit_to_canvas`** — every TextElement is iteratively shrunk before measuring/placing so it fits canvas minus margins. Outline width doesn't scale with font size, so the shrink loops up to 8 times with a 0.95 safety factor. Floor is `min_size=24`.
- **`PERSON_BBOX_FRAC = (1/3, 0, 2/3, 1)`** in [render/composite.py](src/semanticvibe/render/composite.py) — when a scatter has no explicit `scatter_zone`, placements that overlap the central vertical strip get pushed left or right (whichever is closer). Stage 4 MediaPipe layout supersedes this when it runs.

## Whisper gotchas

- **Subprocess isolation.** faster-whisper bundles its own cuDNN through ctranslate2; PyTorch (used by BLIP / Open-CLIP) bundles a different cuDNN build. Loading both in-process on Windows produces "Could not load symbol cudnnGetLibConfig" / heap corruption (`0xC0000409` STACK_BUFFER_OVERRUN). [preprocess/whisper_asr.py](src/semanticvibe/preprocess/whisper_asr.py) runs Whisper in a `subprocess.run` with a small worker source that emits JSON on stdout. Don't refactor back to in-process loading without solving the symbol clash another way.
- **Stage order in [preprocess/pipeline.py](src/semanticvibe/preprocess/pipeline.py)** is load-order-sensitive: librosa (numba JIT) before any GPU model. Reordering reintroduces the same heap-corruption class of bug.
- **Loudnorm pre-amplification.** [preprocess/librosa_beats.py](src/semanticvibe/preprocess/librosa_beats.py) `extract_wav` applies ffmpeg's EBU R128 loudnorm to ~-16 LUFS. Quiet phone recordings (mean -35 dB and below) otherwise fall under Whisper's speech threshold and return zero segments.
- **VAD off by default.** Silero VAD is tuned for speech and aggressively drops speech that overlaps music in the same band — typical for music videos. `transcribe(vad=False)` is the default; flip to True for podcast / interview material with long silences.
- **UTF-8 stdout in the worker.** The subprocess sets `PYTHONIOENCODING=utf-8` + reconfigures stdout, otherwise Whisper's Simplified-Chinese characters can't be encoded by cp950 on Traditional-Chinese Windows and the worker exits with a UnicodeEncodeError that looks like "Whisper found nothing".

## Other locked-in decisions

- **Python 3.12, hard-pinned.** MediaPipe 0.10.33 publishes wheels for cp39–cp312 only — no Python 3.13+ wheel yet. Bump alongside the next mediapipe release that ships 3.13 wheels.
- **`uv` + [pyproject.toml](pyproject.toml).** PyTorch CUDA 12.1 index is wired via `[tool.uv.sources]`. Don't switch to pip/poetry.
- **`Decision` is discriminated** on `type` — `text` / `decoration` / `hero_text`. Every element carries a mandatory `reasoning` field (chain-of-thought for the LLM path; placeholder string for hand-written JSONs is fine).
- **Cost modes** in [config.py](src/semanticvibe/config.py): `dev` → Haiku 4.5 / gpt-4o-mini (default); `prod` → Sonnet 4.6 / gpt-4o.
- **Claude prompt caching is on.** [llm/prompts.py](src/semanticvibe/llm/prompts.py) `SYSTEM_PROMPT` + `FEW_SHOT_EXAMPLES` are the cacheable prefix.
- **Pillow renders text, not MoviePy.** `TextClip` mangles CJK; spec §5.5.1 already decided.
- **`imageio-ffmpeg` bundles ffmpeg.** Don't assume system ffmpeg on PATH.
- **`faster-whisper`** over `openai-whisper` (4–5× faster, Windows wheel).
- **`data/` and `outputs/` are gitignored** (with carve-outs for `data/README.md` and `data/assets_lib/metadata.json`). Asset rebuild instructions in [data/README.md](data/README.md). PNG assets are reproducible via the generator script.
- **mediapipe 0.10.33** uses `mp.tasks.vision.PoseLandmarker` (the new tasks API), NOT the dropped `mp.solutions` API.
- **moviepy 2.x** API (`from moviepy import …`, `with_audio`, `resized`). 1.x patterns (`from moviepy.editor`, `set_audio`, `resize`) are gone.
- **Lazy imports for heavy SDKs** — `anthropic`, `openai`, `moviepy`, `transformers`, `librosa` import inside the functions that need them.

## Environment quirk: project path contains Chinese characters

The project sits at `C:\Users\User\Desktop\AI人文\`. On Traditional-Chinese Windows the system locale is **cp950**, and CPython's `site.py` reads `.pth` files with `encoding="locale"` — so the UTF-8-encoded path stored in `_editable_impl_semanticvibe.pth` cannot be decoded and the venv refuses to start with `UnicodeDecodeError: 'cp950' codec can't decode byte 0x96`. (Reproduced on Python 3.10 and 3.12 alike.) `PYTHONUTF8=1` doesn't help because uv invokes its internal Python with `-I`, which strips `PYTHON*` env vars.

**Workaround in place**: a directory junction `C:\sv → C:\Users\User\Desktop\AI人文`, plus `_editable_impl_semanticvibe.pth` rewritten to `C:\sv\src`. Operate from `C:\sv`.

```powershell
Set-Location C:\sv
uv run pytest
```

If you ever wipe `.venv` or run `uv sync --reinstall`, uv will rewrite the `.pth` back to the AI人文 path and break the venv. Re-apply:

```powershell
[System.IO.File]::WriteAllText("C:\sv\.venv\Lib\site-packages\_editable_impl_semanticvibe.pth", "C:\sv\src`r`n", [System.Text.Encoding]::ASCII)
```

## Common commands (from `C:\sv`)

| Task | Command |
|---|---|
| Tests (all 65) | `uv run pytest` |
| Single test | `uv run pytest tests/test_render.py::test_render_text_returns_rgba_with_content` |
| Lint + format | `uv run ruff check && uv run ruff format` |
| Render from JSON (Stage 5 only) | `uv run python -m semanticvibe.render_demo --video data/test_videos/sing.mp4 --json examples/sing_full.json --output outputs/sing.mp4 --preview` |
| Full pipeline (all 5 stages) | `uv run python -m semanticvibe.cli --video data/test_videos/sing.mp4 --output outputs/sing_auto.mp4 --style warm_handdrawn --language zh --preview --keep-intermediates outputs/intermediates` |
| Re-render after editing Decision JSON | Same `render_demo` command, point at the edited JSON |
| Streamlit | `uv run streamlit run app.py` |
| Regenerate procedural assets | `uv run python scripts/generate_placeholder_assets.py` |

## Example JSONs (start here when authoring a Decision)

| File | What it shows |
|---|---|
| [examples/hand_written_decision.json](examples/hand_written_decision.json) | Minimal 3-element example — original Week-1 reference |
| [examples/sing_full.json](examples/sing_full.json) | Full 10-line lyric overlay for sing.mp4, mixed animations + per-line decorations + heart confetti |
| [examples/suki_decision.json](examples/suki_decision.json) | Bold-pop palette demo with multi-outline glow + scattered hearts + impact burst behind 「好き」 |
| [examples/demo_chinese.json](examples/demo_chinese.json) | Warm hand-drawn manual JSON for a vocals-free clip |
| [examples/baseline_dream.json](examples/baseline_dream.json) | Hand-drawn aesthetic showcase: chalk hero 「夢」 + clustered outlined hearts in upper-left zone, pink/red/white only |

## What changed since the design doc

The design doc ([docs/superpowers/specs/2026-04-25-project-scaffolding-design.md](docs/superpowers/specs/2026-04-25-project-scaffolding-design.md)) was a frozen snapshot; live state has since evolved:

- **Schema:** `TextElement` gained `outline_layers` + `shadow_offset`; `DecorationElement` gained `count` / `scatter` / `scatter_zone` / `size_steps` / `color_tint` / `base_size` / `wiggle_amp`; new `HeroTextElement` variant.
- **Stage 1:** Whisper runs in a subprocess on a loudnorm wav (not the raw mp4); VAD off by default; mediapipe uses tasks API.
- **Stage 5:** `fit_to_canvas` auto-shrink, yuv420p + even-dim enforcement, `PERSON_BBOX_FRAC` simple avoidance, ambient wiggle on decorations, chalk hero rendering.
- **Heuristic:** emits multi-outline + heart confetti by default so the no-API-key path looks decent.

The design doc deliberately defines a stable baseline; treat it as historical context, not a spec to chase.

## When the user asks for work

- **Aesthetic / styling tweaks** → look at [examples/baseline_dream.json](examples/baseline_dream.json) for the current hand-drawn target. The user's reference clips were "hand-drawn picturebook" not "geometric stickers".
- **Schema changes** → high-impact; ripple to schemas, examples, heuristic, render. Add tests under `tests/test_schemas.py`. Update the example JSONs and CLAUDE.md's "Element types" section together.
- **Stage X internals** → its package already exists with stable signatures; mutate within. Don't introduce new top-level modules without the user agreeing.
- **Anything quoting spec section numbers** → the `.docx` is binary; ask the user to paste the relevant section.
- **Adding dependencies** → must work on Python 3.12 + Windows + CUDA 12.1. Check Windows wheel availability before committing.
- **Reading frames to verify visuals** → use `clip.save_frame('outputs/probe.png', t=N)` then `Read` the PNG. The user can't see what the rendered video actually looks like unless you check.
