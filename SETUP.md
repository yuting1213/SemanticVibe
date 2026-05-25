# SemanticVibe — Setup

One-line bootstrap for fresh clones.

## Prerequisites (install once)

- **Python 3.12** (NOT 3.13+ — MediaPipe wheel limitation)
- **[uv](https://github.com/astral-sh/uv)** for dependency management
- **[Ollama](https://ollama.com)** for local LLM/VLM (qwen2.5:7b + qwen2.5vl:7b)
- **Node.js 18+** if you plan to use the Hyperframes renderer (`--renderer hyperframes`)
- **NVIDIA GPU with ≥6 GB VRAM** for CUDA path; CPU works but slow

## Quick start

```powershell
# Windows
git clone https://github.com/yuting1213/SemanticVibe.git
cd SemanticVibe
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

> The `-ExecutionPolicy Bypass` flag lets the script run without
> permanently changing your system policy. Alternatively, run
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
> once and then you can use `.\scripts\bootstrap.ps1` directly.

```bash
# macOS / Linux
git clone https://github.com/yuting1213/SemanticVibe.git
cd SemanticVibe
bash scripts/bootstrap.sh
```

The bootstrap script:
1. Verifies `uv` and `ollama` are installed
2. Runs `uv sync` to install Python deps
3. (Windows only) Auto-fixes the `.pth` editable-install file for cp950 locales
4. Pulls `qwen2.5:7b` (4.7 GB) and `qwen2.5vl:7b` (5.9 GB) via `ollama pull`

After bootstrap, you still need:
- A test video at `data/test_videos/<your_file>.mp4`
- Sticker assets at `assets/stickers/<tag>/*.png` (your team's pack, or run `scripts/generate_placeholder_assets.py` for 9 procedural placeholders)
- Rebuild the asset index: `uv run python scripts/build_index.py`

Then render:
```powershell
uv run python -m semanticvibe.render `
    --video data\test_videos\<file> `
    --out outputs\test.mp4 --preview --vlm-gestures `
    --renderer hyperframes --style baseline_kenpa --subtitle-style outlined
```

## Windows + non-ASCII repo path (cp950 issue)

If you clone into a path containing CJK characters (e.g. `C:\Users\你\Desktop\...`):

- Traditional-Chinese Windows uses cp950 as the system locale
- CPython's `site.py` reads `.pth` files with `encoding="locale"`
- A UTF-8-encoded path with CJK chars triggers `UnicodeDecodeError: 'cp950' codec can't decode byte 0x96`

`bootstrap.ps1` handles this automatically by:
1. Detecting non-ASCII chars in the repo path
2. Creating a `C:\sv` junction pointing to your repo
3. Re-running setup from `C:\sv` (which is ASCII-safe)

If you wipe `.venv` or run `uv sync --reinstall` later, the `.pth` gets clobbered. Re-apply with:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fix_pth.ps1
```

## Manual setup (if bootstrap isn't enough)

```powershell
# 1. Python deps
uv sync

# 2. (Windows-cp950 only) Repair .pth
.\scripts\fix_pth.ps1

# 3. LLM/VLM models
ollama pull qwen2.5:7b      # text alignment (lyric -> tag)
ollama pull qwen2.5vl:7b    # gesture VLM (frame -> gesture)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `UnicodeDecodeError: 'cp950' codec can't decode...` | Run `.\scripts\fix_pth.ps1` |
| `ModuleNotFoundError: No module named 'semanticvibe'` | Run `.\scripts\fix_pth.ps1` (the `.pth` file was clobbered by `uv sync`) |
| `Ollama unreachable at http://localhost:11434` | Start Ollama: `ollama serve` (or use the desktop app) |
| `model 'qwen2.5:7b' not found` | `ollama pull qwen2.5:7b` |
| `[asset_retrieval] assets/index.json missing` | `uv run python scripts/build_index.py` |
| `Could not load symbol cudnnGetLibConfig` (Whisper subprocess crash) | Already handled by `preprocess/whisper_asr.py` running Whisper in a subprocess. If you see this in main process, you imported faster-whisper directly — don't |
| Hyperframes renderer: `puppeteer Chromium download stalled` | First run downloads ~150 MB Chromium. Subsequent runs cache it. |

## What's NOT in the repo

These are `.gitignore`d — provide your own:
- `data/test_videos/*.mp4` — your video sources
- `assets/stickers/<tag>/*.png` — your sticker library (3-5 PNGs per tag minimum)
- `assets/index.json` — auto-built from stickers
- `data/fonts/*.ttf` — Pillow-renderable CJK fonts
- `outputs/` — render outputs
- `.cache/` — alignment + VLM caches

See [CLAUDE.md](CLAUDE.md) for project deep-dive and stage-by-stage spec.
