#!/usr/bin/env bash
# bootstrap.sh — One-command setup for fresh clones on macOS / Linux.
#
# Walks through:
#   1. Verify prerequisites (uv, ollama)
#   2. uv sync
#   3. ollama pull qwen2.5:7b + qwen2.5vl:7b (the two models the pipeline uses)
#   4. Print next-step hints (assets, test videos)
#
# Usage from repo root:
#   bash scripts/bootstrap.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== SemanticVibe bootstrap ==="
echo "Repo: $REPO_ROOT"
echo

# --- 1. Prerequisites --------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not found. Install from https://github.com/astral-sh/uv" >&2
    exit 1
fi

SKIP_OLLAMA=0
if ! command -v ollama >/dev/null 2>&1; then
    echo "WARNING: 'ollama' not found. Pipeline will fail at the LLM/VLM step."
    echo "         Install from https://ollama.com and re-run, or use --provider rule_based."
    SKIP_OLLAMA=1
fi

# --- 2. uv sync --------------------------------------------------------------
echo
echo "[1/2] uv sync (Python deps)..."
uv sync

# --- 3. Ollama pulls ---------------------------------------------------------
if [ "$SKIP_OLLAMA" -eq 0 ]; then
    echo
    echo "[2/2] Pulling Ollama models (qwen2.5:7b ~4.7GB, qwen2.5vl:7b ~5.9GB)..."
    for model in "qwen2.5:7b" "qwen2.5vl:7b"; do
        echo "  pulling $model"
        ollama pull "$model" || echo "  WARNING: pull failed for $model"
    done
fi

# --- 4. Next steps -----------------------------------------------------------
echo
echo "=== Bootstrap done ==="
echo
echo "Next:"
echo "  1. Drop a video into data/test_videos/ (.mp4 or .mov)"
echo "  2. Populate assets/stickers/<tag>/*.png (zipped asset pack from"
echo "     your team OR run: uv run python scripts/generate_placeholder_assets.py"
echo "     for 9 procedural placeholders)"
echo "  3. Build the asset index: uv run python scripts/build_index.py"
echo "  4. Render: uv run python -m semanticvibe.render --video data/test_videos/<file> --out outputs/test.mp4 --preview"
