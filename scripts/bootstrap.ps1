# bootstrap.ps1 — One-command setup for fresh clones on Windows.
#
# Walks through:
#   1. Verify prerequisites (uv, ollama)
#   2. Optionally suggest C:\sv junction if cloned into a CJK path
#   3. uv sync
#   4. fix_pth.ps1 (handles cp950 .pth issue if any)
#   5. ollama pull qwen2.5:7b + qwen2.5vl:7b (the two models the pipeline uses)
#   6. Print next-step hints (assets, test videos)
#
# Usage from repo root:
#   .\scripts\bootstrap.ps1

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path

Write-Host "=== SemanticVibe bootstrap ===" -ForegroundColor Cyan
Write-Host "Repo: $repoRoot" -ForegroundColor Gray
Write-Host ""

# --- 1. Prerequisites ---------------------------------------------------------
function Test-Cmd($name) { return (Get-Command $name -ErrorAction SilentlyContinue) -ne $null }

if (-not (Test-Cmd "uv")) {
    Write-Host "ERROR: 'uv' not found. Install from https://github.com/astral-sh/uv" -ForegroundColor Red
    exit 1
}
if (-not (Test-Cmd "ollama")) {
    Write-Host "WARNING: 'ollama' not found. Pipeline will fail at the LLM/VLM step." -ForegroundColor Yellow
    Write-Host "         Install from https://ollama.com and re-run, or use --provider rule_based." -ForegroundColor Yellow
    $skipOllama = $true
}

# --- 2. CJK path check --------------------------------------------------------
function Test-Ascii([string]$s) { return ($s -match '^[\x00-\x7F]+$') }

if (-not (Test-Ascii $repoRoot)) {
    Write-Host "Repo path contains non-ASCII characters." -ForegroundColor Yellow
    if (-not (Test-Path "C:\sv")) {
        Write-Host "Creating C:\sv junction so the Python venv can load (cp950 workaround)..." -ForegroundColor Yellow
        New-Item -ItemType Junction -Path "C:\sv" -Target $repoRoot | Out-Null
        Write-Host "OK: C:\sv -> $repoRoot" -ForegroundColor Green
        Write-Host "Re-run bootstrap from C:\sv for the rest of setup:" -ForegroundColor Yellow
        Write-Host "  cd C:\sv; .\scripts\bootstrap.ps1" -ForegroundColor Cyan
        exit 0
    } else {
        $junctionTarget = (Get-Item "C:\sv").Target
        Write-Host "Existing C:\sv junction -> $junctionTarget" -ForegroundColor Gray
    }
}

# --- 3. uv sync ---------------------------------------------------------------
Write-Host ""
Write-Host "[1/3] uv sync (Python deps)..." -ForegroundColor Cyan
& uv sync
if ($LASTEXITCODE -ne 0) { Write-Host "uv sync failed" -ForegroundColor Red; exit 1 }

# --- 4. fix_pth ---------------------------------------------------------------
Write-Host ""
Write-Host "[2/3] Repairing .pth (cp950 safety)..." -ForegroundColor Cyan
& "$PSScriptRoot\fix_pth.ps1"

# --- 5. Ollama pulls ----------------------------------------------------------
if (-not $skipOllama) {
    Write-Host ""
    Write-Host "[3/3] Pulling Ollama models (qwen2.5:7b ~4.7GB, qwen2.5vl:7b ~5.9GB)..." -ForegroundColor Cyan
    foreach ($model in @("qwen2.5:7b", "qwen2.5vl:7b")) {
        Write-Host "  pulling $model" -ForegroundColor Gray
        & ollama pull $model
        if ($LASTEXITCODE -ne 0) { Write-Host "  WARNING: pull failed for $model" -ForegroundColor Yellow }
    }
}

# --- 6. Next steps ------------------------------------------------------------
Write-Host ""
Write-Host "=== Bootstrap done ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next:" -ForegroundColor Cyan
Write-Host "  1. Drop a video into data\test_videos\ (.mp4 or .mov)"
Write-Host "  2. Populate assets\stickers\<tag>\*.png (zipped asset pack from"
Write-Host "     your team OR run: uv run python scripts\generate_placeholder_assets.py"
Write-Host "     for 9 procedural placeholders)"
Write-Host "  3. Build the asset index: uv run python scripts\build_index.py"
Write-Host "  4. Render: uv run python -m semanticvibe.render --video data\test_videos\<file> --out outputs\test.mp4 --preview"
