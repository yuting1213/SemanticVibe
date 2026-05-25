# fix_pth.ps1 — Repair the editable-install .pth on Traditional-Chinese
# Windows (cp950).
#
# Background:
#   `uv sync` writes `.venv/Lib/site-packages/_editable_impl_semanticvibe.pth`
#   in UTF-8. On Traditional-Chinese Windows the system locale is cp950 and
#   CPython's site.py reads .pth files with `encoding="locale"` — so any
#   non-ASCII char in the repo path makes the venv refuse to start with
#   `UnicodeDecodeError: 'cp950' codec can't decode byte 0x96`.
#
# This script:
#   1. Detects the repo root from the script's own location.
#   2. Picks an ASCII-safe path: if the repo root itself is ASCII, uses
#      `<repo_root>\src`; if it contains non-ASCII chars, falls back to
#      `C:\sv\src` (the documented junction) and verifies that junction
#      actually points back here.
#   3. Rewrites the .pth in pure ASCII so site.py can load it under any
#      system locale.
#
# Usage:
#   .\scripts\fix_pth.ps1
#
# Run this:
#   - Every time after `uv sync --reinstall` (which clobbers the .pth)
#   - Once after first `uv sync` if you cloned into a CJK path

$ErrorActionPreference = "Stop"

# 1. Locate repo root + .pth file.
$repoRoot  = (Resolve-Path "$PSScriptRoot\..").Path
$pthPath   = Join-Path $repoRoot ".venv\Lib\site-packages\_editable_impl_semanticvibe.pth"

if (-not (Test-Path $pthPath)) {
    Write-Host "ERROR: .pth file not found at $pthPath" -ForegroundColor Red
    Write-Host "Run 'uv sync' first to create the venv." -ForegroundColor Red
    exit 1
}

# 2. Decide which path to write. Prefer the repo root if ASCII-safe.
function Test-Ascii([string]$s) {
    return ($s -match '^[\x00-\x7F]+$')
}

$srcPath = Join-Path $repoRoot "src"

if (-not (Test-Ascii $srcPath)) {
    # Repo path has non-ASCII chars. Fall back to the C:\sv junction
    # documented in CLAUDE.md.
    $junctionRoot = "C:\sv"
    $junctionSrc  = "C:\sv\src"

    if (-not (Test-Path $junctionRoot)) {
        Write-Host "Repo path contains non-ASCII characters:" -ForegroundColor Yellow
        Write-Host "  $repoRoot" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "Create a junction first, then re-run:" -ForegroundColor Yellow
        Write-Host "  New-Item -ItemType Junction -Path C:\sv -Target `"$repoRoot`"" -ForegroundColor Cyan
        exit 1
    }

    # Sanity check: junction must point at this repo.
    $junctionTarget = (Get-Item $junctionRoot).Target
    if ($junctionTarget -ne $repoRoot) {
        Write-Host "WARNING: C:\sv points to $junctionTarget, not this repo" -ForegroundColor Yellow
        Write-Host "  This repo: $repoRoot" -ForegroundColor Yellow
        Write-Host "  Re-create the junction if you want this repo." -ForegroundColor Yellow
    }

    $srcPath = $junctionSrc
}

# 3. Write .pth in pure ASCII.
[System.IO.File]::WriteAllText($pthPath, "$srcPath`r`n", [System.Text.Encoding]::ASCII)

Write-Host "OK: .pth rewritten" -ForegroundColor Green
Write-Host "  $pthPath" -ForegroundColor Gray
Write-Host "  -> $srcPath" -ForegroundColor Gray
