#!/usr/bin/env bash
# Install ffmpeg for ViDub on Windows from Git Bash, via winget.
#
# Usage:
#   bash tools/install_ffmpeg.sh

set -euo pipefail

echo "ViDub: installing ffmpeg..."

if command -v winget >/dev/null 2>&1; then
    if winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements; then
        echo "ffmpeg installed via winget."
        echo "Open a new shell and run: vidub doctor"
        exit 0
    fi
    echo "winget install failed." >&2
else
    echo "winget was not found on PATH." >&2
fi

echo ""
echo "Fallback options:"
echo "    choco install ffmpeg          # Chocolatey"
echo "    winget install Gyan.FFmpeg -e # from PowerShell"
echo "Or download a build from https://www.gyan.dev/ffmpeg/builds/ and add its bin/ directory to PATH."
exit 1
