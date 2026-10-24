# Install ffmpeg for ViDub on Windows via winget, with a Chocolatey fallback hint.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File tools/install_ffmpeg.ps1

$ErrorActionPreference = "Stop"

Write-Host "ViDub: installing ffmpeg..."

if (Get-Command winget -ErrorAction SilentlyContinue) {
    try {
        winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
        Write-Host "ffmpeg installed via winget."
        Write-Host "Open a new shell and run: vidub doctor"
        exit 0
    } catch {
        Write-Warning "winget install failed: $_"
    }
} else {
    Write-Warning "winget was not found on PATH."
}

Write-Host ""
Write-Host "Fallback: install ffmpeg with Chocolatey:"
Write-Host "    choco install ffmpeg"
Write-Host "Or download a build from https://www.gyan.dev/ffmpeg/builds/ and add its bin/ directory to PATH."
exit 1
