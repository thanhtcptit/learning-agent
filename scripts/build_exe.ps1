param(
    [switch]$Clean = $true
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

if ($Clean) {
    uv run --with pyinstaller pyinstaller --noconfirm --clean learning-agent.spec
    exit $LASTEXITCODE
}

uv run --with pyinstaller pyinstaller --noconfirm learning-agent.spec
