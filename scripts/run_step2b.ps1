$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python "scripts\step2b_revision.py"

$zipPath = "results\step2b_outputs.zip"
if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path "results\step2b\*" -DestinationPath $zipPath -Force
Write-Host "Wrote $zipPath"
