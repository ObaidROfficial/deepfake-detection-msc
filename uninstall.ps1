# Remove everything this project installed - Windows.
#
#   .\uninstall.ps1              show what would be removed, then ask
#   .\uninstall.ps1 -DryRun      show only, change nothing
#   .\uninstall.ps1 -Yes         skip the confirmation prompt
#   .\uninstall.ps1 -Caches      also clear shared caches (see warning below)
#
# WHAT LIVES WHERE
#
# Almost everything is inside the project folder and disappears when you delete
# it. Three things are written outside, which is why this script exists:
#
#   %USERPROFILE%\.cache\torch\hub\checkpoints   ImageNet weights torchvision
#                                                downloads on first use (~35MB)
#   %USERPROFILE%\.cache\huggingface             left by older versions
#   %USERPROFILE%\.kaggle\kaggle.json            only if you created it
#
# The shared caches are NOT removed by default - they are shared with other
# Python projects on this machine. Use -Caches only if you are sure.

param(
    [switch]$DryRun,
    [switch]$Yes,
    [switch]$Caches
)

$ErrorActionPreference = "Continue"
$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path

function SizeOf($p) {
    if (-not (Test-Path $p)) { return "-" }
    $b = (Get-ChildItem $p -Recurse -Force -ErrorAction SilentlyContinue |
          Measure-Object -Property Length -Sum).Sum
    if (-not $b) { return "0 B" }
    if ($b -gt 1GB) { return "{0:N1} GB" -f ($b/1GB) }
    if ($b -gt 1MB) { return "{0:N1} MB" -f ($b/1MB) }
    return "{0:N0} KB" -f ($b/1KB)
}

Write-Host "==================================================================="
Write-Host "  OBI DeepFake Image Detector - uninstall"
Write-Host "==================================================================="
Write-Host "`nProject folder: $HERE`n"

$inside = @(
    "$HERE\venv", "$HERE\data\raw", "$HERE\data\features",
    "$HERE\models", "$HERE\demo_samples", "$HERE\results\figures"
)

Write-Host "INSIDE the project folder" -ForegroundColor Cyan
foreach ($p in $inside) {
    if (Test-Path $p) {
        "{0,-10} {1}" -f (SizeOf $p), $p.Replace("$HERE\","") | Write-Host "   $_"
    }
}
Write-Host "   (all of the above go if you delete the project folder)"

$torch  = "$env:USERPROFILE\.cache\torch\hub\checkpoints"
$hf     = "$env:USERPROFILE\.cache\huggingface"
$kaggle = "$env:USERPROFILE\.kaggle\kaggle.json"

Write-Host "`nOUTSIDE the project folder" -ForegroundColor Cyan
if (Test-Path $torch) {
    Write-Host ("   {0,-10} {1}" -f (SizeOf $torch), $torch) -ForegroundColor Yellow
    Get-ChildItem $torch -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host "            $($_.Name)" }
}
if (Test-Path $hf)     { Write-Host ("   {0,-10} {1}" -f (SizeOf $hf), $hf) -ForegroundColor Yellow }
if (Test-Path $kaggle) { Write-Host ("   {0,-10} {1}" -f (SizeOf $kaggle), $kaggle) -ForegroundColor Yellow }

Write-Host "`nShared caches are used by other Python projects on this machine." -ForegroundColor Yellow
Write-Host "They are left alone unless you pass -Caches." -ForegroundColor Yellow

if ($DryRun) {
    Write-Host "`nDry run - nothing was changed." -ForegroundColor Green
    exit 0
}

Write-Host "`nThis will remove:"
Write-Host "   - the virtual environment, downloaded datasets, cached features,"
Write-Host "     model weights and generated figures inside the project folder"
if ($Caches) { Write-Host "   - the shared torch and huggingface caches" -ForegroundColor Red }
Write-Host "`nIt will NOT remove the project folder itself, your source code, or git"
Write-Host "history. Delete the folder manually afterwards if you want it all gone."

if (-not $Yes) {
    $reply = Read-Host "`nContinue? [y/N]"
    if ($reply -notmatch '^[Yy]$') { Write-Host "Cancelled."; exit 0 }
}

Write-Host ""
foreach ($p in $inside) {
    if (Test-Path $p) {
        Remove-Item $p -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "   removed $($p.Replace("$HERE\",''))" -ForegroundColor Green
    }
}
Get-ChildItem $HERE -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "   removed __pycache__ directories" -ForegroundColor Green

if ($Caches) {
    foreach ($p in @($torch, $hf)) {
        if (Test-Path $p) {
            Remove-Item $p -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "   removed $p" -ForegroundColor Green
        }
    }
}

Write-Host "`nDone." -ForegroundColor Green
Write-Host "`nStill present (deliberately):"
Write-Host "   - the project folder, source code and git history"
if (-not $Caches) { Write-Host "   - shared torch/huggingface caches (rerun with -Caches to clear)" }
if (Test-Path $kaggle) { Write-Host "   - $kaggle  (your credential - delete it yourself if you wish)" }
Write-Host "`nTo remove everything, delete the folder:"
Write-Host "   cd ..; Remove-Item -Recurse -Force '$(Split-Path -Leaf $HERE)'"
Write-Host ""
