# Launch the detector window - Windows.
#
#   .\run.ps1                      open the detector window
#   .\run.ps1 -Image photo.jpg     classify one image on the command line
#
# Uses the virtual environment's Python directly, so there is no need to
# activate it first. Forgetting to activate is the most common way to fail at
# this step.

param(
    [string]$Image,
    [string]$Folder
)

$ErrorActionPreference = "Continue"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

$py = ".\venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "No virtual environment found." -ForegroundColor Red
    Write-Host "Run the setup script first:"
    Write-Host "    .\setup.ps1"
    exit 1
}

if (-not (Get-ChildItem "models\*_best.pt" -ErrorAction SilentlyContinue)) {
    Write-Host "No model weights in models\" -ForegroundColor Yellow
    Write-Host "Download them from:"
    Write-Host "    https://github.com/ObaidROfficial/deepfake-detection-msc/releases"
    Write-Host "or rerun .\setup.ps1 which fetches them automatically."
    exit 1
}

if ($Image)  { & $py src\predict.py --image  $Image;  exit $LASTEXITCODE }
if ($Folder) { & $py src\predict.py --folder $Folder; exit $LASTEXITCODE }

Write-Host "Starting the detector..." -ForegroundColor Green
Write-Host "The window opens in a few seconds."
$args = @("app.py")
& $py @args
