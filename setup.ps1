# One-command setup for Windows.
#
#   .\setup.ps1              full install (experiments + GUI)
#   .\setup.ps1 -Minimal     demo only - smaller and faster
#   .\setup.ps1 -SkipWeights don't download model weights
#
# Creates a virtual environment, installs the CPU build of PyTorch and the
# dependencies, downloads the trained weights from GitHub Releases, and runs a
# smoke test. Safe to re-run - existing steps are skipped.
#
# If PowerShell blocks this with an execution policy error, run:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

param(
    [switch]$Minimal,
    [switch]$SkipWeights
)

# NOT "Stop". PowerShell treats native-command stderr as terminating under
# Stop, and pip and git both write progress there.
$ErrorActionPreference = "Continue"

$RELEASE = "https://github.com/ObaidROfficial/deepfake-detection-msc/releases/latest/download"
$WEIGHTS = @("mobilenet_v2_best.pt", "efficientnet_b0_best.pt")

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Ok($msg)       { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg)     { Write-Host "    $msg" -ForegroundColor Yellow }
function Fail($msg)     { Write-Host "    $msg" -ForegroundColor Red }

Write-Host "==================================================================="
Write-Host "  Deepfake Detector - setup"
Write-Host "  MSc Advanced Computing Project (COM7014)"
Write-Host "==================================================================="

# ---------------------------------------------------------------- 1. Python
Step 1 "Checking Python"
$py = $null
foreach ($cmd in @("py -3.11", "py -3", "python")) {
    $parts = $cmd.Split(" ")
    $exe = $parts[0]
    if (Get-Command $exe -ErrorAction SilentlyContinue) {
        $v = & $exe $parts[1..$parts.Length] --version 2>&1
        if ($LASTEXITCODE -eq 0) { $py = $cmd; Ok "Using '$cmd' -> $v"; break }
    }
}
if (-not $py) {
    Fail "No Python found. Install Python 3.11 from python.org and re-run."
    exit 1
}

# ------------------------------------------------------------------ 2. venv
Step 2 "Virtual environment"
if (Test-Path "venv") {
    Ok "venv already exists - reusing it"
} else {
    Invoke-Expression "$py -m venv venv"
    if (-not (Test-Path "venv")) { Fail "Could not create venv."; exit 1 }
    Ok "Created venv"
}
$vpy = ".\venv\Scripts\python.exe"

# ----------------------------------------------------------- 3. dependencies
Step 3 "Installing dependencies (this is the slow part)"
& $vpy -m pip install --upgrade pip --quiet
Ok "pip upgraded"

Write-Host "    Installing PyTorch (CPU build, ~200MB)..."
& $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
if ($LASTEXITCODE -ne 0) { Fail "PyTorch install failed."; exit 1 }
Ok "PyTorch installed"

if ($Minimal) {
    Write-Host "    Installing demo dependencies only..."
    & $vpy -m pip install pillow pyyaml numpy pandas "opencv-python<5" gradio --quiet
    Ok "Demo dependencies installed (experiments will not run - omit -Minimal for those)"
} else {
    Write-Host "    Installing full requirements..."
    & $vpy -m pip install -r requirements.txt --quiet
    if ($LASTEXITCODE -ne 0) { Fail "Requirements install failed."; exit 1 }
    Ok "All requirements installed"
}

# --------------------------------------------------------------- 4. weights
Step 4 "Trained model weights"
if ($SkipWeights) {
    Warn "Skipped by request."
} else {
    New-Item -ItemType Directory -Force -Path "models" | Out-Null
    foreach ($w in $WEIGHTS) {
        $dest = "models\$w"
        if (Test-Path $dest) { Ok "$w already present"; continue }
        Write-Host "    Downloading $w ..."
        try {
            Invoke-WebRequest -Uri "$RELEASE/$w" -OutFile $dest -UseBasicParsing
            $mb = [math]::Round((Get-Item $dest).Length / 1MB, 1)
            Ok "$w downloaded ($mb MB)"
        } catch {
            if (Test-Path $dest) { Remove-Item $dest -Force }
            Warn "Could not download $w"
            Warn "Get it manually from the Releases page and put it in models\"
        }
    }
}

# ------------------------------------------------------------ 5. smoke test
Step 5 "Verifying the install"
$check = @'
import sys
try:
    import torch, torchvision, PIL, yaml, numpy
    print(f"    torch {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
except Exception as e:
    print(f"    IMPORT FAILED: {e}"); sys.exit(1)
try:
    import cv2; print("    opencv available - face detection enabled")
except ImportError:
    print("    opencv missing - face cropping will be skipped")
try:
    import gradio; print(f"    gradio {gradio.__version__} - GUI available")
except ImportError:
    print("    gradio missing - GUI unavailable (command line still works)")
from pathlib import Path
w = sorted(p.name for p in Path("models").glob("*_best.pt"))
print(f"    weights found: {w if w else 'NONE - download from Releases'}")
'@
$check | & $vpy -
if ($LASTEXITCODE -ne 0) { Fail "Verification failed - see above."; exit 1 }

# ------------------------------------------------------------------- done
Write-Host "`n===================================================================" -ForegroundColor Green
Write-Host "  Setup complete" -ForegroundColor Green
Write-Host "===================================================================" -ForegroundColor Green
Write-Host @"

  ####################################################################
  #                                                                  #
  #   TO START THE APP, RUN:      .\run.ps1                          #
  #                                                                  #
  ####################################################################

  run.ps1 uses the virtual environment directly, so you do not need to
  activate it first.

  Other options:

      .\run.ps1 -Share                  temporary public link
      .\run.ps1 -Image photo.jpg        classify one image, no GUI

  Or do it manually if you prefer:

      .\venv\Scripts\Activate.ps1
      python app.py

  To reproduce the reported experiments you also need the datasets -
  see REPRODUCE.md, Path C.

"@
