#!/usr/bin/env bash
# One-command setup for macOS and Linux.
#
#   ./setup.sh                full install (experiments + GUI)
#   ./setup.sh --minimal      demo only - smaller and faster
#   ./setup.sh --skip-weights don't download model weights
#
# Creates a virtual environment, installs the CPU build of PyTorch and the
# dependencies, downloads the trained weights from GitHub Releases, and runs a
# smoke test. Safe to re-run.
#
# You may need to make it executable first:  chmod +x setup.sh

set -uo pipefail

MINIMAL=0
SKIP_WEIGHTS=0
for arg in "$@"; do
  case "$arg" in
    --minimal)      MINIMAL=1 ;;
    --skip-weights) SKIP_WEIGHTS=1 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

RELEASE="https://github.com/ObaidROfficial/deepfake-detection-msc/releases/latest/download"
WEIGHTS=("mobilenet_v2_best.pt" "efficientnet_b0_best.pt")

C_CYAN='\033[0;36m'; C_GREEN='\033[0;32m'; C_YELLOW='\033[0;33m'
C_RED='\033[0;31m';  C_OFF='\033[0m'
step() { echo -e "\n${C_CYAN}[$1] $2${C_OFF}"; }
ok()   { echo -e "    ${C_GREEN}$1${C_OFF}"; }
warn() { echo -e "    ${C_YELLOW}$1${C_OFF}"; }
fail() { echo -e "    ${C_RED}$1${C_OFF}"; }

echo "==================================================================="
echo "  Deepfake Detector - setup"
echo "  MSc Advanced Computing Project (COM7014)"
echo "==================================================================="

# ---------------------------------------------------------------- 1. Python
step 1 "Checking Python"
PY=""
for c in python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1; then
    PY="$c"; ok "Using '$c' -> $($c --version 2>&1)"; break
  fi
done
[ -z "$PY" ] && { fail "No Python found. Install Python 3.11 and re-run."; exit 1; }

# ------------------------------------------------------------------ 2. venv
step 2 "Virtual environment"
if [ -d venv ]; then
  ok "venv already exists - reusing it"
else
  "$PY" -m venv venv || { fail "Could not create venv."; exit 1; }
  ok "Created venv"
fi
VPY="./venv/bin/python"

# ----------------------------------------------------------- 3. dependencies
step 3 "Installing dependencies (this is the slow part)"
"$VPY" -m pip install --upgrade pip --quiet && ok "pip upgraded"

echo "    Installing PyTorch (CPU build, ~200MB)..."
"$VPY" -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu --quiet \
  || { fail "PyTorch install failed."; exit 1; }
ok "PyTorch installed"

if [ "$MINIMAL" -eq 1 ]; then
  echo "    Installing demo dependencies only..."
  "$VPY" -m pip install pillow pyyaml numpy pandas opencv-python gradio --quiet
  ok "Demo dependencies installed (omit --minimal for the experiments)"
else
  echo "    Installing full requirements..."
  "$VPY" -m pip install -r requirements.txt --quiet \
    || { fail "Requirements install failed."; exit 1; }
  ok "All requirements installed"
fi

# --------------------------------------------------------------- 4. weights
step 4 "Trained model weights"
if [ "$SKIP_WEIGHTS" -eq 1 ]; then
  warn "Skipped by request."
else
  mkdir -p models
  for w in "${WEIGHTS[@]}"; do
    if [ -f "models/$w" ]; then ok "$w already present"; continue; fi
    echo "    Downloading $w ..."
    if curl -fSL --progress-bar "$RELEASE/$w" -o "models/$w"; then
      ok "$w downloaded ($(du -h "models/$w" | cut -f1))"
    else
      rm -f "models/$w"
      warn "Could not download $w"
      warn "Get it manually from the Releases page and put it in models/"
    fi
  done
fi

# ------------------------------------------------------------ 5. smoke test
step 5 "Verifying the install"
"$VPY" - <<'PYCHECK'
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
PYCHECK
[ $? -ne 0 ] && { fail "Verification failed - see above."; exit 1; }

echo -e "\n${C_GREEN}===================================================================${C_OFF}"
echo -e "${C_GREEN}  Setup complete${C_OFF}"
echo -e "${C_GREEN}===================================================================${C_OFF}"
cat <<'EOF'

  ####################################################################
  #                                                                  #
  #   TO START THE APP, RUN:      ./run.sh                           #
  #                                                                  #
  ####################################################################

  run.sh activates the virtual environment for you. Note that on macOS
  the bare command `python` does not exist - only `python3` - so
  running `python app.py` without activating the venv will fail with
  "command not found".

  Other options:

      ./run.sh --share                  temporary public link
      ./run.sh --image photo.jpg        classify one image, no GUI

  Or do it manually if you prefer:

      source venv/bin/activate
      python app.py

  To reproduce the reported experiments you also need the datasets -
  see REPRODUCE.md, Path C.

EOF
