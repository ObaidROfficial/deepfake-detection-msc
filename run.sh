#!/usr/bin/env bash
# Launch the graphical interface - macOS and Linux.
#
#   ./run.sh                     start the web interface
#   ./run.sh --share             also create a temporary public link
#   ./run.sh --image photo.jpg   classify one image on the command line instead
#
# Activates the virtual environment for you. This exists because forgetting to
# activate it is the single most common way to fail at this step: on macOS the
# bare command `python` does not exist at all, so `python app.py` outside the
# venv gives "command not found" rather than anything informative.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

G='\033[0;32m'; R='\033[0;31m'; Y='\033[0;33m'; O='\033[0m'

if [ ! -x "venv/bin/python" ]; then
  echo -e "${R}No virtual environment found.${O}"
  echo "Run the setup script first:"
  echo "    chmod +x setup.sh && ./setup.sh"
  exit 1
fi

if ! ls models/*_best.pt >/dev/null 2>&1; then
  echo -e "${Y}No model weights in models/${O}"
  echo "Download them from:"
  echo "    https://github.com/ObaidROfficial/deepfake-detection-msc/releases"
  echo "or rerun ./setup.sh which fetches them automatically."
  exit 1
fi

# --image (and friends) go to the command line tool; everything else to the GUI
if [ "${1:-}" = "--image" ] || [ "${1:-}" = "--folder" ]; then
  exec ./venv/bin/python src/predict.py "$@"
fi

echo -e "${G}Starting the interface...${O}  (Ctrl+C to stop)"
exec ./venv/bin/python app.py "$@"
