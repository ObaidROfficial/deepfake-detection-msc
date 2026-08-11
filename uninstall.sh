#!/usr/bin/env bash
# Remove everything this project installed - macOS and Linux.
#
#   ./uninstall.sh              show what would be removed, then ask
#   ./uninstall.sh --dry-run    show only, change nothing
#   ./uninstall.sh --yes        skip the confirmation prompt
#   ./uninstall.sh --caches     also clear shared caches (see warning below)
#
# WHAT LIVES WHERE
#
# Almost everything is inside the project folder and disappears when you delete
# it. Three things are written outside, which is why this script exists:
#
#   ~/.cache/torch/hub/checkpoints   ImageNet weights torchvision downloads
#                                    on first use (~35MB for both models)
#   ~/.cache/huggingface             pulled in by Gradio's dependencies
#   ~/.kaggle/kaggle.json            only if you created it for the datasets
#
# The shared caches are NOT removed by default. They are shared with any other
# Python project on this machine, and deleting them would force those projects
# to re-download too. Use --caches only if you are sure.

set -uo pipefail

DRY=0; YES=0; CACHES=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --yes|-y)  YES=1 ;;
    --caches)  CACHES=1 ;;
    *) echo "Unknown option: $a"; exit 1 ;;
  esac
done

G='\033[0;32m'; Y='\033[0;33m'; R='\033[0;31m'; C='\033[0;36m'; O='\033[0m'
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

size_of() { [ -e "$1" ] && du -sh "$1" 2>/dev/null | cut -f1 || echo "-"; }
exists()  { [ -e "$1" ]; }

echo "==================================================================="
echo "  OBI DeepFake Image Detector - uninstall"
echo "==================================================================="
echo -e "\nProject folder: $HERE\n"

echo -e "${C}INSIDE the project folder${O}"
INSIDE=("$HERE/venv" "$HERE/data/raw" "$HERE/data/features" "$HERE/models"
        "$HERE/demo_samples" "$HERE/results/figures")
for p in "${INSIDE[@]}"; do
  exists "$p" && printf "   %-8s %s\n" "$(size_of "$p")" "${p#$HERE/}"
done
echo "   (all of the above go if you delete the project folder)"

echo -e "\n${C}OUTSIDE the project folder${O}"
TORCH_CACHE="$HOME/.cache/torch/hub/checkpoints"
HF_CACHE="$HOME/.cache/huggingface"
KAGGLE="$HOME/.kaggle/kaggle.json"
# macOS sometimes uses ~/Library/Caches
[ -d "$HOME/Library/Caches/torch" ] && TORCH_CACHE="$HOME/Library/Caches/torch/hub/checkpoints"

if exists "$TORCH_CACHE"; then
  echo -e "   ${Y}$(size_of "$TORCH_CACHE")${O}  $TORCH_CACHE"
  ls "$TORCH_CACHE" 2>/dev/null | sed 's/^/            /'
fi
exists "$HF_CACHE" && echo -e "   ${Y}$(size_of "$HF_CACHE")${O}  $HF_CACHE"
exists "$KAGGLE"   && echo -e "   ${Y}$(size_of "$KAGGLE")${O}  $KAGGLE"

echo
echo -e "${Y}Shared caches are used by other Python projects on this machine.${O}"
echo -e "${Y}They are left alone unless you pass --caches.${O}"

if [ "$DRY" -eq 1 ]; then
  echo -e "\n${G}Dry run - nothing was changed.${O}"
  exit 0
fi

echo
echo "This will remove:"
echo "   - the virtual environment, downloaded datasets, cached features,"
echo "     model weights and generated figures inside the project folder"
[ "$CACHES" -eq 1 ] && echo -e "   ${R}- the shared torch and huggingface caches${O}"
echo
echo "It will NOT remove the project folder itself, your source code, or git"
echo "history. Delete the folder manually afterwards if you want it all gone."

if [ "$YES" -eq 0 ]; then
  read -r -p "Continue? [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }
fi

echo
for p in "${INSIDE[@]}"; do
  if exists "$p"; then rm -rf "$p" && echo -e "   ${G}removed${O} ${p#$HERE/}"; fi
done
find "$HERE" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find "$HERE" -name "*.pyc" -delete 2>/dev/null
echo -e "   ${G}removed${O} __pycache__ directories"

if [ "$CACHES" -eq 1 ]; then
  for p in "$TORCH_CACHE" "$HF_CACHE"; do
    if exists "$p"; then rm -rf "$p" && echo -e "   ${G}removed${O} $p"; fi
  done
fi

echo
echo -e "${G}Done.${O}"
echo
echo "Still present (deliberately):"
echo "   - the project folder, source code and git history"
[ "$CACHES" -eq 0 ] && echo "   - shared torch/huggingface caches (rerun with --caches to clear)"
exists "$KAGGLE" && echo "   - $KAGGLE  (your credential - delete it yourself if you wish)"
echo
echo "To remove everything, delete the folder:"
echo "   cd .. && rm -rf \"$(basename "$HERE")\""
echo
