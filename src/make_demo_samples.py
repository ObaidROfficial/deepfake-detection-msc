"""
Pull a handful of images out of the held-out test set for demos and screenshots.

These come from the test split, which no model saw during training, so they are
a fair demonstration rather than a rigged one. Measured accuracy on this split
is 94.3%, so most should classify correctly - and the occasional mistake is
worth showing too.

    python src/make_demo_samples.py
    python src/make_demo_samples.py --n 10 --out my_samples

Files are named so the ground truth is visible:
    real_01_<id>.jpg
    fake_01_<id>.jpg

NOTE ON LICENSING. The output folder is gitignored and must stay that way.
Dataset A's real images derive from NVIDIA's FFHQ under CC BY-NC-SA 4.0 and
this project does not redistribute them - see LICENSE. Generate these locally;
do not commit them or attach them to a release.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, get_logger, load_config, resolve  # noqa: E402

log = get_logger("demo_samples")


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract demo images from the test split")
    ap.add_argument("--n", type=int, default=6, help="images per class")
    ap.add_argument("--out", default="demo_samples", help="output folder")
    ap.add_argument("--split", default="test",
                    choices=["test", "crosstest", "val"],
                    help="'test' = Dataset A (in-domain, ~94%% accuracy). "
                         "'crosstest' = Dataset B, the unseen method the model "
                         "fails on - useful for demonstrating the limitation.")
    args = ap.parse_args()

    config = load_config()
    seed = config["project"]["seed"]

    manifest = resolve("data/manifests") / f"subset_{args.split}.csv"
    if not manifest.exists():
        log.error(
            "No manifest at %s\n"
            "  Run the data pipeline first: python src/data_prep.py --step all",
            manifest,
        )
        sys.exit(1)

    df = pd.read_csv(manifest)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for label, name in ((0, "real"), (1, "fake")):
        pool = df[df.label == label].sample(args.n, random_state=seed)
        for i, (_, row) in enumerate(pool.iterrows(), start=1):
            src = ROOT / row["path"]
            if not src.exists():
                log.warning("Missing: %s", src)
                continue
            dest = out / f"{name}_{i:02d}_{src.stem}{src.suffix}"
            shutil.copy2(src, dest)
            copied += 1

    log.info("Copied %d images to %s", copied, out)
    log.info("")
    log.info("Try them:")
    log.info("    python src/predict.py --folder %s --no-crop", args.out)
    log.info("")
    log.info("  --no-crop is used because these are ALREADY tight face crops.")
    log.info("  Running face detection on them again would crop into the face")
    log.info("  and discard the border the model was trained to see.")
    if args.split == "crosstest":
        log.info("")
        log.info("  These are Dataset B images. The model scores ~50%% on them,")
        log.info("  so expect roughly half to be wrong. That is the finding.")


if __name__ == "__main__":
    main()
