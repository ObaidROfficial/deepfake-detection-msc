"""
Radially averaged power spectra of real and fake images, both datasets.

Added after draft feedback, which pointed out that the report leaned on a
high-frequency explanation without evidence that went beyond the model's own
behaviour. Every result supporting that account so far came from perturbing
inputs and watching accuracy move, which shows the model is sensitive to
high-frequency content but says nothing about whether a real/fake difference
actually lives there in the data.

This measures the data directly and needs no model at all. For each image:
convert to greyscale, apply a Hann window so the image border does not leak
into the spectrum, take the 2-D FFT, and average the power over rings of
constant spatial frequency. Averaging those curves per class gives one profile
per dataset and class.

Two questions it can answer:

  1. Is there a real/fake difference in the frequency domain in Dataset A,
     and is it concentrated at high frequency?
  2. Does the same difference appear in Dataset B?

A difference present in A and absent, or reversed, in B would be a direct
data-level account of why the detectors transfer so poorly. If the curves
instead look alike, the high-frequency story is weaker than the report claims
and the wording has to change. Either outcome is reportable.

    python src/spectral_analysis.py
    python src/spectral_analysis.py --n 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, get_logger, load_config, resolve, set_seed  # noqa: E402

log = get_logger("spectral")

SIZE = 256          # both datasets are natively 256x256


def radial_profile(power: np.ndarray) -> np.ndarray:
    """Average 2-D power over rings of constant frequency."""
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    total = np.bincount(r.ravel(), weights=power.ravel())
    count = np.bincount(r.ravel())
    return total[: cx] / np.maximum(count[: cx], 1)


def image_spectrum(path: Path, window: np.ndarray) -> np.ndarray | None:
    try:
        im = Image.open(path).convert("L")
    except Exception:
        return None
    if im.size != (SIZE, SIZE):
        im = im.resize((SIZE, SIZE), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float64) / 255.0
    a = (a - a.mean()) * window           # de-mean: the DC term is not of interest
    f = np.fft.fftshift(np.fft.fft2(a))
    return radial_profile(np.abs(f) ** 2)


def profile_for(df: pd.DataFrame, label: int, n: int, window) -> np.ndarray:
    rows = df[df["label"] == label].head(n)
    acc, used = None, 0
    for p in rows["path"]:
        prof = image_spectrum(resolve(p), window)
        if prof is None:
            continue
        acc = prof if acc is None else acc + prof
        used += 1
    log.info("  label %d: averaged %d images", label, used)
    return acc / max(used, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500, help="images per class per dataset")
    args = ap.parse_args()

    config = load_config()
    set_seed(config["project"]["seed"])

    win1 = np.hanning(SIZE)
    window = np.outer(win1, win1)

    sets = {
        "A": resolve("data/manifests/subset_test.csv"),
        "B": resolve("data/manifests/subset_crosstest.csv"),
    }

    out = {}
    for name, path in sets.items():
        df = pd.read_csv(path).sample(frac=1, random_state=config["project"]["seed"])
        log.info("Dataset %s (%s)", name, path.name)
        out[f"{name}_real"] = profile_for(df, 0, args.n, window)
        out[f"{name}_fake"] = profile_for(df, 1, args.n, window)

    freq = np.arange(len(out["A_real"])) / SIZE      # cycles per pixel
    table = pd.DataFrame({"freq_cycles_per_pixel": freq, **out})

    results = resolve("results")
    results.mkdir(exist_ok=True)
    csv = results / "spectral_profiles.csv"
    table.to_csv(csv, index=False)
    log.info("wrote %s", csv)

    # The number the report needs: how far apart are real and fake in the top
    # half of the band, and does that gap survive into the second dataset?
    hi = freq >= 0.25
    for name in ("A", "B"):
        r, f = out[f"{name}_real"][hi], out[f"{name}_fake"][hi]
        ratio = float(np.mean(np.log10(f) - np.log10(r)))
        log.info("Dataset %s: mean log10 power difference (fake - real) above "
                 "0.25 cyc/px = %+.4f", name, ratio)

    plot(freq, out, results / "figures" / "fig_6_26_spectral_profiles.png")


def plot(freq, out, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, name, title in zip(
            axes, ("A", "B"),
            ("Dataset A (StyleGAN against FFHQ)",
             "Dataset B (OpenForensics)")):
        ax.plot(freq, np.log10(out[f"{name}_real"]), label="real", lw=1.6)
        ax.plot(freq, np.log10(out[f"{name}_fake"]), label="fake", lw=1.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("spatial frequency (cycles per pixel)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("log$_{10}$ mean power")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
