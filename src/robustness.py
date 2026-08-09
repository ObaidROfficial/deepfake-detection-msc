"""
Robustness evaluation - RQ4.

Re-scores a trained detector under perturbations that images routinely undergo
in the wild, plus a white-box adversarial attack:

  1. JPEG recompression   - social platforms re-encode everything
  2. Gaussian noise       - sensor noise, low light
  3. Gaussian blur        - soft focus, denoising filters
  4. Downscale round-trip - resizing for messaging/thumbnails
  5. Brightness shift     - exposure differences
  6. Contrast shift       - display/processing differences
  7. FGSM                 - deliberate white-box evasion

Why this matters for this project specifically. Chapter 6 established that
input resolution was worth +10 accuracy points, which supports the account that
the discriminative signal lives in high-frequency detail (Frank et al., 2020).

That account makes a falsifiable prediction, recorded here BEFORE the suite is
run. The right distinction is not "changes high-frequency energy" but
**destroys high-frequency information irreversibly** versus **rescales
intensities while preserving it**:

  DESTRUCTIVE (information is lost and cannot be recovered)
      JPEG at low quality, blur, downscale round-trip

  RESCALING (intensities change, spatial structure survives)
      brightness, contrast

  ADDITIVE (buries the signal under noise rather than removing it)
      Gaussian noise

Prediction: the destructive group should hurt substantially more than the
rescaling group.

**Measured caveat, recorded honestly.** Brightness and contrast are *not* clean
controls. Both are multiplicative, so they scale gradient magnitudes too:
measured on a sample image, contrast -0.4 cuts high-frequency energy to 0.60x
of clean and brightness +0.4 raises it to 1.30x, against 0.73x for JPEG q=10.
What separates them is that rescaling is near-invertible and preserves the
spatial pattern of gradients, whereas blur and downscaling discard it (0.45x
and 0.36x with the structure genuinely gone). The prediction rests on that
distinction, not on the raw energy numbers.

If all perturbation families degrade the detector roughly equally, the
high-frequency account is wrong and the resolution result needs another
explanation.

Usage:
    python src/robustness.py --model mobilenet_v2
    python src/robustness.py --model mobilenet_v2 --skip-fgsm
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import IMAGENET_MEAN, IMAGENET_STD, build_model, build_transforms  # noqa: E402
from utils import (  # noqa: E402
    ROOT, ensure_dirs, format_duration, get_logger, load_config, resolve,
    set_seed, setup_threads,
)

log = get_logger("robustness")


# ---------------------------------------------------------------------------
# Perturbations - all operate on a PIL image, applied BEFORE the eval transform
#
# IMPLEMENTED AS CLASSES, NOT CLOSURES. Windows starts DataLoader workers with
# 'spawn', which pickles the dataset (and therefore the perturbation) to send
# it to each worker process. A closure returned from a factory function cannot
# be pickled - it fails with
#     AttributeError: Can't pickle local object 'p_jpeg.<locals>.f'
# A module-level class with __call__ pickles fine because only its attributes
# need serialising. This is the standard fix and keeps multi-worker loading.
# ---------------------------------------------------------------------------

class Identity:
    def __call__(self, im):
        return im


class JPEGCompress:
    def __init__(self, quality: int):
        self.quality = quality

    def __call__(self, im):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=self.quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class GaussianNoise:
    def __init__(self, sigma: float, seed: int = 0):
        self.sigma = sigma
        self.seed = seed

    def __call__(self, im):
        # Derive the RNG seed from image content so the run is reproducible
        # while different images still receive different noise.
        a = np.asarray(im, dtype=np.float32) / 255.0
        rng = np.random.default_rng(self.seed + int(a.sum()) % 100000)
        a = np.clip(a + rng.normal(0, self.sigma, a.shape), 0, 1)
        return Image.fromarray((a * 255).astype(np.uint8))


class GaussianBlurK:
    """k is an odd kernel size; radius ~ k/3 gives comparable smoothing."""

    def __init__(self, k: int):
        self.k = k

    def __call__(self, im):
        return im.filter(ImageFilter.GaussianBlur(radius=self.k / 3.0))


class DownscaleRoundTrip:
    def __init__(self, target: int):
        self.target = target

    def __call__(self, im):
        w, h = im.size
        small = im.resize((self.target, self.target), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)


class BrightnessShift:
    def __init__(self, delta: float):
        self.delta = delta

    def __call__(self, im):
        return ImageEnhance.Brightness(im).enhance(1.0 + self.delta)


class ContrastShift:
    def __init__(self, delta: float):
        self.delta = delta

    def __call__(self, im):
        return ImageEnhance.Contrast(im).enhance(1.0 + self.delta)


def build_perturbations(cfg: dict) -> list[tuple[str, str, float, object]]:
    """(family, label, magnitude, callable) for every setting in config."""
    r = cfg["robustness"]
    out: list[tuple[str, str, float, object]] = [("clean", "clean", 0.0, Identity())]
    for q in r["jpeg_quality"]:
        out.append(("jpeg", f"JPEG q={q}", q, JPEGCompress(q)))
    for s in r["gaussian_noise_sigma"]:
        out.append(("noise", f"noise sigma={s}", s, GaussianNoise(s)))
    for k in r["gaussian_blur_kernel"]:
        out.append(("blur", f"blur k={k}", k, GaussianBlurK(k)))
    for d in r["downscale_factors"]:
        out.append(("downscale", f"downscale {d}px", d, DownscaleRoundTrip(d)))
    for d in r["brightness_shift"]:
        out.append(("brightness", f"brightness {d:+.1f}", d, BrightnessShift(d)))
    for d in r["contrast_shift"]:
        out.append(("contrast", f"contrast {d:+.1f}", d, ContrastShift(d)))
    return out


# ---------------------------------------------------------------------------
# Dataset that applies a perturbation before the model's transform
# ---------------------------------------------------------------------------

class PerturbedDataset(Dataset):
    def __init__(self, manifest_csv, root, perturb, transform, limit=None, seed=42):
        df = pd.read_csv(manifest_csv)
        if limit and limit < len(df):
            df = (
                pd.concat([g.sample(limit // 2, random_state=seed)
                           for _, g in df.groupby("label")])
                .sample(frac=1, random_state=seed).reset_index(drop=True)
            )
        self.df = df
        self.root = Path(root)
        self.perturb = perturb
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        try:
            with Image.open(self.root / row["path"]) as im:
                img = self.perturb(im.convert("RGB"))
            return self.transform(img), int(row["label"])
        except Exception:
            return torch.zeros(3, 224, 224), int(row["label"])


# ---------------------------------------------------------------------------
# FGSM
# ---------------------------------------------------------------------------

def fgsm_evaluate(model, loader, epsilon, device="cpu", use_true_labels=False):
    """
    Fast Gradient Sign Method.

    epsilon is expressed in [0,1] pixel space. Inputs reaching the model are
    already ImageNet-normalised, so the perturbation must be divided by the
    per-channel standard deviation to keep epsilon interpretable as a fraction
    of the pixel range.

    LABEL CHOICE MATTERS - this is not a detail.

    `use_true_labels=True` computes the gradient of the loss with respect to
    the GROUND-TRUTH label and steps away from it. On a binary task that
    guarantees every sample is pushed toward the opposite class, so accuracy
    collapses to ~0 and ROC-AUC to ~0 (perfectly inverted ranking) at even tiny
    epsilon. This is the well-known "label leaking" effect: the attack is
    handed the answer, so the numbers measure the construction of the
    experiment rather than the robustness of the model.

    `use_true_labels=False` (the default) uses the model's OWN PREDICTED label.
    This is the realistic threat model - an attacker probing a deployed
    detector sees its output, not the ground truth - and it is what the
    reported figures should be based on.

    Both are worth running: the difference between them is itself a finding
    about how easily adversarial robustness numbers can be overstated.
    """
    model.eval()
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    criterion = nn.CrossEntropyLoss()

    probs, ys = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if use_true_labels:
            target = y
        else:
            # Attacker only observes the model's own decision.
            with torch.no_grad():
                target = model(x).argmax(dim=1)

        x.requires_grad_(True)
        loss = criterion(model(x), target)
        model.zero_grad()
        loss.backward()

        x_adv = x + (epsilon / std) * x.grad.sign()
        with torch.no_grad():
            p = torch.softmax(model(x_adv), dim=1)[:, 1]
        probs.append(p.cpu().numpy())
        ys.append(y.cpu().numpy())

    return np.concatenate(probs), np.concatenate(ys)


# ---------------------------------------------------------------------------

def score(y, prob):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
    }


def plot_degradation(df, name, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = df[df.family == "clean"].accuracy.iloc[0]
    fams = [f for f in df.family.unique() if f != "clean"]

    # Size the grid to the data. A hardcoded 2x3 silently dropped families
    # once FGSM was split into predicted-label and true-label variants.
    ncols = 3
    nrows = -(-len(fams) // ncols)  # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for ax, fam in zip(axes, fams):
        d = df[df.family == fam].sort_values("magnitude")
        ax.plot(range(len(d)), d.accuracy, "o-", lw=2, label="accuracy")
        ax.plot(range(len(d)), d.roc_auc, "s--", lw=1.5, alpha=0.7, label="ROC-AUC")
        ax.axhline(clean, ls=":", c="green", lw=1.4, label=f"clean ({clean:.3f})")
        ax.axhline(0.5, ls="--", c="red", lw=1, label="chance")
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([l.split(maxsplit=1)[-1] for l in d.label],
                           rotation=30, fontsize=8)
        ax.set_title(fam, fontsize=11)
        ax.set_ylim(0.4, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[len(fams):]:
        ax.axis("off")

    fig.suptitle(f"{name} - robustness to input degradation", fontsize=13)
    plt.tight_layout()
    plt.savefig(figdir / f"fig_7_01_robustness_{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Summary: worst-case drop per family
    fig, ax = plt.subplots(figsize=(8, 4.5))
    worst = (df[df.family != "clean"].groupby("family").accuracy.min()
             .sort_values())
    colours = ["#C44E52" if v < 0.6 else "#DD8452" if v < 0.8 else "#4C72B0"
               for v in worst.values]
    ax.barh(worst.index, clean - worst.values, color=colours)
    ax.set_xlabel("worst-case accuracy drop from clean (points)")
    ax.set_title(f"{name} - which perturbations hurt most")
    for i, v in enumerate(clean - worst.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(figdir / f"fig_7_02_worst_case_{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Robustness evaluation")
    ap.add_argument("--model", default="mobilenet_v2",
                    choices=["mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--split", default="test", help="which split to perturb")
    ap.add_argument("--limit", type=int, default=None,
                    help="subsample the split (faster, less precise)")
    ap.add_argument("--skip-fgsm", action="store_true")
    args = ap.parse_args()

    config = load_config()
    set_seed(config["project"]["seed"])
    setup_threads(config)
    ensure_dirs(config)
    device = config["compute"]["device"]
    img_size = config["subset"]["image_size"]

    ckpt = resolve(config["paths"]["models"]) / f"{args.model}_best.pt"
    if not ckpt.exists():
        log.error("No checkpoint at %s. Train first.", ckpt)
        sys.exit(1)

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model = build_model(args.model, pretrained=False).to(device)
    model.load_state_dict(ck["model_state"])
    transform = build_transforms(img_size, config["augmentation"], train=False)
    manifest = resolve("data/manifests") / f"subset_{args.split}.csv"

    perts = build_perturbations(config)
    log.info("=" * 62)
    log.info("ROBUSTNESS: %s on split '%s'", args.model, args.split)
    log.info("=" * 62)
    log.info("%d perturbation settings", len(perts))

    import time
    rows, t0 = [], time.perf_counter()

    for family, label, magnitude, fn in perts:
        ds = PerturbedDataset(manifest, ROOT, fn, transform, limit=args.limit,
                              seed=config["project"]["seed"])
        loader = DataLoader(ds, batch_size=config["training"]["batch_size"],
                            shuffle=False,
                            num_workers=config["compute"]["dataloader_workers"])
        probs, ys = [], []
        model.eval()
        with torch.no_grad():
            for x, y in loader:
                p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
                probs.append(p.cpu().numpy()); ys.append(y.numpy())
        prob, y = np.concatenate(probs), np.concatenate(ys)

        m = score(y, prob)
        m.update({"family": family, "label": label, "magnitude": magnitude})
        rows.append(m)
        log.info("  %-20s acc=%.4f  f1=%.4f  auc=%.4f",
                 label, m["accuracy"], m["f1"], m["roc_auc"])

    if not args.skip_fgsm:
        n = config["robustness"]["fgsm_subset_size"]
        # Run both label regimes. The realistic one (predicted labels) is what
        # gets reported; the true-label one is included to quantify how much
        # label leaking inflates apparent attack strength.
        for use_true, fam, tag in (
            (False, "fgsm", "FGSM(pred)"),
            (True, "fgsm_trueLabel", "FGSM(true)"),
        ):
            log.info("-" * 62)
            log.info("FGSM white-box - %s labels%s", "GROUND-TRUTH" if use_true
                     else "model-predicted",
                     " [label leaking, upper bound only]" if use_true else
                     " [realistic threat model]")
            for eps in config["robustness"]["fgsm_epsilon"]:
                ds = PerturbedDataset(manifest, ROOT, Identity(), transform,
                                      limit=n, seed=config["project"]["seed"])
                loader = DataLoader(ds, batch_size=config["training"]["batch_size"],
                                    shuffle=False, num_workers=0)
                prob, y = fgsm_evaluate(model, loader, eps, device,
                                        use_true_labels=use_true)
                m = score(y, prob)
                m.update({"family": fam, "label": f"{tag} eps={eps}",
                          "magnitude": eps})
                rows.append(m)
                log.info("  %-22s acc=%.4f  f1=%.4f  auc=%.4f",
                         m["label"], m["accuracy"], m["f1"], m["roc_auc"])

    df = pd.DataFrame(rows)
    out = resolve(config["paths"]["results"]) / f"robustness_{args.model}.csv"
    df.to_csv(out, index=False)
    plot_degradation(df, args.model, resolve(config["paths"]["figures"]))

    clean = df[df.family == "clean"].accuracy.iloc[0]
    log.info("=" * 62)
    log.info("SUMMARY - worst case per perturbation family (clean = %.4f)", clean)
    log.info("=" * 62)
    worst = df[df.family != "clean"].groupby("family").accuracy.min().sort_values()
    for fam, acc in worst.items():
        log.info("  %-12s worst acc=%.4f   drop=%.3f", fam, acc, clean - acc)

    log.info("")
    log.info("PREDICTION CHECK (recorded in the module docstring before running)")
    log.info("  DESTRUCTIVE  (jpeg, blur, downscale) - information irreversibly lost")
    log.info("  RESCALING    (brightness, contrast)  - spatial structure preserved")
    log.info("  Prediction: destructive should hurt substantially more.")
    dest = worst.reindex(["jpeg", "blur", "downscale"]).dropna()
    resc = worst.reindex(["brightness", "contrast"]).dropna()
    if len(dest) and len(resc):
        d, r = dest.mean(), resc.mean()
        log.info("  mean worst-case accuracy: destructive=%.4f  rescaling=%.4f  (gap %.3f)",
                 d, r, r - d)
        if r - d > 0.05:
            log.info("  => SUPPORTED. Consistent with the high-frequency account "
                     "of the 128->224px resolution gain.")
        elif r - d > 0.02:
            log.info("  => WEAKLY supported. Direction is right but the margin is "
                     "small; do not overstate this in the report.")
        else:
            log.info("  => NOT supported. All families degrade similarly, so the "
                     "high-frequency explanation of the resolution result needs "
                     "revisiting in ch7.")
        log.info("  NOTE: brightness/contrast are imperfect controls - both are "
                 "multiplicative and do alter gradient magnitudes. See docstring.")

    log.info("")
    log.info("Saved: %s  (%s)", out, format_duration(time.perf_counter() - t0))


if __name__ == "__main__":
    main()
