"""
Classical machine learning baselines.

Two baselines, both deliberately chosen:

  1. Logistic Regression on flattened 32x32 greyscale pixels.
     A deliberately weak floor. If this scores well the task is trivial.

  2. HOG + Local Binary Patterns -> Linear SVM.
     A classical computer vision pipeline. HOG encodes gradient orientation,
     LBP encodes local texture.

Baseline 2 matters more than it looks. The Chapter 5 exploration found that
sharpness (gradient variance) was the strongest single real/fake cue in
Dataset A (AUC 0.611) and also the biggest difference between Dataset A and
Dataset B (AUC 0.822). HOG and LBP are *precisely* gradient and texture
descriptors. So this baseline directly measures how much of the task is
solvable from exactly the signal we expect to be dataset-specific.

The prediction: HOG+LBP should score respectably in-domain and collapse on
Dataset B. If it does, that is strong evidence for the shortcut argument
before we have even trained a CNN.

Both baselines are evaluated on:
  - the in-domain test set (Dataset A)
  - the cross-dataset set (Dataset B, never seen in training)

Usage:
    python src/baselines.py                    # both baselines
    python src/baselines.py --model logreg     # just one
    python src/baselines.py --force            # ignore cached features
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    ROOT,
    ensure_dirs,
    format_duration,
    get_logger,
    load_config,
    resolve,
    set_seed,
    setup_threads,
)

log = get_logger("baselines")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _load_grey(path: Path, size: int) -> np.ndarray:
    """Load an image as a greyscale float array in [0, 1]."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def extract_pixels(paths: list[Path], size: int = 32) -> np.ndarray:
    """Baseline 1 features: raw downscaled pixels, flattened."""
    from tqdm import tqdm

    out = np.zeros((len(paths), size * size), dtype=np.float32)
    for i, p in enumerate(tqdm(paths, desc="pixels", unit="img")):
        try:
            out[i] = _load_grey(p, size).ravel()
        except Exception:
            pass  # leave as zeros; counted in the load-failure report
    return out


def extract_hog_lbp(paths: list[Path], cfg: dict, size: int = 224) -> np.ndarray:
    """
    Baseline 2 features: HOG (gradients) + LBP histogram (texture).

    `size` is passed from config.subset.image_size so the baselines see the
    same input resolution as the CNN. This matters: raising the CNN from 128px
    to 224px gained 10 accuracy points, so comparing a 224px CNN against
    128px baselines would attribute a resolution advantage to architecture.
    """
    from skimage.feature import hog, local_binary_pattern
    from tqdm import tqdm

    ppc = cfg["hog_pixels_per_cell"]
    orient = cfg["hog_orientations"]
    lbp_r = cfg["lbp_radius"]
    lbp_p = cfg["lbp_points"]
    n_lbp_bins = lbp_p + 2  # uniform LBP produces P+2 distinct codes

    feats = None
    for i, p in enumerate(tqdm(paths, desc="hog+lbp", unit="img")):
        try:
            g = _load_grey(p, size)
            h = hog(
                g,
                orientations=orient,
                pixels_per_cell=(ppc, ppc),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
                feature_vector=True,
            )
            # LBP on uint8, not float. The operator thresholds neighbours
            # against the centre pixel; on floats, near-equal pixels flip on
            # floating-point noise, which skimage warns about. Integer input
            # makes the comparison exact. HOG stays on float - it uses
            # gradient magnitudes, where the precision is wanted.
            g8 = (g * 255).astype(np.uint8)
            lbp = local_binary_pattern(g8, lbp_p, lbp_r, method="uniform")
            hist, _ = np.histogram(
                lbp.ravel(), bins=n_lbp_bins, range=(0, n_lbp_bins), density=True
            )
            v = np.concatenate([h, hist]).astype(np.float32)
        except Exception:
            v = None

        if feats is None:
            if v is None:
                continue  # cannot size the array until one image succeeds
            feats = np.zeros((len(paths), v.size), dtype=np.float32)
        if v is not None:
            feats[i] = v

    if feats is None:
        raise RuntimeError("Every image failed to load - check the manifests.")
    return feats


def cached_features(
    kind: str, split: str, paths: list[Path], cfg: dict, force: bool = False,
    image_size: int = 224,
) -> np.ndarray:
    """
    Extract features, caching to .npy so reruns are fast.

    Extraction over 10k images takes minutes, and we rerun this while tuning.
    Cache key includes the feature kind and split name.
    """
    cache_dir = resolve("data/features")
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Resolution is part of the cache key - otherwise a 128px cache would be
    # silently reused after switching the config to 224px.
    cache = cache_dir / f"{kind}_{split}_{image_size}.npy"

    if cache.exists() and not force:
        arr = np.load(cache)
        if arr.shape[0] == len(paths):
            log.info("  cached %s/%s -> %s", kind, split, arr.shape)
            return arr
        log.warning("  cache size mismatch for %s/%s, re-extracting", kind, split)

    t0 = time.perf_counter()
    if kind == "pixels":
        # Pixels stay at 32x32 by design - this is the deliberately weak floor,
        # and flattening 224x224 would give 50k features with no benefit.
        arr = extract_pixels(paths)
    elif kind == "hoglbp":
        arr = extract_hog_lbp(paths, cfg, size=image_size)
    else:
        raise ValueError(kind)
    log.info(
        "  extracted %s/%s -> %s in %s",
        kind, split, arr.shape, format_duration(time.perf_counter() - t0),
    )
    np.save(cache, arr)
    return arr


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(name: str, dataset: str, y_true, y_pred, y_score) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "model": name,
        "evaluated_on": dataset,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def plot_confusion(y_true, y_pred, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(
    name: str, scores: dict[str, tuple[np.ndarray, np.ndarray]], path: Path
) -> None:
    """
    Distribution of decision scores on Dataset A vs Dataset B, split by class.

    This is the diagnostic for *why* cross-dataset performance fails. Two very
    different failures look identical in an accuracy table:

      - Calibration failure: B's scores keep their real/fake ordering but the
        whole distribution slides to one side of the threshold.
      - Representation failure: B's scores are genuinely scrambled, with real
        and fake overlapping completely or ordered backwards.

    The first is fixable by re-thresholding. The second is not, and means the
    learned features carry no transferable information.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, len(scores), figsize=(5.5 * len(scores), 4),
                             sharex=True)
    if len(scores) == 1:
        axes = [axes]

    for ax, (tag, (sc, y)) in zip(axes, scores.items()):
        for lbl, colour, nm in ((0, "#4C72B0", "Real"), (1, "#C44E52", "Fake")):
            sns.kdeplot(sc[y == lbl], ax=ax, fill=True, alpha=0.4,
                        color=colour, label=nm)
        ax.axvline(0 if sc.min() < 0 else 0.5, ls="--", c="k", lw=1,
                   label="decision threshold")
        ax.set_title(tag, fontsize=10)
        ax.set_xlabel("decision score")
        ax.legend(fontsize=8)

    fig.suptitle(f"{name} - decision score distributions", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc(curves: list[tuple[str, np.ndarray, np.ndarray]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for label, fpr, tpr in curves:
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})", lw=1.8)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Baseline ROC curves")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_manifest(split: str, limit: int | None = None, seed: int = 42):
    df = pd.read_csv(resolve("data/manifests") / f"subset_{split}.csv")
    if limit and len(df) > limit:
        # Stratified so the subsample stays balanced
        df = (
            pd.concat([g.sample(limit // 2, random_state=seed)
                       for _, g in df.groupby("label")])
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True)
        )
    paths = [ROOT / p for p in df["path"]]
    return paths, df["label"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Classical ML baselines")
    ap.add_argument("--model", choices=["logreg", "hoglbp", "all"], default="all")
    ap.add_argument("--force", action="store_true", help="ignore cached features")
    args = ap.parse_args()

    config = load_config()
    seed = config["project"]["seed"]
    set_seed(seed)
    setup_threads(config)
    ensure_dirs(config)

    bcfg = config["models"]["baselines"]
    n_train = bcfg["train_size"]
    specs = {m["name"]: m for m in bcfg["models"]}

    log.info("=" * 62)
    log.info("CLASSICAL BASELINES")
    log.info("=" * 62)
    log.info("Train subsample: %d images", n_train)
    if args.model in ("hoglbp", "all"):
        log.info(
            "Note: HOG+LBP extraction runs over ~20,000 images and LBP is the "
            "slow part. Expect roughly 15-30 minutes on first run. Features "
            "are cached to data/features/, so reruns are near-instant."
        )

    log.info("Loading manifests ...")
    tr_paths, y_tr = load_manifest("train", n_train, seed)
    te_paths, y_te = load_manifest("test", seed=seed)
    cx_paths, y_cx = load_manifest("crosstest", seed=seed)
    log.info(
        "  train=%d  test(A)=%d  crosstest(B)=%d", len(tr_paths), len(te_paths), len(cx_paths)
    )

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    figdir = resolve(config["paths"]["figures"])
    results, roc_curves = [], []

    jobs = []
    if args.model in ("logreg", "all"):
        jobs.append(("logistic_regression", "pixels"))
    if args.model in ("hoglbp", "all"):
        jobs.append(("hog_lbp_svm", "hoglbp"))

    for name, kind in jobs:
        spec = specs[name]
        log.info("-" * 62)
        log.info("%s  (%s)", name, spec["description"])

        img_size = config["subset"]["image_size"]
        X_tr = cached_features(kind, "train", tr_paths, spec, args.force, img_size)
        X_te = cached_features(kind, "test", te_paths, spec, args.force, img_size)
        X_cx = cached_features(kind, "crosstest", cx_paths, spec, args.force, img_size)

        if name == "logistic_regression":
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=spec["max_iter"], random_state=seed),
            )
        else:
            # LinearSVC has no predict_proba, so decision_function feeds the AUC.
            # (Pipeline only exposes predict_proba if the final estimator has
            # it, so the hasattr check below resolves correctly.)
            # max_iter raised from the default 1000: with ~1,790 standardised
            # features the solver often hits the cap and warns otherwise.
            clf = make_pipeline(
                StandardScaler(), LinearSVC(random_state=seed, max_iter=5000)
            )

        t0 = time.perf_counter()
        clf.fit(X_tr, y_tr)
        log.info("  fitted in %s", format_duration(time.perf_counter() - t0))

        def score(X):
            if hasattr(clf, "predict_proba"):
                return clf.predict_proba(X)[:, 1]
            return clf.decision_function(X)

        score_dists = {}
        for tag, X, y in (("A (in-domain)", X_te, y_te), ("B (cross-dataset)", X_cx, y_cx)):
            pred, sc = clf.predict(X), score(X)
            score_dists[tag] = (sc, y)
            r = evaluate(name, tag, y, pred, sc)
            r["pred_fake_rate"] = float((pred == 1).mean())
            results.append(r)
            log.info(
                "  %-18s acc=%.4f  f1=%.4f  auc=%.4f  predicted-fake=%.1f%%",
                tag, r["accuracy"], r["f1"], r["roc_auc"], r["pred_fake_rate"] * 100,
            )
            # An AUC meaningfully below 0.5 means the ranking is inverted, not
            # merely uninformative. Worth calling out rather than burying in a
            # table, because it says something specific about the failure mode.
            if r["roc_auc"] < 0.47:
                log.warning(
                    "     ^ AUC below chance: ranking is systematically "
                    "INVERTED on this set, not just uninformative."
                )
            fpr, tpr, _ = roc_curve(y, sc)
            roc_curves.append((f"{name} / {tag}", fpr, tpr))
            plot_confusion(
                y, pred, f"{name}\n{tag}",
                figdir / f"fig_6_cm_{name}_{tag.split()[0]}.png",
            )

        plot_score_distributions(
            name, score_dists, figdir / f"fig_6_scores_{name}.png"
        )

        drop = (
            [r for r in results if r["model"] == name and "in-domain" in r["evaluated_on"]][0]["accuracy"]
            - [r for r in results if r["model"] == name and "cross" in r["evaluated_on"]][0]["accuracy"]
        )
        log.info("  >> generalisation drop A->B: %.1f percentage points", drop * 100)

        # Separate a calibration failure from a representation failure.
        #
        # Both polarities must be tested. An AUC below 0.50 means the ranking
        # is inverted, so the rule "score ABOVE threshold => fake" is the wrong
        # way round and testing only that direction understates how much
        # information the representation actually retains.
        sc_b, y_b = score_dists["B (cross-dataset)"]
        thr = 0.5 if sc_b.min() >= 0 else 0.0
        grid = np.quantile(sc_b, np.linspace(0.01, 0.99, 99))

        best_fwd = max(((sc_b > t).astype(int) == y_b).mean() for t in grid)
        best_inv = max(((sc_b <= t).astype(int) == y_b).mean() for t in grid)
        best, polarity = (
            (best_fwd, "normal") if best_fwd >= best_inv else (best_inv, "INVERTED")
        )

        log.info(
            "  >> Dataset B: %.1f%% of scores sit on one side of the threshold",
            100 * max((sc_b > thr).mean(), (sc_b <= thr).mean()),
        )
        log.info(
            "  >> Best accuracy at ANY threshold: %.4f normal / %.4f inverted "
            "-> %.4f (%s polarity)",
            best_fwd, best_inv, best, polarity,
        )
        if best < 0.55:
            log.info(
                "     => REPRESENTATION failure: no threshold in either "
                "direction recovers usable accuracy. Features carry no "
                "transferable signal."
            )
        elif polarity == "INVERTED":
            log.info(
                "     => Signal SURVIVES but with REVERSED sign. The features "
                "transfer; their relationship to the label does not."
            )
        else:
            log.info(
                "     => CALIBRATION failure: ordering is intact, only the "
                "threshold is misplaced. Re-thresholding would recover this."
            )

        # Persist raw scores so this can be re-analysed without refitting.
        pd.DataFrame(
            {"score": sc_b, "label": y_b, "model": name, "dataset": "B"}
        ).to_csv(
            resolve(config["paths"]["results"]) / f"scores_{name}_B.csv", index=False
        )

    df = pd.DataFrame(results)
    out = resolve(config["paths"]["results"]) / "baseline_metrics.csv"
    df.to_csv(out, index=False)
    plot_roc(roc_curves, figdir / "fig_6_01_baseline_roc.png")

    log.info("=" * 62)
    log.info("SUMMARY")
    log.info("=" * 62)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    log.info("")
    log.info("Saved metrics: %s", out)
    log.info("Saved figures: %s", figdir)


if __name__ == "__main__":
    main()
