"""
Evaluate a trained CNN in-domain (Dataset A) and cross-dataset (Dataset B).

Mirrors the diagnostics used for the classical baselines so the two are
directly comparable, including the both-polarity threshold analysis that
distinguishes a calibration failure from a representation failure.

The central comparison this produces:

    baseline HOG+LBP : 75.3% in-domain -> 48.9% cross-dataset (AUC 0.399)
    CNN              :      ?          ->        ?

If the CNN also collapses toward chance, the limiting factor is the domain gap
rather than model capacity. If it holds up substantially better, that is a
genuine finding about learned representations and should lead the discussion.

Usage:
    python src/evaluate.py --model mobilenet_v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DeepfakeDataset, build_model, build_transforms, predict_scores  # noqa: E402
from utils import (  # noqa: E402
    ROOT,
    ensure_dirs,
    get_logger,
    load_config,
    resolve,
    set_seed,
    setup_threads,
)

log = get_logger("evaluate")


def metrics(y, prob, threshold=0.5) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )
    pred = (prob >= threshold).astype(int)
    return {
        "n": len(y),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
        "pred_fake_rate": float(pred.mean()),
    }


def threshold_diagnostic(prob, y) -> dict:
    """
    Both-polarity best-threshold search.

    An AUC below 0.5 means the ranking is inverted, so testing only
    'score above threshold => fake' understates the retained signal.
    """
    grid = np.quantile(prob, np.linspace(0.01, 0.99, 99))
    fwd = max(((prob > t).astype(int) == y).mean() for t in grid)
    inv = max(((prob <= t).astype(int) == y).mean() for t in grid)
    best, polarity = (fwd, "normal") if fwd >= inv else (inv, "INVERTED")
    if best < 0.55:
        verdict = "REPRESENTATION failure - no transferable signal"
    elif polarity == "INVERTED":
        verdict = "signal SURVIVES but sign REVERSED"
    else:
        verdict = "CALIBRATION failure - ordering intact, threshold misplaced"
    return {
        "best_normal": fwd, "best_inverted": inv,
        "best": best, "polarity": polarity, "verdict": verdict,
    }


def make_loader(split, config, batch_size=None):
    from torch.utils.data import DataLoader
    ds = DeepfakeDataset(
        resolve("data/manifests") / f"subset_{split}.csv",
        ROOT,
        build_transforms(config["subset"]["image_size"], config["augmentation"], train=False),
    )
    return DataLoader(
        ds,
        batch_size=batch_size or config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["compute"]["dataloader_workers"],
        persistent_workers=config["compute"]["dataloader_workers"] > 0,
    )


def plot_all(name, results, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

    # Confusion matrices
    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 3.8))
    if len(results) == 1:
        axes = [axes]
    for ax, (tag, (prob, y)) in zip(axes, results.items()):
        cm = confusion_matrix(y, (prob >= 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(tag, fontsize=10)
    fig.suptitle(f"{name} - confusion matrices", fontsize=12)
    plt.tight_layout(); plt.savefig(figdir / f"fig_6_cnn_cm_{name}.png", dpi=200,
                                    bbox_inches="tight"); plt.close(fig)

    # ROC + PR
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for tag, (prob, y) in results.items():
        fpr, tpr, _ = roc_curve(y, prob)
        from sklearn.metrics import auc as _auc
        a1.plot(fpr, tpr, lw=1.8, label=f"{tag} (AUC={_auc(fpr, tpr):.3f})")
        pr, rc, _ = precision_recall_curve(y, prob)
        a2.plot(rc, pr, lw=1.8, label=tag)
    a1.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate")
    a1.set_title("ROC"); a1.legend(fontsize=8, loc="lower right")
    a2.axhline(0.5, ls="--", c="k", lw=1, label="chance")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision")
    a2.set_title("Precision-Recall"); a2.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{name} - in-domain vs cross-dataset", fontsize=12)
    plt.tight_layout(); plt.savefig(figdir / f"fig_6_cnn_curves_{name}.png", dpi=200,
                                    bbox_inches="tight"); plt.close(fig)

    # Score distributions
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]
    for ax, (tag, (prob, y)) in zip(axes, results.items()):
        for lbl, c, nm in ((0, "#4C72B0", "Real"), (1, "#C44E52", "Fake")):
            sns.kdeplot(prob[y == lbl], ax=ax, fill=True, alpha=0.4, color=c, label=nm)
        ax.axvline(0.5, ls="--", c="k", lw=1, label="threshold")
        ax.set_xlabel("P(fake)"); ax.set_title(tag, fontsize=10); ax.legend(fontsize=8)
    fig.suptitle(f"{name} - predicted probability distributions", fontsize=12)
    plt.tight_layout(); plt.savefig(figdir / f"fig_6_cnn_scores_{name}.png", dpi=200,
                                    bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained CNN")
    ap.add_argument("--model", default="mobilenet_v2",
                    choices=["mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--weights", default=None, help="override checkpoint path")
    args = ap.parse_args()

    config = load_config()
    set_seed(config["project"]["seed"])
    setup_threads(config)
    ensure_dirs(config)
    device = config["compute"]["device"]

    ckpt_path = Path(args.weights) if args.weights else (
        resolve(config["paths"]["models"]) / f"{args.model}_best.pt"
    )
    if not ckpt_path.exists():
        log.error("No checkpoint at %s. Train first:\n"
                  "  python src/train.py --model %s", ckpt_path, args.model)
        sys.exit(1)

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(args.model, pretrained=False).to(device)
    model.load_state_dict(ck["model_state"])
    log.info("=" * 62)
    log.info("EVALUATING %s", args.model)
    log.info("=" * 62)
    log.info("checkpoint: %s (epoch %s, val_loss %.4f)",
             ckpt_path.name, ck.get("epoch", "?"), ck.get("val_loss", float("nan")))

    rows, raw = [], {}
    for split, tag in (("test", "A (in-domain)"), ("crosstest", "B (cross-dataset)")):
        loader = make_loader(split, config)
        prob, y = predict_scores(model, loader, device)
        raw[tag] = (prob, y)

        m = metrics(y, prob)
        m.update({"model": args.model, "evaluated_on": tag})
        rows.append(m)
        log.info("-" * 62)
        log.info("%s", tag)
        log.info("  acc=%.4f  prec=%.4f  rec=%.4f  f1=%.4f  auc=%.4f  pred-fake=%.1f%%",
                 m["accuracy"], m["precision"], m["recall"], m["f1"],
                 m["roc_auc"], m["pred_fake_rate"] * 100)
        if m["roc_auc"] < 0.47:
            log.warning("  ^ AUC below chance: ranking systematically INVERTED")

        if split == "crosstest":
            d = threshold_diagnostic(prob, y)
            log.info("  best at ANY threshold: %.4f normal / %.4f inverted -> %.4f (%s)",
                     d["best_normal"], d["best_inverted"], d["best"], d["polarity"])
            log.info("  => %s", d["verdict"])
            rows[-1].update({f"diag_{k}": v for k, v in d.items()})

        pd.DataFrame({"prob": prob, "label": y}).to_csv(
            resolve(config["paths"]["results"]) / f"scores_cnn_{args.model}_{split}.csv",
            index=False)

    df = pd.DataFrame(rows)
    out = resolve(config["paths"]["results"]) / f"cnn_metrics_{args.model}.csv"
    df.to_csv(out, index=False)
    plot_all(args.model, raw, resolve(config["paths"]["figures"]))

    drop = rows[0]["accuracy"] - rows[1]["accuracy"]
    log.info("=" * 62)
    log.info("generalisation drop A->B: %.1f percentage points", drop * 100)

    # Direct comparison against the classical baselines
    bpath = resolve(config["paths"]["results"]) / "baseline_metrics.csv"
    if bpath.exists():
        b = pd.read_csv(bpath)
        log.info("")
        log.info("COMPARISON vs classical baselines")
        log.info("%-22s %-20s %8s %8s", "model", "evaluated on", "acc", "auc")
        for _, r in b.iterrows():
            log.info("%-22s %-20s %8.4f %8.4f",
                     r["model"], r["evaluated_on"], r["accuracy"], r["roc_auc"])
        for r in rows:
            log.info("%-22s %-20s %8.4f %8.4f",
                     r["model"], r["evaluated_on"], r["accuracy"], r["roc_auc"])

    log.info("")
    log.info("Saved: %s", out)


if __name__ == "__main__":
    main()
