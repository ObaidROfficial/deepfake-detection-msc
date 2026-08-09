"""
CNN training: two-stage transfer learning on CPU.

Stage 1 - backbone frozen, train the classifier head only (fast, ~3 epochs).
Stage 2 - unfreeze the final feature block, fine-tune at a low learning rate.

Designed around the constraint that this runs for hours on a laptop that is
also being used for other work:

  * Checkpoints every epoch. An interrupted run resumes with --resume rather
    than starting over.
  * Threads capped below the core count so the machine stays usable.
  * Early stopping on validation loss.
  * Best-by-validation weights saved separately from the last epoch.

Run this in a terminal, not a notebook. Notebook kernels drop on long runs.

Usage:
    python src/train.py --model mobilenet_v2
    python src/train.py --model mobilenet_v2 --resume
    python src/train.py --model efficientnet_b0
    python src/train.py --model mobilenet_v2 --limit-train 2000   # smoke test

WINDOWS NOTE: if DataLoader workers hang or crash, set
compute.dataloader_workers to 0 in config.yaml. Windows uses spawn rather than
fork, which occasionally misbehaves with worker processes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import (  # noqa: E402
    DeepfakeDataset,
    build_model,
    build_transforms,
    set_backbone_frozen,
)
from utils import (  # noqa: E402
    ROOT,
    count_parameters,
    ensure_dirs,
    format_duration,
    get_logger,
    load_config,
    resolve,
    set_seed,
    setup_threads,
)

log = get_logger("train")


def make_loader(split, config, train, limit=None, shuffle=None):
    aug = config["augmentation"]
    size = config["subset"]["image_size"]
    ds = DeepfakeDataset(
        resolve("data/manifests") / f"subset_{split}.csv",
        ROOT,
        build_transforms(size, aug, train=train),
    )
    if limit and limit < len(ds):
        # Deterministic stratified-ish subset for smoke tests
        rng = np.random.default_rng(config["project"]["seed"])
        idx = rng.choice(len(ds), size=limit, replace=False)
        ds = Subset(ds, idx.tolist())

    return DataLoader(
        ds,
        batch_size=config["training"]["batch_size"],
        shuffle=train if shuffle is None else shuffle,
        num_workers=config["compute"]["dataloader_workers"],
        pin_memory=config["compute"]["pin_memory"],
        persistent_workers=config["compute"]["dataloader_workers"] > 0,
    )


def run_epoch(model, loader, criterion, optimiser=None, device="cpu", desc=""):
    """One pass. Pass optimiser=None for evaluation."""
    from tqdm import tqdm

    training = optimiser is not None
    model.train() if training else model.eval()

    total_loss, correct, n = 0.0, 0, 0
    bar = tqdm(loader, desc=desc, unit="batch", leave=False)

    with torch.set_grad_enabled(training):
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            if training:
                optimiser.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimiser.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            correct += (logits.argmax(1) == y).sum().item()
            n += bs
            bar.set_postfix(loss=f"{total_loss/n:.4f}", acc=f"{correct/n:.4f}")

    return total_loss / max(n, 1), correct / max(n, 1)


def save_checkpoint(path, model, optimiser, epoch, stage, history, best_val):
    torch.save(
        {
            "epoch": epoch,
            "stage": stage,
            "model_state": model.state_dict(),
            "optimiser_state": optimiser.state_dict() if optimiser else None,
            "history": history,
            "best_val": best_val,
        },
        path,
    )


def plot_curves(history, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (a, b, title) in zip(
        axes,
        [("train_loss", "val_loss", "Loss"), ("train_acc", "val_acc", "Accuracy")],
    ):
        ax.plot(h["epoch"], h[a], "o-", label="train", lw=1.8)
        ax.plot(h["epoch"], h[b], "o-", label="validation", lw=1.8)
        # Mark where stage 2 (fine-tuning) begins
        s2 = h[h["stage"] == 2]
        if len(s2):
            ax.axvline(s2["epoch"].iloc[0] - 0.5, ls="--", c="grey", lw=1)
            ax.text(s2["epoch"].iloc[0] - 0.45, ax.get_ylim()[1] * 0.97,
                    " fine-tune", fontsize=8, va="top", color="grey")
        ax.set_xlabel("epoch")
        ax.set_title(f"{name} - {title}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a CNN deepfake detector")
    ap.add_argument("--model", default="mobilenet_v2",
                    choices=["mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--resume", action="store_true",
                    help="resume from the last checkpoint")
    ap.add_argument("--limit-train", type=int, default=None,
                    help="use only N training images (for a quick smoke test)")
    args = ap.parse_args()

    config = load_config()
    seed = config["project"]["seed"]
    set_seed(seed)
    n_threads = setup_threads(config)
    ensure_dirs(config)

    device = config["compute"]["device"]
    tcfg = config["training"]
    mdir = resolve(config["paths"]["models"])
    ckpt_path = mdir / f"{args.model}_last.pt"
    best_path = mdir / f"{args.model}_best.pt"

    log.info("=" * 62)
    log.info("TRAINING: %s", args.model)
    log.info("=" * 62)
    log.info("device=%s threads=%d batch=%d image=%dpx",
             device, n_threads, tcfg["batch_size"], config["subset"]["image_size"])

    train_loader = make_loader("train", config, True, limit=args.limit_train)
    val_loader = make_loader("val", config, False)
    log.info("train batches=%d  val batches=%d", len(train_loader), len(val_loader))

    model = build_model(args.model, pretrained=True).to(device)
    total, _ = count_parameters(model)
    log.info("parameters: %s", f"{total:,}")

    criterion = nn.CrossEntropyLoss()
    history, best_val, start_epoch, start_stage = [], float("inf"), 1, 1

    if args.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        history = ck["history"]
        best_val = ck["best_val"]
        start_epoch = ck["epoch"] + 1
        start_stage = ck["stage"]
        log.info("Resumed from epoch %d (stage %d)", ck["epoch"], start_stage)

    n_frozen = tcfg["epochs_frozen"]
    n_ft = tcfg["epochs_finetune"]
    total_epochs = n_frozen + n_ft
    patience, since_improve = tcfg["early_stopping_patience"], 0
    t_start = time.perf_counter()

    for epoch in range(start_epoch, total_epochs + 1):
        stage = 1 if epoch <= n_frozen else 2

        # (Re)configure trainable parameters and optimiser at stage boundaries
        if epoch == start_epoch or (stage == 2 and epoch == n_frozen + 1):
            # Per-model override takes precedence over the global default.
            # The two backbones have different block structures, so matching
            # the integer would not match adaptation capacity - see the note
            # in config.yaml under models.cnn.
            per_model = next(
                (m.get("unfreeze_blocks") for m in config["models"]["cnn"]
                 if m["name"] == args.model), None
            )
            n_unfreeze = per_model if per_model is not None else tcfg.get("unfreeze_blocks", 1)
            set_backbone_frozen(
                model, args.model, frozen=(stage == 1),
                unfreeze_blocks=n_unfreeze,
            )
            _, trainable = count_parameters(model)
            lr = tcfg["lr_frozen"] if stage == 1 else tcfg["lr_finetune"]
            optimiser = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr, weight_decay=tcfg["weight_decay"],
            )
            log.info("-" * 62)
            log.info("STAGE %d | lr=%g | trainable params=%s (%.1f%% of %s)%s",
                     stage, lr, f"{trainable:,}",
                     100 * trainable / total, f"{total:,}",
                     f" | unfreeze_blocks={n_unfreeze}" if stage == 2 else "")

        t0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimiser, device,
            f"e{epoch}/{total_epochs} train")
        va_loss, va_acc = run_epoch(
            model, val_loader, criterion, None, device,
            f"e{epoch}/{total_epochs} val")
        dt = time.perf_counter() - t0

        history.append({
            "epoch": epoch, "stage": stage,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
            "seconds": dt,
        })
        log.info(
            "epoch %2d/%d [s%d] train loss=%.4f acc=%.4f | val loss=%.4f acc=%.4f | %s",
            epoch, total_epochs, stage, tr_loss, tr_acc, va_loss, va_acc,
            format_duration(dt),
        )

        if tcfg["checkpoint_every_epoch"]:
            save_checkpoint(ckpt_path, model, optimiser, epoch, stage, history, best_val)

        if va_loss < best_val:
            best_val = va_loss
            since_improve = 0
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch, "val_loss": va_loss, "val_acc": va_acc,
                        "model_name": args.model}, best_path)
            log.info("   ^ best so far, saved to %s", best_path.name)
        else:
            since_improve += 1
            if since_improve >= patience:
                log.info("Early stopping: no improvement in %d epochs.", patience)
                break

    elapsed = time.perf_counter() - t_start
    log.info("=" * 62)
    log.info("Finished in %s", format_duration(elapsed))
    log.info("Best validation loss: %.4f", best_val)

    hist_path = resolve(config["paths"]["results"]) / f"history_{args.model}.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    plot_curves(history, args.model,
                resolve(config["paths"]["figures"]) / f"fig_6_curves_{args.model}.png")
    log.info("Saved history: %s", hist_path)
    log.info("Best weights:  %s", best_path)
    log.info("")
    log.info("Next: python src/evaluate.py --model %s", args.model)


if __name__ == "__main__":
    main()
