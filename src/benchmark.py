"""
CPU training speed benchmark.

Run this BEFORE committing to the subset sizes in config.yaml.

The whole project plan rests on an assumption: that MobileNetV2 and
EfficientNet-B0 can be trained on 20,000 images at 128px on a CPU-only
laptop within a sensible number of hours. This script tests that assumption
on synthetic data (no dataset download needed) and extrapolates to a full
training run.

If the projected time is unacceptable, reduce in this order:
    1. subset.train_size  20000 -> 12000
    2. subset.image_size  128   -> 96
    3. drop efficientnet_b0, keep mobilenet_v2 only

Usage:
    python src/benchmark.py
    python src/benchmark.py --images 500 --models mobilenet_v2
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    count_parameters,
    format_duration,
    get_logger,
    load_config,
    resolve,
    set_seed,
    setup_threads,
)

log = get_logger("benchmark")


def build_model(name: str, pretrained: bool = False):
    """
    Build a model by name.

    pretrained=False for benchmarking so we do not download ~14MB of weights
    just to time a forward pass. Real training uses pretrained=True.
    """
    import torch.nn as nn
    import torchvision.models as tvm

    if name == "mobilenet_v2":
        m = tvm.mobilenet_v2(weights="DEFAULT" if pretrained else None)
        m.classifier[1] = nn.Linear(m.last_channel, 2)
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m


def benchmark_model(
    name: str,
    n_images: int,
    image_size: int,
    batch_size: int,
    config: dict,
) -> dict:
    """
    Time forward+backward passes on synthetic data.

    Synthetic rather than real images because this measures compute, not I/O,
    and it means the benchmark runs before any dataset is downloaded.
    Real training will be somewhat slower due to JPEG decoding and
    augmentation, so we apply a correction factor at the end.
    """
    import torch
    import torch.nn as nn

    log.info("-" * 62)
    log.info("Model: %s", name)

    model = build_model(name, pretrained=False)
    total, trainable = count_parameters(model)
    log.info("Parameters: %s total, %s trainable", f"{total:,}", f"{trainable:,}")

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_batches = max(1, n_images // batch_size)

    # Warm-up. The first couple of batches are unrepresentative because of
    # lazy allocation and thread pool spin-up.
    warmup = min(2, n_batches)
    x = torch.randn(batch_size, 3, image_size, image_size)
    y = torch.randint(0, 2, (batch_size,))
    for _ in range(warmup):
        optimiser.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimiser.step()

    # Timed run
    log.info(
        "Timing %d batches of %d at %dpx ...", n_batches, batch_size, image_size
    )
    start = time.perf_counter()
    for _ in range(n_batches):
        x = torch.randn(batch_size, 3, image_size, image_size)
        y = torch.randint(0, 2, (batch_size,))
        optimiser.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimiser.step()
    elapsed = time.perf_counter() - start

    imgs_done = n_batches * batch_size
    per_image = elapsed / imgs_done

    # Inference speed matters too - the robustness suite does a lot of it
    model.eval()
    with torch.no_grad():
        x = torch.randn(batch_size, 3, image_size, image_size)
        t0 = time.perf_counter()
        for _ in range(max(1, n_batches // 2)):
            model(x)
        infer_elapsed = time.perf_counter() - t0
    infer_per_image = infer_elapsed / (max(1, n_batches // 2) * batch_size)

    log.info(
        "Train: %s for %d images  =  %.1f img/sec  (%.1f ms/img)",
        format_duration(elapsed),
        imgs_done,
        1 / per_image,
        per_image * 1000,
    )
    log.info(
        "Infer: %.1f img/sec  (%.1f ms/img)",
        1 / infer_per_image,
        infer_per_image * 1000,
    )

    return {
        "model": name,
        "params": total,
        "train_sec_per_image": per_image,
        "infer_sec_per_image": infer_per_image,
    }


def project_full_run(result: dict, config: dict) -> None:
    """Extrapolate the benchmark to the real training schedule."""
    sub = config["subset"]
    tr = config["training"]

    n_train = sub["train_size"]
    n_val = sub["val_size"]
    epochs = tr["epochs_frozen"] + tr["epochs_finetune"]

    # Real data is slower than synthetic tensors: JPEG decode, resize,
    # augmentation, and DataLoader overhead. 1.35x is a conservative
    # allowance based on typical CPU pipelines. The Week 1 real run will
    # tell us the true factor.
    IO_FACTOR = 1.35

    # Stage 1 (frozen backbone) is cheaper because most layers need no
    # gradients. Roughly 55% of full backward cost for these architectures.
    frozen_cost = result["train_sec_per_image"] * 0.55
    full_cost = result["train_sec_per_image"]

    train_s = (
        tr["epochs_frozen"] * n_train * frozen_cost
        + tr["epochs_finetune"] * n_train * full_cost
    ) * IO_FACTOR
    val_s = epochs * n_val * result["infer_sec_per_image"] * IO_FACTOR
    total_s = train_s + val_s

    log.info("")
    log.info("  Projected full training run for %s:", result["model"])
    log.info("    %d train images x %d epochs", n_train, epochs)
    log.info("    training  ~%s", format_duration(train_s))
    log.info("    validation ~%s", format_duration(val_s))
    log.info("    TOTAL     ~%s", format_duration(total_s))

    if total_s > 6 * 3600:
        log.warning(
            "    >6 hours. Recommend reducing train_size to 12000 "
            "or image_size to 96 in config.yaml."
        )
    elif total_s > 4 * 3600:
        log.warning("    >4 hours. Workable but consider a smaller subset.")
    else:
        log.info("    Acceptable. Proceed with config as written.")

    # Robustness suite cost - it re-scores the test set many times over
    rb = config["robustness"]
    n_perturbations = (
        len(rb["jpeg_quality"])
        + len(rb["gaussian_noise_sigma"])
        + len(rb["gaussian_blur_kernel"])
        + len(rb["downscale_factors"])
        + len(rb["brightness_shift"])
        + len(rb["contrast_shift"])
        + 1  # clean baseline
    )
    rob_s = (
        n_perturbations
        * sub["test_size"]
        * result["infer_sec_per_image"]
        * IO_FACTOR
    )
    # FGSM needs a backward pass per image, so it costs roughly training rate
    fgsm_s = (
        len(rb["fgsm_epsilon"])
        * rb["fgsm_subset_size"]
        * result["train_sec_per_image"]
        * IO_FACTOR
    )
    log.info(
        "    robustness suite (%d perturbations) ~%s",
        n_perturbations,
        format_duration(rob_s),
    )
    log.info("    FGSM adversarial ~%s", format_duration(fgsm_s))


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU training benchmark")
    ap.add_argument("--images", type=int, default=500, help="Images to time")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["mobilenet_v2", "efficientnet_b0"],
        help="Which models to benchmark",
    )
    ap.add_argument("--image-size", type=int, default=None)
    args = ap.parse_args()

    config = load_config()
    set_seed(config["project"]["seed"])
    n_threads = setup_threads(config)

    try:
        import torch
    except ImportError:
        log.error(
            "PyTorch is not installed. See SETUP.md - install the CPU wheel:\n"
            "  pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cpu"
        )
        sys.exit(1)

    image_size = args.image_size or config["subset"]["image_size"]
    batch_size = config["training"]["batch_size"]

    log.info("=" * 62)
    log.info("CPU BENCHMARK")
    log.info("=" * 62)
    log.info("Machine:      %s", platform.processor() or platform.machine())
    log.info("Platform:     %s %s", platform.system(), platform.release())
    log.info("Python:       %s", platform.python_version())
    log.info("PyTorch:      %s", torch.__version__)
    log.info("CUDA:         %s", "available" if torch.cuda.is_available() else "not available (expected)")
    log.info("Threads:      %d", n_threads)
    log.info("Image size:   %dpx", image_size)
    log.info("Batch size:   %d", batch_size)

    results = []
    for name in args.models:
        try:
            r = benchmark_model(name, args.images, image_size, batch_size, config)
            results.append(r)
            project_full_run(r, config)
        except Exception as e:  # noqa: BLE001
            log.error("Benchmark failed for %s: %s", name, e)

    if results:
        log.info("=" * 62)
        log.info("SUMMARY")
        log.info("=" * 62)
        for r in results:
            log.info(
                "%-18s %s params | %5.1f img/s train | %5.1f img/s infer",
                r["model"],
                f"{r['params'] / 1e6:.1f}M",
                1 / r["train_sec_per_image"],
                1 / r["infer_sec_per_image"],
            )

        out = resolve(config["paths"]["results"]) / "benchmark.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        pd.DataFrame(results).to_csv(out, index=False)
        log.info("Saved: %s", out)
        log.info("")
        log.info("Send these numbers back before we lock the subset sizes.")


if __name__ == "__main__":
    main()
