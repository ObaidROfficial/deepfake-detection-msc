"""
Shared helper functions used across the project.

Keeping these in one place means the seeding and thread setup is identical
in every script, which matters for reproducibility.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# Project root is one level up from src/
ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config.yaml. Defaults to the one in the project root."""
    if path is None:
        path = ROOT / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find config file at {path}. "
            "Are you running from the project root?"
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    """
    Seed everything we can.

    Note: this does not make CPU training bit-for-bit deterministic across
    different machines, because some backend kernels vary. It does make the
    data sampling and splits exactly reproducible, which is the part that
    actually matters for the report.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # too slow on CPU if True
    except ImportError:
        pass


def setup_threads(config: dict[str, Any] | None = None) -> int:
    """
    Cap the number of threads PyTorch uses.

    The machine has 12 cores but we deliberately do not use all of them.
    Saturating every core makes the laptop unresponsive for hours while
    training runs, and the speedup past ~8 threads is marginal anyway
    because of memory bandwidth limits.
    """
    if config is None:
        config = load_config()
    n_threads = config["compute"]["torch_threads"]

    try:
        import torch

        torch.set_num_threads(n_threads)
    except ImportError:
        pass

    # These have to be set before numpy/scipy import to take full effect,
    # so scripts should call setup_threads() early.
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))

    return n_threads


def get_logger(name: str = "deepfake", level: int = logging.INFO) -> logging.Logger:
    """Console logger with a consistent format."""
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


def ensure_dirs(config: dict[str, Any] | None = None) -> None:
    """Create all the output directories listed in config, if missing."""
    if config is None:
        config = load_config()
    for key, rel_path in config["paths"].items():
        (ROOT / rel_path).mkdir(parents=True, exist_ok=True)


def resolve(rel_path: str) -> Path:
    """Turn a config-relative path into an absolute one."""
    return ROOT / rel_path


def format_duration(seconds: float) -> str:
    """Human readable duration. Used a lot in the benchmark output."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} hours"


def count_parameters(model) -> tuple[int, int]:
    """Return (total params, trainable params) for a torch model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
