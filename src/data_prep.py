"""
Dataset download, inspection and subsetting.

Two datasets are used:

  Dataset A (xhlulu/140k-real-and-fake-faces)
      Training, validation and in-domain testing.
      Real  = Flickr / FFHQ photographs
      Fake  = StyleGAN synthesised faces (1M Fake Faces)

  Dataset B (manjilkarki/deepfake-and-real-images)
      Cross-dataset evaluation only. Never used for training.
      Derived from OpenForensics - in-the-wild face forgeries with varied
      pose, lighting and background.

The point of using two is that they are produced by different methods. A model
that scores 97% on A and 60% on B has learned StyleGAN artefacts rather than
a general notion of facial forgery. That gap is a finding, not a failure.

Usage:
    python src/data_prep.py --step download
    python src/data_prep.py --step inspect
    python src/data_prep.py --step subset
    python src/data_prep.py --step all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, ensure_dirs, get_logger, load_config, resolve, set_seed  # noqa: E402

log = get_logger("data_prep")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Folder names that indicate class. Both datasets use some variant of these.
REAL_TOKENS = {"real", "reals", "authentic", "original", "live"}
FAKE_TOKENS = {"fake", "fakes", "deepfake", "manipulated", "synthetic", "spoof"}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def check_kaggle_credentials() -> bool:
    """
    Verify kaggle.json is present before attempting any download.

    Fails loudly with instructions rather than throwing a confusing
    authentication error three minutes into a download.
    """
    candidates = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path(os.environ["KAGGLE_CONFIG_DIR"]) / "kaggle.json"
        if "KAGGLE_CONFIG_DIR" in os.environ
        else None,
    ]
    for c in candidates:
        if c and c.exists():
            log.info("Found Kaggle credentials at %s", c)
            return True

    log.error(
        "\n"
        "  Kaggle credentials not found.\n"
        "\n"
        "  1. Sign in at kaggle.com\n"
        "  2. Avatar (top right) -> Settings\n"
        "  3. Scroll to 'API' -> 'Create New Token'\n"
        "  4. Move the downloaded kaggle.json to:\n"
        "         %s\n"
        "\n"
        "  You must also visit each dataset page once and click\n"
        "  'I Understand and Accept' before the API will serve it.\n"
        "  See SETUP.md for the links.\n",
        Path.home() / ".kaggle" / "kaggle.json",
    )
    return False


def download_dataset(slug: str, dest: Path) -> bool:
    """Download and unzip a Kaggle dataset. Skips if already present."""
    dest.mkdir(parents=True, exist_ok=True)

    # Crude but effective check: if there are already images in there, skip.
    existing = sum(1 for _ in dest.rglob("*") if _.suffix.lower() in IMAGE_EXTS)
    if existing > 1000:
        log.info("%s already has %d images, skipping download.", dest.name, existing)
        return True

    log.info("Downloading %s -> %s (this will take a while)", slug, dest)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        log.error("The 'kaggle' command is not on PATH. Did you pip install kaggle?")
        return False
    except subprocess.CalledProcessError as e:
        log.error(
            "Kaggle download failed (exit %s). Most common cause: you have not "
            "accepted the dataset terms on the website yet. Open the dataset page "
            "and click 'I Understand and Accept', then rerun.",
            e.returncode,
        )
        return False

    log.info("Download complete: %s", slug)
    return True


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------

def infer_label(path: Path) -> int | None:
    """
    Work out real(0) / fake(1) from the folder path.

    Both datasets organise images into class folders but do not agree on
    naming, so we check every path component against known tokens.
    Returns None if we cannot tell, and those files get reported rather
    than silently mislabelled.
    """
    for part in reversed(path.parts):
        p = part.strip().lower().replace("-", "_").replace(" ", "_")
        if p in REAL_TOKENS:
            return 0
        if p in FAKE_TOKENS:
            return 1
    return None


def scan_dataset(root: Path) -> pd.DataFrame:
    """Walk a dataset folder and build a dataframe of every labelled image."""
    if not root.exists():
        log.warning("Path does not exist: %s", root)
        return pd.DataFrame(columns=["path", "label", "split_hint"])

    rows, unlabelled = [], []
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    log.info("Scanning %d image files under %s", len(files), root.name)

    for p in tqdm(files, desc=f"scan {root.name}", unit="img"):
        label = infer_label(p)
        if label is None:
            unlabelled.append(p)
            continue
        # Preserve any train/valid/test hint the dataset author baked in
        hint = ""
        for part in p.parts:
            pl = part.lower()
            if pl in {"train", "training"}:
                hint = "train"
            elif pl in {"valid", "validation", "val"}:
                hint = "val"
            elif pl in {"test", "testing"}:
                hint = "test"
        # Store paths RELATIVE to the project root. Absolute paths would bake
        # this machine's filesystem layout into a committed CSV, which breaks
        # reproducibility for anyone else and leaks local directory names.
        try:
            rel = p.relative_to(ROOT).as_posix()
        except ValueError:
            rel = p.as_posix()  # outside the repo, keep as-is
        rows.append({"path": rel, "label": label, "split_hint": hint})

    if unlabelled:
        log.warning(
            "%d files could not be labelled from their path (e.g. %s). "
            "These are excluded.",
            len(unlabelled),
            unlabelled[0] if unlabelled else "",
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        log.info(
            "  -> %d labelled images | real=%d fake=%d",
            len(df),
            (df.label == 0).sum(),
            (df.label == 1).sum(),
        )
    return df


def inspect(config: dict) -> dict[str, pd.DataFrame]:
    """Scan both datasets and save full manifests."""
    manifests = resolve(config["paths"]["manifests"])
    manifests.mkdir(parents=True, exist_ok=True)
    raw = resolve(config["paths"]["data_raw"])

    out = {}
    for key in ("a_train", "b_crosstest"):
        ds = config["datasets"][key]
        df = scan_dataset(raw / key)
        out[key] = df
        if not df.empty:
            dest = manifests / f"{key}_full.csv"
            df.to_csv(dest, index=False)
            log.info("Wrote manifest: %s (%d rows)", dest.name, len(df))
        else:
            log.error("No images found for %s (%s).", key, ds["kaggle_slug"])
    return out


# ---------------------------------------------------------------------------
# Subset
# ---------------------------------------------------------------------------

def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """
    Take a class-balanced sample of n rows total (n/2 per class).

    If a class has fewer than n/2 available we take everything it has and
    warn, rather than failing, so the pipeline still runs on partial data.
    """
    per_class = n // 2
    parts = []
    for label in (0, 1):
        pool = df[df.label == label]
        if len(pool) < per_class:
            log.warning(
                "Only %d images available for label=%d, wanted %d. Taking all.",
                len(pool),
                label,
                per_class,
            )
            parts.append(pool)
        else:
            parts.append(pool.sample(n=per_class, random_state=seed))
    return (
        pd.concat(parts)
        .sample(frac=1, random_state=seed)  # shuffle
        .reset_index(drop=True)
    )


def build_subsets(config: dict, manifests_in: dict[str, pd.DataFrame]) -> None:
    """
    Create the seeded train/val/test splits from Dataset A and the
    cross-test split from Dataset B.

    Splits are disjoint by construction: we sample train first, remove those
    rows, then sample val, remove, then test. No image can leak between splits.
    """
    seed = config["project"]["seed"]
    sub = config["subset"]
    manifests = resolve(config["paths"]["manifests"])

    # ---- Dataset A: train / val / test ----
    df_a = manifests_in.get("a_train", pd.DataFrame())
    if df_a.empty:
        log.error("Dataset A manifest is empty. Run --step download first.")
        return

    # Sanity check before we start: is there even enough data to fill the
    # requested splits? Without this, an oversized train_size silently eats
    # everything and leaves val/test empty, which would not be obvious until
    # evaluation produced nonsense.
    needed = sub["train_size"] + sub["val_size"] + sub["test_size"]
    if len(df_a) < needed:
        log.error(
            "Dataset A has %d images but the config asks for %d "
            "(train %d + val %d + test %d). Reduce the sizes in config.yaml.",
            len(df_a),
            needed,
            sub["train_size"],
            sub["val_size"],
            sub["test_size"],
        )
        return

    # Dataset A ships official train/valid/test folders. If configured (and
    # the hints are actually present), draw each subset from the matching
    # official folder so our protocol matches published work on this dataset.
    use_official = sub.get("respect_official_splits", False) and (
        df_a.split_hint.notna().all() and (df_a.split_hint != "").all()
    )
    if sub.get("respect_official_splits", False) and not use_official:
        log.warning(
            "respect_official_splits is on but not every image carries a "
            "train/val/test hint. Falling back to a random split across "
            "the whole dataset."
        )
    if use_official:
        log.info("Drawing each subset from Dataset A's official split folders.")
    else:
        log.info("Drawing subsets randomly across the whole of Dataset A.")

    remaining = df_a.copy()
    for split, size in (
        ("train", sub["train_size"]),
        ("val", sub["val_size"]),
        ("test", sub["test_size"]),
    ):
        pool = remaining[remaining.split_hint == split] if use_official else remaining
        if use_official and len(pool) < size:
            log.error(
                "Official '%s' folder holds %d images but %d were requested. "
                "Lower %s_size in config.yaml.",
                split,
                len(pool),
                size,
                split,
            )
            return
        chosen = stratified_sample(pool, size, seed)
        if chosen.empty:
            log.error("Split '%s' came out empty. Aborting.", split)
            return
        remaining = remaining[~remaining.path.isin(chosen.path)]
        chosen["split"] = split
        dest = manifests / f"subset_{split}.csv"
        chosen.to_csv(dest, index=False)
        log.info(
            "%-6s %5d images (real=%d fake=%d) -> %s",
            split,
            len(chosen),
            (chosen.label == 0).sum(),
            (chosen.label == 1).sum(),
            dest.name,
        )

    # ---- Dataset B: cross-test only ----
    df_b = manifests_in.get("b_crosstest", pd.DataFrame())
    if df_b.empty:
        log.error("Dataset B manifest is empty. Cross-dataset test unavailable.")
        return

    cross = stratified_sample(df_b, sub["crosstest_size"], seed)
    cross["split"] = "crosstest"
    dest = manifests / "subset_crosstest.csv"
    cross.to_csv(dest, index=False)
    log.info(
        "%-6s %5d images (real=%d fake=%d) -> %s",
        "cross",
        len(cross),
        (cross.label == 0).sum(),
        (cross.label == 1).sum(),
        dest.name,
    )

    # ---- Leakage check ----
    # Paranoid, but a train/test leak would invalidate every number in the
    # report, so it is worth the two seconds.
    tr = set(pd.read_csv(manifests / "subset_train.csv").path)
    va = set(pd.read_csv(manifests / "subset_val.csv").path)
    te = set(pd.read_csv(manifests / "subset_test.csv").path)
    assert not (tr & va), "LEAK: train and val overlap"
    assert not (tr & te), "LEAK: train and test overlap"
    assert not (va & te), "LEAK: val and test overlap"
    log.info("Leakage check passed - all splits disjoint.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset preparation pipeline")
    ap.add_argument(
        "--step",
        choices=["download", "inspect", "subset", "all"],
        default="all",
        help="Which stage to run",
    )
    args = ap.parse_args()

    config = load_config()
    set_seed(config["project"]["seed"])
    ensure_dirs(config)

    raw = resolve(config["paths"]["data_raw"])

    if args.step in ("download", "all"):
        if not check_kaggle_credentials():
            sys.exit(1)
        for key in ("a_train", "b_crosstest"):
            slug = config["datasets"][key]["kaggle_slug"]
            if not download_dataset(slug, raw / key):
                log.error("Stopping - could not download %s", slug)
                sys.exit(1)

    manifests_in = {}
    if args.step in ("inspect", "all"):
        manifests_in = inspect(config)

    if args.step in ("subset", "all"):
        if not manifests_in:
            # Reload from disk if inspect was run separately
            mdir = resolve(config["paths"]["manifests"])
            for key in ("a_train", "b_crosstest"):
                f = mdir / f"{key}_full.csv"
                manifests_in[key] = (
                    pd.read_csv(f) if f.exists() else pd.DataFrame()
                )
        build_subsets(config, manifests_in)

    log.info("Done.")


if __name__ == "__main__":
    main()
