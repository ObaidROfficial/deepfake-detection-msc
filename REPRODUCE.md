# Reproducing the Results

This guide is written for a marker or reviewer who wants to verify the results
reported in the dissertation.

There are four paths depending on how much time you have and what you want to
establish. **Path B is probably the one you want** — it takes about five
minutes and lets you run the detector on your own images, with no datasets and
no Kaggle account required.

| Path | What it does | Compute | Downloads |
|---|---|---|---|
| **A. Inspect** | Check every reported figure against the committed result files | ~1 min | none |
| **B. Try it on an image** | Run the detector on your own photos | ~5 min setup | 25 MB weights only |
| **C. Verify** | Re-run evaluation and robustness using the released weights | ~50 min | 6 GB data + weights |
| **D. Full rebuild** | Retrain everything from scratch | ~8 hours | 6 GB data |

**Path B needs no datasets and no Kaggle account** — just Python, PyTorch and a
25 MB weights file. If you want to confirm the artefact is a working detector
rather than a set of result tables, that is the fastest way.

---

## Hardware this was developed on

- CPU-only. Intel, 12 physical cores / 14 logical, no CUDA
- 16 GB RAM
- Windows 11, Python 3.11.6, PyTorch 2.13.0+cpu

**No GPU is required for any path.** If you do have a CUDA GPU, set
`compute.device: cuda` in `config.yaml` and everything will run considerably
faster, though the exact numbers may shift very slightly because some CPU and
GPU kernels differ in floating-point behaviour.

If your machine has fewer than 12 cores, reduce `compute.torch_threads` in
`config.yaml` to roughly two-thirds of your core count. If you have less than
16 GB of RAM, reduce `models.baselines.train_size` from 10000 — the HOG+LBP
feature matrix is the largest memory consumer at roughly 490 MB.

---

## Path A — Inspect the results (1 minute, no setup)

Every number in the report is committed to this repository. No installation
required.

| Reported in | File |
|---|---|
| Baseline metrics (Table 6.1, 6.3) | `results/baseline_metrics.csv` |
| MobileNetV2 metrics | `results/cnn_metrics_mobilenet_v2.csv` |
| EfficientNet-B0 metrics | `results/cnn_metrics_efficientnet_b0.csv` |
| Training curves data | `results/history_*.csv` |
| Robustness suite (Table 6.5, 6.6) | `results/robustness_mobilenet_v2.csv` |
| Raw per-image scores | `results/scores_*.csv` |
| CPU benchmark | `results/benchmark.csv` |
| All 25 figures | `results/figures/` |

The exact images used in every split are listed in `data/manifests/subset_*.csv`,
so the experiment is pinned to specific files rather than to a sampling
procedure.

---

## Path B — Try the detector on your own images (~5 min)

No datasets, no Kaggle account, no training. This is the quickest way to
confirm the artefact actually works.

### 1a. Install — the installer (nothing to type)

Download the installer for your platform from the
[latest release](https://github.com/ObaidROfficial/deepfake-detection-msc/releases/latest)
and run it. Python is **not** needed beforehand; if the machine has nothing
between 3.9 and 3.13, the installer offers to fetch the official Python 3.11.9
from python.org, checks it against the published checksum, and installs it
alongside whatever is already there.

The executables are not code-signed, so Windows shows "Windows protected your
PC" (*More info* → *Run anyway*) and macOS refuses a double-click (right-click
→ *Open* → confirm). Details in [installer/README.md](installer/README.md).

Once it finishes, open it from the desktop shortcut on Windows or from the
Applications folder on macOS, and skip to step 2.

### 1b. Install — one command

If you would rather have the source tree as well, or you are already at a
terminal. This route does need Python 3.9–3.13 installed.

**Windows**

```powershell
git clone https://github.com/ObaidROfficial/deepfake-detection-msc.git
cd deepfake-detection-msc
.\setup.ps1 -Minimal
```

**macOS / Linux**

```bash
git clone https://github.com/ObaidROfficial/deepfake-detection-msc.git
cd deepfake-detection-msc
chmod +x setup.sh && ./setup.sh --minimal
```

The setup script creates a virtual environment, installs the CPU build of
PyTorch and the demo dependencies, downloads the trained weights from the
Releases page, and verifies the install. It is safe to re-run — completed steps
are skipped.

`-Minimal` / `--minimal` installs only what the demo needs. Omit it to install
everything required for the experiments as well.

If PowerShell refuses to run the script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

<details>
<summary>Manual install, if you would rather not run a script</summary>

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow pyyaml numpy pandas opencv-python
```

Then download `mobilenet_v2_best.pt` from the
[Releases page](https://github.com/ObaidROfficial/deepfake-detection-msc/releases)
into `models/`.
</details>

### 2. Launch the interface

If you used the installer, the desktop shortcut (Windows) or the app in
Applications (macOS) does this. From a source install:

```powershell
.\run.ps1        # or ./run.sh on macOS/Linux
```

Opens its own window. Choose an image and it classifies immediately.

### 3. Or use the command line

Source install only — the installer does not put `src/` on your path.

```powershell
python src\predict.py --image path\to\your\photo.jpg
```

Or a whole folder, optionally writing a CSV:

```powershell
python src\predict.py --folder path\to\images\
python src\predict.py --folder path\to\images\ --csv predictions.csv
```

To use the other architecture:

```powershell
python src\predict.py --image photo.jpg --model efficientnet_b0
```

### What you will see

```
  photo.jpg
    verdict     : FAKE
    P(fake)     : 0.9871
    confidence  : high
```

The tool inspects the input and warns when it is being asked to work outside
the conditions it was validated on — images below 224px, or heavily compressed
JPEGs — citing the measured accuracy for that condition rather than a generic
caution.

Every run ends with a summary of measured performance, including the cases
where the detector fails. That is deliberate. Chapter 8 argues that a detector
which silently fails while appearing to work is worse than no detector, so the
demonstration tool is built to state its own operating limits.

### Suggested things to try

- **A real photograph of a face** — should return REAL
- **A StyleGAN face** from thispersondoesnotexist.com — should return FAKE with high confidence, since this is exactly the distribution it was trained on
- **A deepfake made by a different method**, such as a face swap — this is where it will likely fail, and failing here is the central finding of the project rather than a bug
- **The same image downscaled to 64×64** — watch the warning appear and the confidence change
- **A non-face image** — the model has only ever seen faces and will still return a confident verdict, which illustrates why the caveats matter

---

## Path C — Verify the reported results (~50 min)

This re-runs evaluation and the robustness suite against the trained models,
reproducing Tables 6.1, 6.3, 6.4, 6.5 and 6.6 without any training.

### 1. Environment

```powershell
git clone https://github.com/ObaidROfficial/deepfake-detection-msc.git
cd deepfake-detection-msc
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On macOS or Linux use `source venv/bin/activate`.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The `--index-url` matters: without it pip installs the CUDA build, roughly
2.5 GB of GPU libraries that are unnecessary on a CPU-only machine.

For an exact match to the development environment, use
`pip install -r requirements-lock.txt` instead of `requirements.txt`.

Confirm:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 2. Kaggle credentials

Both datasets are downloaded from Kaggle and require a free account.

1. Sign in at kaggle.com → avatar → **Settings** → **API** → **Create New Token**
2. Move the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
   (`C:\Users\<you>\.kaggle\kaggle.json` on Windows)
3. Visit both dataset pages and click **"I Understand and Accept"**:
   - https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
   - https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images

Step 3 is required — the API returns 403 without it.

### 3. Data

```powershell
python src\data_prep.py --step download
python src\data_prep.py --step inspect
```

Roughly 5.4 GB and 45 minutes on a typical connection. `--step inspect`
regenerates the full manifests, which are deliberately not committed because
they are large and derivable.

**Do not run `--step subset`.** The subset manifests are already committed, and
regenerating them is unnecessary. They are reproducible from the same seed if
you wish to confirm that, but the committed versions are what produced the
reported results.

### 4. Trained weights

Download from the [Releases page](https://github.com/ObaidROfficial/deepfake-detection-msc/releases)
and place in `models/`:

- `mobilenet_v2_best.pt` (8.9 MB)
- `efficientnet_b0_best.pt` (16 MB)

### 5. Run

```powershell
python src\evaluate.py --model mobilenet_v2
python src\evaluate.py --model efficientnet_b0
python src\baselines.py
python src\robustness.py --model mobilenet_v2
```

Approximate times on the reference hardware: evaluation 3 minutes each,
baselines 20 minutes (feature extraction dominates, then cached), robustness
suite 35 minutes.

### Expected output

| Model | In-domain acc | In-domain AUC | Cross-dataset acc | Cross-dataset AUC |
|---|---|---|---|---|
| Logistic Regression | 0.7358 | 0.8121 | 0.4883 | 0.5063 |
| HOG + LBP → SVC | 0.7860 | 0.8706 | 0.4790 | 0.4751 |
| MobileNetV2 | 0.9430 | 0.9869 | 0.5110 | 0.5593 |
| EfficientNet-B0 | 0.9457 | 0.9910 | 0.5038 | 0.5285 |

Robustness, MobileNetV2, clean baseline 0.9430: JPEG q=10 → 0.7288;
noise σ=0.10 → 0.6797; blur k=7 → 0.7943; downscale 32px → 0.5580;
brightness ±0.4 → 0.9105 worst; contrast ±0.4 → 0.9048 worst;
FGSM ε=0.005 (predicted labels) → 0.0520.

**On exact agreement.** Evaluation, baselines and the deterministic
perturbations should reproduce exactly, because they involve no randomness
beyond the seeded sampling. The Gaussian noise perturbation seeds its generator
from image content, so it is also deterministic. FGSM is deterministic given
fixed weights. If any figure differs by more than ~0.001, something has
changed — most likely a package version, so try `requirements-lock.txt`.

---

## Path D — Full rebuild (~8 hours)

Follow Path C steps 1–3, then instead of downloading weights:

```powershell
python src\benchmark.py                          # ~2 min, optional
python src\baselines.py                          # ~20 min
python src\train.py --model mobilenet_v2         # ~2.4 h
python src\evaluate.py --model mobilenet_v2      # ~3 min
python src\robustness.py --model mobilenet_v2    # ~35 min
python src\train.py --model efficientnet_b0      # ~4.4 h
python src\evaluate.py --model efficientnet_b0   # ~3 min
```

Training checkpoints every epoch. If interrupted, add `--resume`:

```powershell
python src\train.py --model mobilenet_v2 --resume
```

**Expect training results to differ slightly.** CPU training is not
bit-for-bit deterministic across machines — some backend kernels vary in
operation order, and results depend on thread count. Accuracy should land
within roughly ±0.5 points of the reported figures. What *is* exactly
reproducible is the data sampling, the splits and the evaluation protocol,
which is what the comparisons rest on.

Timing will also vary substantially with background load. During development,
identical epochs ranged from 11.4 to 48.8 minutes purely because of other
activity on the machine.

### Smoke test

To confirm the pipeline works before committing hours to it:

```powershell
python src\train.py --model mobilenet_v2 --limit-train 2000
```

About 7 minutes. Validation accuracy should exceed 0.70. If it sits at 0.50,
something is wrong with the data or labels and full training would waste time.

---

## Troubleshooting

**`kaggle: command not found`** — the virtual environment is not active, or
`pip install -r requirements.txt` did not complete.

**`403 Forbidden` on download** — dataset terms not accepted. See Path C step 2.

**DataLoader workers hang or crash on Windows** — set
`compute.dataloader_workers: 0` in `config.yaml`. Windows uses `spawn` rather
than `fork` for worker processes, which is occasionally fragile. Slower but
reliable.

**`OSError: [Errno 28] No space left on device`** — the datasets plus cached
features need roughly 10 GB free.

**Machine becomes unusable during training** — lower
`compute.torch_threads` in `config.yaml`.

**Memory error during the HOG+LBP baseline** — lower
`models.baselines.train_size` from 10000. The feature matrix is
10,000 × 6,110 float32, approximately 244 MB before scaling and 489 MB after.

**Numbers differ from the report** — check `pip freeze` against
`requirements-lock.txt`. Version differences in scikit-learn or PyTorch are
the most likely cause.

---

## Removing it afterwards

**If you used the installer (Path B) on Windows,** run
`OBI-DeepFake-Image-Detector-Uninstall.exe` from the install folder, where the
installer put it. It shows what will be removed and how much space that frees
before it does anything, and it covers the shared caches with the same default
described below.

**If you used the installer on macOS,** drag the app from Applications to the
Trash. The virtual environment, the weights and the code all sit inside the
bundle, so that is a complete removal apart from the shared PyTorch cache in
the table below.

**If you installed from source,** read on.

Almost everything lives inside the project folder, so deleting that folder
removes the virtual environment, the datasets, the cached features, the model
weights and the generated figures in one go:

```bash
cd ..
rm -rf deepfake-detection-msc          # Windows: Remove-Item -Recurse -Force
```

Three things are written **outside** the project folder and survive that:

| Path | What | Size |
|---|---|---|
| `~/.cache/torch/hub/checkpoints` | ImageNet weights torchvision downloads on first use | ~35 MB |
| `~/.cache/huggingface` | left by older versions that used Gradio | varies |
| `~/.kaggle/kaggle.json` | only if you created it for the datasets | 1 KB |

An uninstall script handles all of this:

```bash
chmod +x uninstall.sh
./uninstall.sh --dry-run     # show what would go, change nothing
./uninstall.sh               # remove, with confirmation
./uninstall.sh --caches      # also clear the shared caches
```

Windows equivalent: `.\uninstall.ps1`, `-DryRun`, `-Caches`.

**The shared caches are left alone by default.** They are used by any other
Python project on the machine, so clearing them would force those to
re-download too. Pass `--caches` only if you are sure. The Kaggle credential is
never deleted automatically, since it is yours rather than the project's.

---

## What is deliberately not in this repository

| Excluded | Why | How to obtain |
|---|---|---|
| Dataset images (~5.4 GB) | Licensing — not ours to redistribute | `data_prep.py --step download` |
| Trained weights (~68 MB) | Binaries do not belong in git history | GitHub Releases |
| Full manifests (~22 MB) | Large and derivable | `data_prep.py --step inspect` |
| Cached features | Derivable, tens of MB | Regenerated by `baselines.py` |
| `kaggle.json` | Credential | Your own Kaggle account |

Dataset licences are recorded in `LICENSE`. Dataset A's real images derive from
NVIDIA's FFHQ under CC BY-NC-SA 4.0 (non-commercial use only); Dataset B
derives from OpenForensics, released for research purposes.
