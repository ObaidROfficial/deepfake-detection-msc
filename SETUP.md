# Setup Guide

Follow these in order. Should take about 20 minutes plus download time.

---

## 1. Check your Python version

Open PowerShell and run:

```powershell
python --version
```

You need **Python 3.11**. If you have 3.12 or 3.13, most things will work but some
packages lag behind — 3.11 is the safe choice. Download from python.org if needed.

---

## 2. Create a virtual environment

From inside the `deepfake-detection-msc` folder:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script with an execution policy error:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

You'll know it worked when your prompt starts with `(venv)`.

---

## 3. Install PyTorch — CPU build

**This step matters.** Run it *before* `requirements.txt`, and use the CPU index URL:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you skip the `--index-url` flag, pip downloads the CUDA build instead — about
2.5GB of GPU libraries that your Intel integrated graphics cannot use. It wastes
the bandwidth and the disk space and gains you nothing.

Verify:

```powershell
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expect `CUDA: False`. That is correct for this machine, not an error.

---

## 4. Install everything else

```powershell
pip install -r requirements.txt
```

---

## 5. Kaggle API token

This is the one part I can't do for you — it needs your login.

1. Sign in at [kaggle.com](https://www.kaggle.com)
2. Click your **avatar** (top right) → **Settings**
3. Scroll to the **API** section
4. Click **Create New Token** — a file called `kaggle.json` downloads
5. Move it to `C:\Users\obaid\.kaggle\kaggle.json`

Create the folder first if it doesn't exist:

```powershell
mkdir $env:USERPROFILE\.kaggle
move $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\
```

Verify:

```powershell
kaggle datasets list -s deepfake
```

If that prints a list, you're set.

> **Never commit `kaggle.json`.** It's a credential tied to your account. It's
> already listed in `.gitignore`, but be aware of it.

---

## 6. Accept the dataset terms

The API won't serve a dataset until you've accepted its terms on the website
once. Open both pages, click **"I Understand and Accept"**:

| Dataset | Role | Link |
|---|---|---|
| 140k Real and Fake Faces | Training + in-domain testing | https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces |
| Deepfake and Real Images | Cross-dataset testing only | https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images |

If you skip this, the download fails with a 403 and a confusing error message.

---

## 7. Run the benchmark first

**Do this before downloading anything.** It takes a couple of minutes, needs no
data, and tells us whether the planned subset sizes are realistic on your CPU:

```powershell
python src\benchmark.py
```

It prints a projected total training time for each model. Send me those numbers.
If it says more than about 6 hours per model, we shrink the subset before
committing to it, rather than discovering the problem four hours into a run.

---

## 8. Download the datasets

```powershell
python src\data_prep.py --step download
```

Roughly 4GB for Dataset A and 2GB for Dataset B. Expect this to take a while
on a normal connection. It skips anything already downloaded, so it's safe to
interrupt and rerun.

---

## 9. Build the subsets

```powershell
python src\data_prep.py --step inspect
python src\data_prep.py --step subset
```

This scans both datasets, writes full manifests, then draws the seeded
train/val/test and cross-test splits. It runs a leakage assertion at the end —
if any image appears in two splits it stops immediately rather than quietly
producing invalid results.

Manifests land in `data/manifests/`. These **are** committed to git (they're
small CSVs of file paths), which is what makes the experiment reproducible.

---

## 10. Explore the data

```powershell
jupyter lab
```

Open `notebooks/01_data_exploration.ipynb` and run all cells. This produces the
figures for Chapter 5 of the report.

---

## Troubleshooting

**`kaggle: command not found`**
The venv isn't active, or `pip install -r requirements.txt` didn't finish.
Reactivate with `.\venv\Scripts\Activate.ps1`.

**`403 Forbidden` on download**
You haven't accepted the dataset terms. See step 6.

**`OSError: [Errno 28] No space left on device`**
Both datasets plus processed copies need roughly 10GB free. Check your drive.

**Training makes the laptop unusable**
Lower `compute.torch_threads` in `config.yaml` from 8 to 6 or 4.

**A run got interrupted**
Checkpointing is on by default (`training.checkpoint_every_epoch`), so training
resumes from the last completed epoch rather than starting over.

**OneDrive keeps syncing the data folder**
`data/` is gitignored but OneDrive doesn't read `.gitignore`. Right-click the
`data` folder → **Always keep on this device** is fine, but if OneDrive is
struggling with tens of thousands of small files, consider moving the repo
outside your OneDrive folder — for example `C:\Users\obaid\projects\`.
