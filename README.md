# AI-Based Deepfake Detection Using Machine Learning with Robustness Evaluation

MSc Advanced Computing Project (COM7014) — Arden University

**Author:** Muhammad Obaid Ur Rehman (stu3000)
**Supervisor:** Tony Stark
**Submission:** 21 October 2026

---

## Overview

This project builds and evaluates a machine learning system for detecting
deepfake facial imagery. It goes beyond reporting a single accuracy figure in
two ways that matter for real deployment:

1. **Cross-dataset generalisation.** The model is trained on one dataset and
   tested on a second produced by a completely different forgery method. This
   tests whether it learned a general notion of facial forgery or just memorised
   the artefacts of one generator.

2. **Robustness evaluation.** The trained model is re-scored against JPEG
   compression, noise, blur, downscaling, brightness and contrast shifts, and
   FGSM adversarial perturbation. Images shared on social platforms are
   recompressed and resized constantly, so a detector that only works on pristine
   inputs is of limited practical value.

---

## Research questions

1. How effective are machine learning models in detecting deepfake media?
2. Which features or patterns are most useful in distinguishing real and fake media?
3. How does the performance of the proposed model compare to baseline approaches?
4. How robust is the detection model against manipulated or adversarial inputs?
5. What are the key limitations of current deepfake detection techniques?

---

## Datasets

| | Dataset | Role | Composition |
|---|---|---|---|
| **A** | [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | Train / validate / in-domain test | 70k real (Flickr, FFHQ) + 70k fake (StyleGAN) |
| **B** | [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) | Cross-dataset test **only** | ~190k images derived from [OpenForensics](https://zenodo.org/record/5528418), in-the-wild face forgeries |

The two datasets use different generation methods on purpose. Dataset A is fully
synthetic GAN-generated faces; Dataset B is in-the-wild manipulation with varied
pose, lighting and background. Dataset B is never used for training.

### Note on dataset selection

The original project proposal specified FaceForensics++ and the Kaggle DFDC
corpus. Neither was usable in practice:

- **FaceForensics++** requires a signed EULA submitted to TU Munich with manual
  approval, typically one to two weeks with no guarantee inside the project
  window. It is also distributed as raw video requiring frame extraction and
  face cropping.
- **Kaggle DFDC** is approximately 470GB of video. Decoding that volume on a
  CPU-only machine is not achievable in the time available.

Both substitutes provide pre-extracted 256×256 face crops with open access,
which removes the video decoding stage entirely. This deviation is documented
and justified in Chapter 3 of the report, and its consequences — chiefly reduced
comparability with FF++-benchmarked literature — are discussed in Chapter 7.

---

## Models

| Tier | Model | Purpose |
|---|---|---|
| Baseline 1 | Logistic Regression on flattened pixels | Deliberately weak floor |
| Baseline 2 | HOG + LBP features → Linear SVM | Classical computer vision baseline |
| Main | MobileNetV2 (ImageNet pretrained) | Primary model, chosen for CPU feasibility |
| Comparison | EfficientNet-B0 (ImageNet pretrained) | Architecture comparison |

Both CNNs use two-stage transfer learning: train the classifier head with the
backbone frozen, then unfreeze the trailing feature blocks and fine-tune at a
low learning rate. The two networks are configured to **matched trainable
capacity** (1,208,642 against 1,131,954 parameters, within 6.4%) rather than the
same nominal setting, because their backbones are structured very differently —
19 small blocks against 9 large stages. Without this the comparison would
measure adaptation capacity rather than architecture.

---

## Headline results

All models trained on Dataset A at 224×224, evaluated on a held-out split of
Dataset A (in-domain) and on Dataset B (never seen during training).

| Model | In-domain acc | In-domain AUC | Cross-dataset acc | Cross-dataset AUC | Gap |
|---|---|---|---|---|---|
| Logistic Regression | 0.7358 | 0.8121 | 0.4883 | 0.5063 | 24.7 |
| HOG + LBP → SVC | 0.7860 | 0.8706 | 0.4790 | 0.4751 | 30.7 |
| MobileNetV2 | 0.9430 | 0.9869 | 0.5110 | 0.5593 | 43.2 |
| EfficientNet-B0 | 0.9457 | 0.9910 | 0.5038 | 0.5285 | 44.2 |

**In-domain accuracy rises 21 points across the four models. Cross-dataset
accuracy never leaves the neighbourhood of chance, and the generalisation gap
widens monotonically.** Every gain from better representation learning was
absorbed by the source domain.

A threshold diagnostic confirms this is a *representation* failure rather than a
*calibration* one: the best accuracy achievable at any threshold in either
polarity is 0.5453. No operating point recovers usable performance.

Because MobileNetV2 and EfficientNet-B0 are structurally very different and fail
identically at matched capacity, the result is not attributable to one
architecture choice.

### Robustness (MobileNetV2, clean baseline 0.9430)

| Perturbation | Worst setting | Accuracy | Drop |
|---|---|---|---|
| Downscale round-trip | 32px | 0.5580 | 0.385 |
| Gaussian noise | σ = 0.10 | 0.6797 | 0.263 |
| JPEG compression | quality 10 | 0.7288 | 0.214 |
| Gaussian blur | k = 7 | 0.7943 | 0.149 |
| Contrast | ±40% | 0.9048 | 0.038 |
| Brightness | ±40% | 0.9105 | 0.032 |
| **FGSM (white-box)** | **ε = 0.005** | **0.0520** | **0.891** |

Perturbations that destroy high-frequency information cost up to 38.5 accuracy
points; intensity rescaling of comparable perceptual magnitude costs at most
3.8. This supports the account that the discriminative signal is concentrated in
the high-frequency band (Frank et al., 2020), which independently explains why
raising input resolution from 128px to 224px was worth **+10.1 accuracy
points**.

---

## Install and run

**Windows**

```powershell
git clone https://github.com/ObaidROfficial/deepfake-detection-msc.git
cd deepfake-detection-msc
.\setup.ps1
```

**macOS / Linux**

```bash
git clone https://github.com/ObaidROfficial/deepfake-detection-msc.git
cd deepfake-detection-msc
chmod +x setup.sh && ./setup.sh
```

That creates a virtual environment, installs the CPU build of PyTorch and all
dependencies, downloads the trained weights from Releases, and verifies the
install. Add `-Minimal` (Windows) or `--minimal` (Unix) to install only what
the demo needs, skipping the experiment dependencies.

Then:

```powershell
.\venv\Scripts\Activate.ps1     # source venv/bin/activate on macOS/Linux
python app.py                   # graphical interface at http://127.0.0.1:7860
```

## Try it on your own image

```powershell
python src\predict.py --image your_photo.jpg
```

```
  your_photo.jpg
    verdict     : FAKE
    P(fake)     : 0.9871
    confidence  : high
```

No datasets or Kaggle account needed. `predict.py` also accepts `--folder` for
batch classification and `--csv` to write results out.

The tool warns when an input falls outside the conditions the model was
validated on, and every run prints a summary of where the detector is known to
fail. That is deliberate rather than decorative — see the results below.

## For markers and reviewers

**See [REPRODUCE.md](REPRODUCE.md).** Four paths depending on available time:
inspect the committed results (~1 min), run the detector on your own images
(~5 min), re-verify every reported number from the released weights (~50 min),
or rebuild everything from scratch (~8 h). No GPU needed for any of them.

---

## Quick start

Full installation instructions in **[SETUP.md](SETUP.md)**. Short version:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python src\benchmark.py                      # check CPU speed first
python src\data_prep.py --step all           # download + subset
jupyter lab                                  # notebooks/01_data_exploration.ipynb
```

---

## Repository layout

```
deepfake-detection-msc/
├── config.yaml              # every setting lives here
├── requirements.txt         # + requirements-lock.txt with exact versions
├── SETUP.md                 # step-by-step install guide
├── data/
│   ├── raw/                 # downloaded datasets (gitignored)
│   └── manifests/           # seeded split CSVs (committed - this is what pins the experiment)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── utils.py             # config loading, seeding, thread setup
│   ├── data_prep.py         # download, scan, subset, leakage assertion
│   ├── benchmark.py         # CPU speed test
│   ├── baselines.py         # logistic regression + HOG/LBP SVM
│   ├── model.py             # dataset class, transforms, model builders
│   ├── train.py             # two-stage transfer learning, checkpoint/resume
│   ├── evaluate.py          # in-domain + cross-dataset, threshold diagnostic
│   └── robustness.py        # degradation suite + FGSM
├── models/                  # trained weights (gitignored - see Releases)
├── results/                 # metrics CSVs and 23 figures
└── docs/                    # report chapters in Markdown
```

### Run order

```powershell
python src\benchmark.py                        # optional: check CPU speed
python src\data_prep.py --step all             # download, inspect, subset
python src\baselines.py                        # classical baselines
python src\train.py --model mobilenet_v2       # ~2.4 h on CPU
python src\evaluate.py --model mobilenet_v2
python src\robustness.py --model mobilenet_v2  # ~35 min
python src\train.py --model efficientnet_b0    # ~4.4 h on CPU
python src\evaluate.py --model efficientnet_b0
```

---

## Reproducibility

- Every random operation is seeded from `config.yaml` (`project.seed: 42`)
- Split manifests are committed as CSVs, so the exact same subset can be
  rebuilt from the same source datasets
- `data_prep.py` asserts that train, validation and test splits are disjoint and
  fails loudly on any overlap
- All hyperparameters live in `config.yaml` rather than scattered through code

Note that CPU training is not bit-for-bit deterministic across different
machines, since some backend kernels vary. The data sampling and splits are
exactly reproducible, which is the part that affects the validity of the results.

---

## Hardware

Developed and trained on a CPU-only laptop: 12 cores, 16GB RAM, Intel integrated
graphics, no CUDA. Thread count is capped at 8 in `config.yaml` so the machine
stays usable during long runs. This constraint shaped several design decisions,
particularly the choice of MobileNetV2 and the decision to subsample rather than
train on the full 140k images. Those trade-offs are discussed in the report.

One measured consequence worth noting: identical training epochs varied between
11.4 and 48.8 minutes purely because of background load on the machine. CPU
training is far more sensitive to system contention than GPU training, which
limits how precisely timing claims can be stated.

---

## Status

Experimental work complete. Four models trained and evaluated, robustness suite
and adversarial evaluation run, all results in `results/`. Report chapters
drafted in `docs/` and being finalised.

Total compute for all experiments: approximately **eight hours on CPU**.

Outstanding: literature review citations to be verified, robustness suite to be
repeated on EfficientNet-B0, final report assembly.

---

## Licence

Code released under the MIT Licence — see [LICENSE](LICENSE).

Datasets remain under their original licences and are **not** redistributed here.
Dataset A's real images derive from NVIDIA's FFHQ (CC BY-NC-SA 4.0). Dataset B
derives from OpenForensics, released for research use. Both are downloaded
directly from Kaggle by the user under their own account.

---

## Academic integrity

This repository is submitted as part of an assessed MSc project. It is public so
that the work can be verified and built upon. If you are a student on the same
module, use it as a reference for approach and cite it — do not copy it.
