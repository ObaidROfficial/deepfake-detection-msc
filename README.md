# AI-Based Deepfake Detection Using Machine Learning with Robustness Evaluation

MSc Advanced Computing Project (COM7014), Arden University

**Author:** Muhammad Obaid Ur Rehman (STU1215887)
**Supervisor:** Dr. Muhammad Saqib
**Submission:** 21 October 2026

---

## Overview

This project builds and evaluates a machine learning system that detects deepfake
face images. Rather than reporting one accuracy figure it looks at two things that
matter when a detector is actually used.

1. **Cross-dataset generalisation.** The model is trained on one dataset and tested
   on a second dataset made by a completely different forgery method. This tests
   whether the model learned what a fake face is in general or only learned the
   artefacts of one generator.

2. **Robustness evaluation.** The trained model is tested again under JPEG
   compression, noise, blur, downscaling, brightness and contrast changes and FGSM
   adversarial perturbation. Images on social media are compressed and resized all
   the time so a detector that only works on clean images is of limited use.

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

The two datasets were chosen because they use different generation methods.
Dataset A is fully synthetic GAN generated faces whilst Dataset B is in-the-wild
manipulation with different poses, lighting and backgrounds. Dataset B is never
used for training.

### Note on dataset selection

The approved proposal specified FaceForensics++ and the Kaggle DFDC dataset but
neither could be used.

- **FaceForensics++** needs a signed EULA to be sent to TU Munich and manually
  approved. This usually takes one to two weeks with no guarantee it would fall
  inside the project timeline. The data is also raw video which needs frame
  extraction and face cropping.
- **Kaggle DFDC** is approximately 470GB of video. Decoding this on a CPU only
  machine was not possible in the time available.

Both replacement datasets come as pre-extracted 256×256 face crops with open
access so no video decoding is needed. This change is explained in Chapter 3 of
the report and the main consequence, which is that results are less comparable
to work that uses FF++, is discussed in Chapter 7.

---

## Models

| Tier | Model | Purpose |
|---|---|---|
| Baseline 1 | Logistic Regression on flattened pixels | Deliberately weak floor |
| Baseline 2 | HOG + LBP features → Linear SVM | Classical computer vision baseline |
| Main | MobileNetV2 (ImageNet pretrained) | Primary model, chosen for CPU feasibility |
| Comparison | EfficientNet-B0 (ImageNet pretrained) | Architecture comparison |

Both CNNs use two-stage transfer learning. The classifier head is trained first
with the backbone frozen and then the last feature blocks are unfrozen and
fine-tuned at a low learning rate. The two networks are set to **matched
trainable capacity** (1,208,642 against 1,131,954 parameters, within 6.4%)
rather than the same setting as their backbones are built very differently with
19 small blocks against 9 large stages. Without this the comparison would be
measuring adaptation capacity and not architecture.

---

## Headline results

All models were trained on Dataset A at 224×224 and tested on a held-out split
of Dataset A (in-domain) and on Dataset B (never seen in training).

| Model | In-domain acc | In-domain AUC | Cross-dataset acc | Cross-dataset AUC | Gap |
|---|---|---|---|---|---|
| Logistic Regression | 0.7358 | 0.8121 | 0.4883 | 0.5063 | 24.7 |
| HOG + LBP → SVC | 0.7860 | 0.8706 | 0.4790 | 0.4751 | 30.7 |
| MobileNetV2 | 0.9430 | 0.9869 | 0.5110 | 0.5593 | 43.2 |
| EfficientNet-B0 | 0.9457 | 0.9910 | 0.5038 | 0.5285 | 44.2 |

**In-domain accuracy goes up 21 points across the four models but cross-dataset
accuracy stays near chance and the generalisation gap gets wider every time.**
The better representations were completely absorbed by the source domain.

A threshold diagnostic shows this is a *representation* failure and not a
*calibration* failure. The best accuracy possible at any threshold in either
polarity is 0.5453 so no operating point gives usable performance.

MobileNetV2 and EfficientNet-B0 are structurally very different and fail in the
same way at matched capacity so the result is not down to one architecture
choice.

### Robustness (MobileNetV2, clean baseline 0.9430)

| Perturbation | Worst setting | Accuracy | Drop |
|---|---|---|---|
| Downscale round-trip | 32px | 0.5580 | 0.385 |
| Gaussian noise | σ = 0.10 | 0.6798 | 0.263 |
| JPEG compression | quality 10 | 0.7288 | 0.214 |
| Gaussian blur | k = 7 | 0.7943 | 0.149 |
| Contrast | ±40% | 0.9048 | 0.038 |
| Brightness | ±40% | 0.9105 | 0.032 |
| **FGSM (white-box)** | **ε = 0.005** | **0.0520** | **0.891** |

Perturbations that remove high frequency information cost up to 38.5 accuracy
points whilst changing intensity by a similar visible amount costs at most 3.8.
This is consistent with the explanation that the useful signal sits in the high
frequency band (Frank et al., 2020) and with raising the input resolution from
128px to 224px being worth **+10.1 accuracy points**. It does not prove which
features the network actually uses.

---

## Install and run

### Option 1: the installer (no command line)

Download the installer for your platform from the
[latest release](https://github.com/ObaidROfficial/deepfake-detection-msc/releases/latest)
and run it.

| Platform | File | Requires |
|---|---|---|
| Windows | `OBI-DeepFake-Image-Detector-Setup.exe` | 64-bit Windows 10 or later |
| macOS | `OBI-DeepFake-Image-Detector-Setup-macOS-arm64.zip` | **Apple Silicon** (M1 or later), macOS 11+ |

**Intel Macs are not supported** as the macOS installer is built for `arm64`
only. On an Intel Mac install from source with `setup.sh` instead. The
application itself works there and it is only the compiled installer that is
tied to the architecture.

On macOS unzip the file first. The executable permission is lost in a normal
download which is why it is a `.zip` and not a plain file.

The installer sets everything up and downloads the trained weights. On Windows
it puts a shortcut on the desktop and on macOS it installs an app into the
Applications folder. **Python does not need to be installed beforehand.** If the
machine has no suitable version the installer offers to download the official
Python 3.11.9 from python.org, checks it against the published checksum and
installs it. Around 700MB is downloaded in total so it needs an internet
connection and a few minutes.

No suitable version means no Python between 3.9 and 3.13. A machine with only
Python 3.14 counts as this because numpy has no wheel for it and falls back to a
source build that fails, which is what happened when this project was first set
up. The installer checks for this at the start and installs 3.11.9 alongside
rather than failing halfway. The existing Python is left as it is and stays the
default for everything else.

The executables are not code signed (this needs a paid certificate from
Microsoft or Apple) so expect two things.

- **Windows** shows "Windows protected your PC". Click *More info* → *Run anyway*.
- **macOS** will not open it on a double-click. Right-click the file, choose
  *Open* and then confirm.

How the installer works and how to build it yourself is explained in
[installer/README.md](installer/README.md).

To remove it later run the uninstaller in the install folder on Windows, or on
macOS drag the app from Applications to the Trash. See
[Uninstalling](#uninstalling).

### Option 2: from source

This is for anyone who wants the experiment code as well as the application.

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

This option does need Python 3.9–3.13 installed already. It creates a virtual
environment, installs the CPU version of PyTorch and all dependencies, downloads
the trained weights from Releases and checks the install. Add `-Minimal`
(Windows) or `--minimal` (Unix) to install only what the demo needs and skip the
experiment dependencies.

Then:

```powershell
.\run.ps1        # or ./run.sh on macOS/Linux
```

With either option the detector opens in its own window. Choose an image and it
gives a REAL or FAKE verdict with a confidence percentage.

The interface warns when an image is outside the conditions the model was
tested on and states where the detector is known to fail. This is on purpose
because of the results above.

### Classifying images from a script

The same inference code is available as a command line tool which is useful for
batches. It needs the source install.

```powershell
python src\predict.py --image your_photo.jpg
```

```
  your_photo.jpg
    verdict     : FAKE
    P(fake)     : 0.9871
    confidence  : high
```

`predict.py` also takes `--folder` to classify a batch and `--csv` to save the
results. No datasets or Kaggle account are needed.

## For markers and reviewers

**See [REPRODUCE.md](REPRODUCE.md).** There are four options depending on how
much time you have. You can look at the committed results (~1 min), run the
detector on your own images (~5 min), check every reported number from the
released weights (~50 min) or rebuild everything from scratch (~8 h). None of
these need a GPU.

---

## Rebuilding the experiments

The two options above install the application. To run the experiments again the
datasets are also needed which means a Kaggle account. Full instructions are in
**[SETUP.md](SETUP.md)**. If you would rather set up the environment by hand than
use `setup.ps1` the short version is below.

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
├── app.py                   # GRAPHICAL INTERFACE - the main way to use this
├── setup.ps1 / setup.sh     # one-command install from source
├── run.ps1 / run.sh         # launch the interface
├── uninstall.ps1 / .sh      # remove a source install
├── installer/               # the GUI installer shipped on Releases
│   ├── installer.py         #   welcome -> options -> progress -> finish
│   ├── uninstaller.py       #   GUI uninstaller, carried inside the Windows installer
│   └── build.py             #   compiles both with PyInstaller
├── config.yaml              # every setting lives here
├── requirements.txt         # + requirements-lock.txt with exact versions
├── README.md
├── REPRODUCE.md             # for markers: four verification paths
├── SETUP.md                 # detailed manual install
├── data/
│   ├── raw/                 # downloaded datasets (gitignored)
│   └── manifests/           # seeded split CSVs (committed - this pins the experiment)
├── notebooks/
│   └── 01_data_exploration.ipynb
├── src/
│   ├── predict.py           # command line classifier + shared inference logic
│   ├── make_demo_samples.py # pull demo images from the held-out test split
│   ├── utils.py             # config loading, seeding, thread setup
│   ├── data_prep.py         # download, scan, subset, leakage assertion
│   ├── benchmark.py         # CPU speed test
│   ├── baselines.py         # logistic regression + HOG/LBP SVM
│   ├── model.py             # dataset class, transforms, model builders
│   ├── train.py             # two-stage transfer learning, checkpoint/resume
│   ├── evaluate.py          # in-domain + cross-dataset, threshold diagnostic
│   └── robustness.py        # degradation suite + FGSM
├── models/                  # trained weights (gitignored - see Releases)
└── results/                 # metrics CSVs and 25 figures
```

### Using the detector

```powershell
python app.py                                  # graphical interface
python src\predict.py --image photo.jpg        # single image
python src\predict.py --folder images\ --csv out.csv   # batch
```

### Uninstalling

**Installer on Windows.** Run `OBI-DeepFake-Image-Detector-Uninstall.exe` from
the install folder. The installer puts it there so it is not a separate
download. It removes the application, its virtual environment, the model weights
and the shortcut.

**Installer on macOS.** Drag the app from Applications to the Trash. Everything
the program uses is inside the app bundle so nothing is left behind except the
shared PyTorch cache in your home folder, which other Python software may also
use. There is no uninstaller on macOS as there is nothing for it to do.

**Installed from source.**

```powershell
.\uninstall.ps1 -DryRun      # show what would be removed, delete nothing
.\uninstall.ps1              # actually remove it
```

```bash
./uninstall.sh --dry-run
./uninstall.sh
```

By default both leave the shared PyTorch and Hugging Face caches in your home
folder alone as other Python projects on the machine use them. Add `-Caches` /
`--caches` to clear those as well. A `kaggle.json` credential file is never
touched as it belongs to you and not to this project.

### Reproducing the experiments

This needs the datasets, see [REPRODUCE.md](REPRODUCE.md).

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
- Split manifests are committed as CSVs so the exact same subset can be rebuilt
  from the same datasets
- `data_prep.py` checks that the train, validation and test splits do not
  overlap and stops with an error if they do
- All hyperparameters are in `config.yaml` and not spread through the code

CPU training is not bit-for-bit reproducible on different machines as some
backend kernels vary. The data sampling and splits are exactly reproducible and
this is the part the validity of the results depends on.

---

## Hardware

The project was developed and trained on a CPU only laptop with 12 cores, 16GB
RAM, Intel integrated graphics and no CUDA. The thread count is capped at 8 in
`config.yaml` so the machine can still be used during long runs. This limit
shaped several design decisions, mainly choosing MobileNetV2 and using a subset
rather than all 140k images. These trade-offs are discussed in the report.

One thing worth noting is that the same training epoch took anywhere between
11.4 and 48.8 minutes only because of other work running on the machine. CPU
training is much more sensitive to this than GPU training which limits how
precise any timing figures can be.

---

## Status

Experimental work is complete. Four models were trained and evaluated and the
robustness suite and adversarial evaluation were run, with all results in
`results/`. The final report has been submitted.

Total compute for all experiments was approximately **eight hours on CPU**.

The robustness suite was only run on MobileNetV2. Repeating it on
EfficientNet-B0 is left as further work.

---

## Licence

The code is released under the MIT Licence, see [LICENSE](LICENSE).

The datasets stay under their original licences and are **not** redistributed
here. Dataset A's real images come from NVIDIA's FFHQ (CC BY-NC-SA 4.0) and
Dataset B comes from OpenForensics which is released for research use. Both are
downloaded directly from Kaggle by the user with their own account.

---

## Academic integrity

This repository is part of an assessed MSc project. It is public so the work can
be checked and built on. If you are a student on the same module please use it
as a reference for the approach and cite it, but do not copy it.
