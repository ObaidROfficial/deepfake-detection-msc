# Chapter 5 — Data Exploration: raw findings

Working notes from `notebooks/01_data_exploration.ipynb`. Not report prose yet —
this is the material Chapter 5 gets written from.

Recorded: 7 August 2026

---

## 5.1 Dataset composition (confirmed, not assumed)

| | Dataset A | Dataset B |
|---|---|---|
| Source | Kaggle `xhlulu/140k-real-and-fake-faces` | Kaggle `manjilkarki/deepfake-and-real-images` |
| Provenance | Real = Flickr/FFHQ, Fake = StyleGAN ("1M Fake Faces") | OpenForensics ([Zenodo 5528418](https://zenodo.org/record/5528418)) |
| Total images | 140,000 | 190,335 |
| Class balance | 70,000 / 70,000 (exact) | 95,201 real / 95,134 fake |
| Native resolution | 256×256 JPEG (uniform) | 256×256 JPEG (uniform) |
| Official splits | train 100k / valid 20k / test 20k | train 140k / val 39k / test 11k |
| Download size | 3.75 GB | 1.68 GB |

**Integrity checks performed:**

- Zero duplicate file paths within either dataset
- Zero overlap between Dataset A and Dataset B
- All 330,335 images labelled successfully from folder structure — no unlabelled files
- Resolution uniform at 256×256 across both (sampled n=400 each), so no resizing artefacts introduced by inconsistent source sizes

## 5.2 Subsets drawn

| Split | n | Real | Fake | Drawn from |
|---|---|---|---|---|
| train | 20,000 | 10,000 | 10,000 | A, official `train` folder |
| val | 4,000 | 2,000 | 2,000 | A, official `valid` folder |
| test | 4,000 | 2,000 | 2,000 | A, official `test` folder |
| crosstest | 6,000 | 3,000 | 3,000 | B (never used for training) |

Each subset is drawn from the *corresponding* official folder rather than
randomly across the whole dataset, so results remain comparable to published
work on this benchmark. Seeded at 42; leakage assertion confirms all splits
disjoint. Manifests committed to the repository.

---

## 5.3 Domain gap — the key result

Four low-level statistics compared across n=500 images per dataset. Reported as
**AUC** rather than p-value alone: at this sample size almost any difference
reaches significance, so the p-value says little about whether a difference is
*usable*. AUC of 0.50 means a feature is useless; 1.00 means perfect separation.

**Canonical figures below are the notebook's own output (n=400, seed 42)** — the
numbers a marker regenerates by running `01_data_exploration.ipynb`. An
independent check at n=500 is given alongside for stability.

### Real vs fake *within* Dataset A

| Feature | p | AUC (n=400) | AUC (n=500 check) | Effect |
|---|---|---|---|---|
| Brightness | 0.102 | 0.547 | 0.538 | negligible, not significant |
| Contrast | 0.069 | 0.553 | 0.546 | negligible, not significant |
| **Sharpness** (gradient variance) | 1.2 × 10⁻⁴ | **0.611** | **0.631** | **moderate** |
| High-frequency energy | 0.0012 | — | 0.584 | weak |

### Dataset A vs Dataset B

| Feature | AUC (n=400) | AUC (n=500 check) | A mean | B mean | Gap |
|---|---|---|---|---|---|
| Brightness | 0.589 | 0.596 | 0.4398 | 0.4042 | weak |
| Contrast | 0.545 | 0.547 | 0.2435 | 0.2338 | negligible |
| **Sharpness** | **0.822** | **0.813** | 0.0027 | 0.0013 | **strong** |
| High-frequency energy | — | 0.796 | 0.0281 | 0.0197 | strong |

**On stability.** Brightness and contrast AUCs shift by roughly 0.01 between the
two samples, which is the same magnitude as their entire distance from 0.50.
They are sampling noise and no argument should rest on them. Sharpness moves by
0.02 on a 0.31 effect — that one is real. This distinction is worth making
explicitly in the report; reporting a "significant" AUC of 0.547 as a finding
would be a mistake.

### What this means

Two observations that only matter when read together:

1. Colour statistics are near-useless for telling real from fake (AUC 0.538 and
   0.546, neither significant). Whatever the CNN learns, it will not be a
   simple colour cue. That is reassuring — the task is not degenerate.

2. **Sharpness is simultaneously the strongest single real/fake cue inside
   Dataset A (AUC 0.631) and by far the largest difference between the two
   datasets (AUC 0.813).** Dataset B averages roughly half the gradient
   variance of Dataset A — OpenForensics images are in-the-wild captures,
   recompressed and softer, where Dataset A's StyleGAN outputs are crisp
   synthetic renders.

An AUC of 0.631 from one trivial statistic is well above chance but nowhere
near sufficient on its own, so a CNN reaching high accuracy is doing more than
exploiting global sharpness. But the direction of the bias is clear, and it
points at the dimension where the two datasets diverge most.

### Prediction for Chapter 6 — recorded before training

> A CNN trained on Dataset A will rely substantially on high-frequency /
> sharpness artefacts. Dataset B occupies a different regime on exactly that
> axis. Therefore in-domain test accuracy should be high, and cross-dataset
> accuracy on Dataset B should fall substantially — plausibly toward the
> 0.55–0.75 range rather than degrading gently.

Written down now so the Chapter 6 result is a genuine test rather than a
retrospective explanation. If the prediction is wrong, that is also worth
reporting and is more interesting than if it is right.

---

## 5.4 Compute benchmark (for Chapter 5 methodology)

Measured on the project machine: Intel 12-core (14 logical), 16 GB RAM,
Intel integrated graphics, no CUDA. PyTorch 2.13.0+cpu, 8 threads, 128×128, batch 32.

| Model | Params | Train throughput | Inference | Projected full run |
|---|---|---|---|---|
| MobileNetV2 | 2.23 M | 15.4 – 24.7 img/s | 60 – 207 img/s | 2.1 – 3.4 h |
| EfficientNet-B0 | 4.01 M | 15.0 – 18.9 img/s | 57 – 115 img/s | 2.8 – 3.5 h |

Ranges come from three runs under differing background load. **The variance is
itself worth reporting**: the fastest and slowest MobileNetV2 measurements
differ by 60% purely because of concurrent activity on the machine (a 3.75 GB
download during one run). CPU-based deep learning is far more sensitive to
system contention than GPU training, which is a practical constraint on
reproducible timing claims and belongs in the Chapter 7 limitations discussion.

### Correction — the projection was wrong by roughly 6×

The benchmark measured **full backpropagation through every parameter**. Actual
training uses **two-stage transfer learning with a frozen backbone**: stage 1
trains 2,562 parameters, stage 2 trains 414,722 of 2,226,434. Gradients never
flow through most of the network, so the backward pass costs a small fraction
of what the benchmark assumed.

Measured on a 2,000-image smoke run (7 Aug): **~143 images/second** end to end
including validation, against the benchmark's projected 15–25 img/s.

| | Benchmark projection | Measured |
|---|---|---|
| MobileNetV2, full 8-epoch run | 2.1 – 3.4 hours | **~25–30 minutes** |

**Methodological note worth including in the report.** Benchmarking full
backpropagation systematically overestimates the cost of transfer learning with
a frozen backbone — here by close to an order of magnitude. The benchmark was
still worth running: it established that the project was feasible at all, and
being wrong in the conservative direction is the right way to be wrong when
planning under a deadline. But the figure quoted in the methodology must be the
measured one, not the projection.

Revised compute budget: roughly **1.5 hours** total across both models
including the robustness suite and FGSM, against the 7 hours originally
planned. That headroom could fund larger training subsets, more epochs, or
higher input resolution — decided after seeing the baseline-configured result,
not before.

---

## 5.5 Dataset independence — verified

The cross-dataset argument depends on Datasets A and B not sharing a real-image
source. If Dataset B's reals were also drawn from FFHQ, the "different sources"
claim would collapse and Chapters 3, 5 and 6 would all need rewriting. Checked
before committing compute to CNN training.

**Documentary evidence.** OpenForensics (Le et al., ICCV 2021, arXiv:2107.14480)
describes itself as targeting *"multi-face forgery detection and segmentation
in-the-wild"*, localising forged faces *"among multiple human faces in
unrestricted natural scenes"*. FFHQ is single-face, landmark-aligned portrait
crops. Different construction entirely.

**Empirical test 1 — perceptual hash overlap.** dHash computed over 700 real
images from each dataset. **Zero identical hashes.** If both drew reals from the
same ~70,000-image FFHQ pool, roughly 7 collisions would be expected at this
sample size.

**Empirical test 2 — mean-image structure.** FFHQ is landmark-aligned, so eyes
and nose sit at fixed pixel coordinates and the average of many samples stays
structured. Unaligned in-the-wild crops average toward a featureless blob.

| | mean-image edge energy | pixel std |
|---|---|---|
| Dataset A reals | 0.7647 | 17.07 |
| Dataset B reals | 0.4511 | 10.25 |

Dataset A's average retains **1.60×** the edge structure — consistent with
alignment in A and its absence in B.

**Conclusion: strong evidence the datasets are independent. Not proof.** Test 1
is necessary but not sufficient — if OpenForensics re-cropped FFHQ source
photographs at different scales, dHashes would not match even for the same
underlying image. Test 2 is the stronger evidence. Taken with the documentary
description, the cross-dataset design is sound and the project proceeds on that
basis. The caveat is recorded here and should appear in Chapter 7 rather than
being quietly dropped.

---

## Open items

- [ ] Visual inspection of `fig_5_02` / `fig_5_03` — are Dataset A fakes obvious to the naked eye? Affects how the accuracy figures should be framed.
- [ ] Check for corrupt/unreadable files across the full subsets (currently only sampled)
- [x] ~~Confirm whether Dataset B's real images overlap FFHQ~~ — resolved, §5.5
