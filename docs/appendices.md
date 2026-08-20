# Appendices

Appendices do not count toward the main body word limit. All material here is
supporting evidence for claims made in Chapters 1 to 8.

---

# Appendix A — Artefact structure and code listings

The complete source is at
<https://github.com/ObaidROfficial/deepfake-detection-msc>. Reproducing any
result requires only the repository and the weights attached to the release;
the datasets are downloaded by script. Full listings are not reproduced here,
since printed code is nearly unreadable and immediately goes stale against the
repository. This appendix records what each module does and where the claims in
Chapter 5 can be checked.

## A.1 Module inventory

| Module | Lines | Responsibility |
|---|---|---|
| `src/utils.py` | ~90 | Config loading, seeding, thread capping, logging, path resolution |
| `src/data_prep.py` | ~320 | Kaggle download, dataset scan, label inference, stratified subsetting, disjointness assertion |
| `src/benchmark.py` | ~120 | CPU throughput measurement before committing to a training budget |
| `src/baselines.py` | ~330 | Logistic regression and HOG+LBP→LinearSVC, feature caching, both-polarity threshold diagnostic |
| `src/model.py` | ~180 | Dataset class, transforms, model builders, freeze/unfreeze, scoring |
| `src/train.py` | ~300 | Two-stage transfer learning, per-epoch checkpointing, `--resume`, early stopping |
| `src/evaluate.py` | ~260 | In-domain and cross-dataset evaluation, threshold diagnostic, score dumps |
| `src/robustness.py` | ~340 | 22-setting degradation suite, FGSM in two label regimes |
| `src/predict.py` | ~380 | Shared inference logic, face detection, out-of-range checks, CLI |
| `app.py` | ~430 | Desktop window (tkinter), importing inference from `predict.py` |
| `installer/installer.py` | ~600 | Graphical installer, including Python bootstrap and checksum verification |
| `installer/uninstaller.py` | ~340 | Graphical uninstaller with self-deletion handling |

## A.2 Code supporting specific claims in the report

| Claim | Where to verify |
|---|---|
| Splits are disjoint and this is enforced, not assumed | `data_prep.py`, `assert_disjoint()` |
| Manifests store relative paths (§5.7 defect 2) | `data_prep.py`, path normalisation before write |
| Feature cache is keyed on resolution (A.5 defect 8) | `baselines.py`, `cache_key()` |
| The threshold diagnostic searches both polarities (§4.4) | `baselines.py` and `evaluate.py`, `best_threshold_accuracy()` |
| Perturbations are classes, not closures (A.5 defect 9) | `robustness.py`, module-level `__call__` classes |
| FGSM uses the predicted label by default (§5.7 defect 10) | `robustness.py`, `fgsm(..., use_true_labels=False)` |
| Epsilon is interpreted in [0,1] pixel space (§5.6) | `robustness.py`, division by per-channel std |
| The GUI and CLI cannot diverge (§4.6) | `app.py` imports `classify_pil` from `predict.py` |
| The installer refuses Python above 3.13 (§5.5) | `installer/installer.py`, `PY_MIN` / `PY_MAX` |
| The Python download is checksum-verified (§5.5) | `installer/installer.py`, `PY_MD5` and `md5_of()` |

## A.3 Configuration

Every result-affecting setting lives in `config.yaml`. The values used for the
reported results:

| Setting | Value |
|---|---|
| `project.seed` | 42 |
| `data.image_size` | 224 |
| `data.train_size` / `val_size` / `test_size` | 20,000 / 4,000 / 4,000 |
| `data.cross_test_size` | 6,000 |
| `data.respect_official_splits` | true |
| `train.epochs_frozen` / `epochs_finetune` | 3 / 32 |
| `train.unfreeze_blocks` (mobilenet_v2 / efficientnet_b0) | 3 / 2 |
| `compute.torch_threads` | 8 |
| `compute.dataloader_workers` | 4 |
| `compute.device` | cpu |

## A.4 Baseline descriptor settings

Referenced from §5.3. These are the exact settings used for the two classical
baselines; changing any of them changes the reported baseline figures.

| Baseline | Setting | Value |
|---|---|---|
| Logistic regression | Input | 32×32 greyscale, flattened |
| Logistic regression | Feature count | 1,024 |
| HOG | Cell size | 16 pixels |
| HOG | Orientation bins | 9 |
| HOG | Block normalisation | 2×2 |
| LBP | Variant | uniform |
| LBP | Points / radius | 24 / 3 |
| LBP | Histogram bins | 26 |
| HOG + LBP combined | Feature count at 224×224 | 6,110 |
| Both | Scaler | `StandardScaler` |
| HOG + LBP | Classifier | `LinearSVC`, `max_iter` = 5,000 |

`max_iter` was raised from the default because the solver frequently failed to
converge at this dimensionality.


## A.5 Implementation defects not discussed in Chapter 5

Section 5.7 discusses the four defects that changed a reported result. These four changed the tooling instead, and are recorded here because several would have produced silent failures rather than visible errors.

**8. Baselines and CNN were compared at different resolutions.** `baselines.py`
extracted HOG and LBP at a hardcoded 128 pixels while the CNN had moved to 224,
which would have credited a resolution advantage to architecture. Resolution
now comes from the shared config and forms part of the cache key.

**9. Perturbation closures could not be pickled.** The robustness suite failed
immediately on Windows with `AttributeError: Can't pickle local object
'p_jpeg.<locals>.f'`. Windows starts DataLoader workers with `spawn` rather
than `fork`, which requires pickling the dataset — including the perturbation
function — to send to each worker, and closures returned from factory functions
cannot be pickled. Rewriting them as module-level classes with `__call__`
resolved it, since only instance attributes need serialising. The `clean` case
passed only because it was a plain module-level function, which made the
failure look inconsistent at first.

**11. The demonstration tool applied the model to inputs it could not handle.**
Testing on an arbitrary photograph exposed a preprocessing gap the experiments
never met, since every dataset image is already a 256×256 face crop. A 768×1232
portrait was classified REAL with 100% reported confidence. Two faults
combined: the evaluation transform resizes to a square, squashing a 1.6:1
portrait vertically, and the face occupied roughly a fifth of the frame, so the
model saw a face about 49 pixels across surrounded by clothing and background
it had never been trained on. Fixed by cropping to the largest detected face
first, warning on aspect ratios beyond 1.35:1, and displaying the cropped
region so the user sees what was classified.

**12. The out-of-range warnings described the wrong image.** After adding face
cropping, the resolution check still ran against the original upload — a large
photograph passed every check while the face within it cropped to 149×149 and
was upscaled to 224, precisely the condition the warning exists to flag.
Reordered so inspection runs on the region actually fed to the model.

---

# Appendix B — Full results

Chapter 6 reports headline figures. This appendix gives the complete measured
output. All values are read directly from the CSV files in `results/`, which
are committed to the repository.

## B.1 In-domain performance, Dataset A test split (n = 4,000)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Predicted "fake" |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7358 | 0.7363 | 0.7345 | 0.7354 | 0.8121 | 49.9% |
| HOG + LBP → LinearSVC | 0.7860 | 0.7889 | 0.7810 | 0.7849 | 0.8706 | 49.5% |
| MobileNetV2 | 0.9430 | 0.9461 | 0.9395 | 0.9428 | 0.9869 | 49.6% |
| EfficientNet-B0 | 0.9457 | 0.9148 | 0.9830 | 0.9477 | 0.9910 | 53.7% |

## B.2 Cross-dataset performance, Dataset B (n = 6,000)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Predicted "fake" |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.4883 | 0.4513 | 0.1080 | 0.1743 | 0.5063 | 12.0% |
| HOG + LBP → LinearSVC | 0.4790 | 0.3779 | 0.0650 | 0.1109 | 0.4751 | 8.6% |
| MobileNetV2 | 0.5110 | — | 0.0827 | 0.1446 | 0.5593 | 7.2% |
| EfficientNet-B0 | 0.5038 | — | 0.0717 | 0.1262 | 0.5285 | 6.8% |

## B.3 Threshold diagnostic, Dataset B, both polarities

| Model | Conventional | Inverted | Best achievable | Polarity selected |
|---|---|---|---|---|
| Logistic Regression | 0.5180 | 0.5130 | 0.5180 | normal |
| HOG + LBP → LinearSVC | 0.5060 | 0.5317 | 0.5317 | **inverted** |
| MobileNetV2 | 0.5453 | 0.4967 | 0.5453 | normal |
| EfficientNet-B0 | 0.5220 | 0.4997 | 0.5220 | normal |

## B.4 Complete robustness suite, MobileNetV2 (clean accuracy 0.9430)

All 22 degradation settings plus both FGSM label regimes, as recorded in
`results/robustness_mobilenet_v2.csv`.

| Family | Setting | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| clean | — | 0.9430 | 0.9428 | 0.9869 |
| JPEG | q = 90 | 0.9310 | 0.9333 | 0.9837 |
| JPEG | q = 70 | 0.9113 | 0.9157 | 0.9770 |
| JPEG | q = 50 | 0.8953 | 0.9019 | 0.9692 |
| JPEG | q = 30 | 0.8650 | 0.8748 | 0.9512 |
| JPEG | q = 10 | 0.7288 | 0.7213 | 0.8094 |
| Noise | σ = 0.01 | 0.9268 | 0.9289 | 0.9814 |
| Noise | σ = 0.05 | 0.8125 | 0.8287 | 0.9126 |
| Noise | σ = 0.10 | 0.6798 | 0.7059 | 0.7552 |
| Blur | k = 3 | 0.8880 | 0.8946 | 0.9631 |
| Blur | k = 5 | 0.8518 | 0.8553 | 0.9286 |
| Blur | k = 7 | 0.7943 | 0.7823 | 0.8813 |
| Downscale | 64px round-trip | 0.7978 | 0.7897 | 0.8860 |
| Downscale | 32px round-trip | 0.5580 | 0.2432 | 0.6963 |
| Brightness | −0.4 | 0.9270 | 0.9260 | 0.9797 |
| Brightness | −0.2 | 0.9440 | 0.9434 | 0.9853 |
| Brightness | +0.2 | 0.9348 | 0.9342 | 0.9820 |
| Brightness | +0.4 | 0.9105 | 0.9083 | 0.9717 |
| Contrast | −0.4 | 0.9048 | 0.8992 | 0.9748 |
| Contrast | −0.2 | 0.9360 | 0.9348 | 0.9837 |
| Contrast | +0.2 | 0.9393 | 0.9389 | 0.9846 |
| Contrast | +0.4 | 0.9230 | 0.9213 | 0.9790 |
| FGSM (predicted label) | ε = 0.005 | 0.0520 | 0.0595 | 0.0747 |
| FGSM (predicted label) | ε = 0.01 | 0.0520 | 0.0595 | 0.0696 |
| FGSM (predicted label) | ε = 0.03 | 0.0820 | 0.0765 | 0.0620 |
| FGSM (*true* label) | ε = 0.005 | 0.0000 | 0.0000 | 0.0000 |
| FGSM (*true* label) | ε = 0.01 | 0.0000 | 0.0000 | 0.0000 |
| FGSM (*true* label) | ε = 0.03 | 0.0300 | 0.0162 | 0.0030 |

The last three rows are retained deliberately. They are the label-leaking
result discussed in §5.7 and §6.6, and the difference between them and the
three rows above is the quantitative case for using the predicted label.

## B.5 Destructive versus rescaling grouping

| Group | Families | Mean worst-case accuracy |
|---|---|---|
| Destructive | JPEG, blur, downscale | 0.6937 |
| Rescaling | brightness, contrast | 0.9076 |

Difference: 21.4 accuracy points, supporting the prediction recorded in
`robustness.py` before the suite was run.

## B.6 Training cost and capacity

| Model | Total params | Trainable params | Training time | Best val. loss |
|---|---|---|---|---|
| MobileNetV2 | 2,226,434 | 1,208,642 | 2.40 h | 0.1390 |
| EfficientNet-B0 | 4,010,110 | 1,131,954 | 4.44 h | 0.1469 |

Trainable counts are within 6.4% of each other, which is the basis for the
matched-capacity comparison in §6.2.

## B.7 Resolution ablation, MobileNetV2

| | 128 px | 224 px | Change |
|---|---|---|---|
| In-domain accuracy | 0.7873 | 0.8880 | +0.1007 |
| In-domain ROC-AUC | 0.8715 | 0.9600 | +0.0885 |

The 224 px figure here (0.8880) predates the later increase in the epoch
budget and the number of unfrozen blocks; the final MobileNetV2 result is
0.9430. The comparison is valid because both rows were produced with every
setting other than resolution held constant.

## B.8 CPU benchmark

| Model | Params | Train s/image | Inference s/image |
|---|---|---|---|
| MobileNetV2 | 2,226,434 | 0.0501 | 0.0165 |
| EfficientNet-B0 | 4,010,110 | 0.0649 | 0.0176 |

These figures measure full backpropagation and are the source of the sevenfold
overestimate discussed in §5.7. Actual training keeps most of the backbone
frozen and achieved roughly 144 images per second.

---

# Appendix C — Project plan: proposed against actual

**[TO COMPLETE BEFORE SUBMISSION — DELETE THIS PARAGRAPH. The "planned" column
must be transcribed from the Gantt chart in the approved proposal. The "actual"
column below is drawn from the project's own record — git history, checkpoint
timestamps and the results files — and is accurate as it stands.]**

| Activity | Planned (from proposal) | Actual | Variance and cause |
|---|---|---|---|
| Literature review | *[transcribe]* | Started week 2, substantially revised week 6 | Late start; §7.4 records this as the main thing I would change |
| Environment setup | *[transcribe]* | Week 1 | Redone once after Python 3.14 proved unusable (defect 5) |
| Dataset acquisition and preparation | *[transcribe]* | Week 1–2 | On plan |
| Dataset independence verification | *not planned* | Week 2 | Added; not in the proposal but necessary to the cross-dataset argument |
| Classical baselines | *[transcribe]* | Week 3 | Re-run once after the resolution change (defect 8) |
| CNN training (MobileNetV2) | *[transcribe]* | Week 3–5 | Three runs where one was planned; see §5.7 defects 6 and 7 |
| Resolution ablation | *not planned* | Week 4 | Emerged from an anomalous result |
| CNN training (EfficientNet-B0) | *[transcribe]* | Week 5 | On plan once the pipeline was settled |
| Evaluation and cross-dataset testing | *[transcribe]* | Week 5 | On plan |
| Robustness and adversarial suite | *[transcribe]* | Week 6 | Re-run after the FGSM label correction |
| Demonstration interface | *not planned* | Week 6 | Added after intermediate review (requirements F10–F12) |
| Installer and uninstaller | *not planned* | Week 7 | Added late (requirement F13) |
| Report writing | *[transcribe]* | Weeks 6–7 | Overlapped implementation rather than following it |

**Commentary.** The largest single variance is CNN training, where three runs
were needed instead of one. The cause is documented in §5.7: the first two runs
were stopped by the epoch budget rather than by convergence, and diagnosing
that led to the resolution ablation which produced the project's cleanest
result. The variance was therefore productive, but it was not planned, and
§7.4 discusses the scheduling cost of working this way.

Four activities were not in the proposal at all: dataset independence
verification, the resolution ablation, the demonstration interface and the
installer. The first two came from evidence encountered during the work; the
second two came from the intermediate review.

---

# Appendix D — Dataset licences and source integrity

## D.1 Licences

| Component | Source | Licence | Compliance |
|---|---|---|---|
| Dataset A — real images | Flickr-Faces-HQ (NVIDIA) | CC BY-NC-SA 4.0 | Academic, non-commercial use only. Satisfied. |
| Dataset A — fake images | StyleGAN-generated | Distributed with the Kaggle dataset | Research use |
| Dataset A — packaging | Kaggle: `xhlulu/140k-real-and-fake-faces` | Kaggle terms | Downloaded under the user's own account |
| Dataset B | OpenForensics (Le et al., 2021) | Research use | Cross-dataset evaluation only, never trained on |
| Dataset B — packaging | Kaggle: `manjilkarki/deepfake-and-real-images` | Kaggle terms | Downloaded under the user's own account |
| Pretrained weights | torchvision ImageNet | BSD-3-Clause | Attribution retained |

No dataset images are redistributed by this project. The repository contains
download scripts, not image data, and `demo_samples/` is gitignored for the
same reason. Users must obtain the data themselves under their own Kaggle
accounts.

## D.2 Dataset independence evidence

The cross-dataset argument requires that the two datasets do not share a
real-image source. This was tested before any CNN training.

| Check | Method | Result | Interpretation |
|---|---|---|---|
| Image overlap | dHash perceptual hashing, 700 real images sampled from each dataset | **0** identical hashes | Roughly 7 collisions would be expected if both drew from the same 70,000-image pool |
| Alignment signature | Mean-image edge energy over the same samples | Dataset A retains **1.60×** Dataset B's edge energy | Consistent with FFHQ landmark alignment and its absence in OpenForensics |
| Documentary | OpenForensics paper description | "unrestricted natural scenes" | Independent sourcing |

This is strong evidence rather than proof. Had OpenForensics re-cropped FFHQ
photographs at different scales, the hashes would not match regardless. The
caveat is carried into §7.3.

## D.3 Sources deliberately not cited

Three passages originally named sources found through database searching but
not read in full. Rather than cite work I had not read, each claim was
generalised or re-supported from a verified reference:

- **Transformer-versus-CNN degradation percentages** — removed. The figures are
  quoted inconsistently across the literature and the underlying protocol was
  not verified. §2.3 retains only the qualitative direction.
- **GAN fingerprinting and fingerprint-removal work** — the general claim is
  retained in §2.4, supported by Frank et al. (2020), without naming specific
  papers.
- **Two recent generalisation surveys** — reduced to a general statement about
  the state of the field rather than a specific attribution.

This is a deliberate trade of breadth for accuracy. Every one of the twenty
references was checked against the original paper for authors, venue, year and
the specific claim attributed to it.

## D.4 Credential and data hygiene

| Item | Handling |
|---|---|
| `kaggle.json` | Never committed; excluded by `.gitignore`; never deleted by the uninstaller, since it belongs to the user rather than the project |
| Model weights | Distributed as GitHub release assets, not committed to git history |
| Dataset images | Never committed or redistributed |
| Full manifests (~22 MB) | Not committed; regenerable in seconds. The *subset* manifests that pin the experiment **are** committed |
| Commit email | Set to a GitHub no-reply address to keep a personal address off a public repository |

---

# Appendix E — Supervisor meeting log

**[TO COMPLETE BEFORE SUBMISSION — DELETE THIS PARAGRAPH. This must be a
factual record of meetings that actually took place. Only the two items the
project's own record independently confirms are pre-filled: the proposal
approval, and the dataset-change notification referred to in §3.3. Complete the
dates, format and content from your own notes; do not submit with the
placeholders in place.]**

| # | Date | Format | Topics discussed | Actions agreed |
|---|---|---|---|---|
| 1 | *[date]* | *[format]* | Project proposal, scope, research questions | Proposal approved |
| 2 | *[date]* | *[format]* | *[topics]* | *[actions]* |
| 3 | *[date]* | *[format]* | Dataset substitution: FaceForensics++ licence delay and DFDC size against a CPU-only machine | Change approved; justification to be documented in the report (§3.3) and its cost to comparability acknowledged (§7.3) |
| 4 | *[date]* | *[format]* | Intermediate review | Artefact scope extended to a usable demonstration tool (requirements F10–F12) |
| 5 | *[date]* | *[format]* | *[topics]* | *[actions]* |
| 6 | *[date]* | *[format]* | *[topics]* | *[actions]* |

**Ethics approval.** Registered on the Arden Ethics Portal and approved before
any work began, in accordance with the Arden University Ethics Policy (Arden
University, 2023). Approval reference: *[insert]*.
