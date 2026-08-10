# Chapter 4: Design

> **DRAFT v1 — 11 August 2026.** Target 1,400 words.

---

## 4.1 System architecture

The artefact is a pipeline of five stages, each implemented as a separate
module that can be run independently:

```
   Kaggle API
       │
       ▼
 [ data_prep.py ] ──▶ manifests (CSV)  ── seeded, disjoint, committed to git
       │
       ├──▶ [ baselines.py ] ──▶ classical model metrics
       │
       ├──▶ [ train.py ] ──▶ model weights (.pt) + training history
       │           │
       │           ├──▶ [ evaluate.py ]   ──▶ in-domain + cross-dataset metrics
       │           └──▶ [ robustness.py ] ──▶ degradation + adversarial metrics
       │
       └──▶ [ 01_data_exploration.ipynb ] ──▶ figures for Chapter 6
```

Two design decisions shape this structure.

**Manifests as the interface.** `data_prep.py` produces CSV manifests of file
paths and labels, and every downstream stage reads those rather than
discovering images independently. The exact image set for any experiment is
therefore a small text file that can be committed, inspected and diffed, and
the baselines and CNN are guaranteed to be evaluated on identical images.

**Configuration in one file.** All hyperparameters, paths and experiment
settings live in `config.yaml`; no result-affecting constant is written into
code. Motivated by requirements N5 and N6, this proved valuable in practice —
the resolution ablation required changing a single line.

## 4.2 Data pipeline design

The pipeline has three stages, separated so each can be rerun without repeating
the others.

**Download** retrieves both datasets via the Kaggle API. It checks for existing
data and skips if present, so an interrupted download can be resumed.

**Inspect** walks the extracted directories and infers a label for every image
from its folder path. The two datasets use different naming conventions —
lowercase `real`/`fake` in one, capitalised `Real`/`Fake` in the other — so
label inference checks each path component against a token set rather than
assuming a fixed structure. Files that cannot be labelled are excluded and
counted rather than silently assigned a default.

**Subset** draws the working splits. Three properties were designed in:

*Stratification.* Every split is drawn with equal numbers per class, so
accuracy is meaningful and the 50% chance level is a genuine floor.

*Respect for official splits.* Dataset A ships its own train/validation/test
folders. Each subset is drawn from the corresponding official folder rather
than randomly across the whole dataset. A random draw would still have been
internally valid, but the test set would contain images the dataset authors
placed in training, making comparison against published results on this
benchmark awkward.

*Verified disjointness.* After the splits are written, the code asserts that
no path appears in more than one split and fails immediately if it does. A
train/test leak would invalidate every number in the report, and the check
costs a fraction of a second.

Paths are stored relative to the project root rather than absolute, so
manifests are portable and do not embed the developer's directory structure.

## 4.3 Model selection

### Baselines

Two baselines were designed, with different purposes.

**Logistic regression on flattened 32×32 greyscale pixels** is a deliberately
weak floor. It applies no feature engineering whatsoever. Its role is
diagnostic: if this scores highly, the task is trivial and the project's
premise is questionable.

**HOG combined with Local Binary Patterns, classified by a linear SVM**, is a
genuine classical computer vision pipeline. HOG encodes the distribution of
gradient orientations; LBP encodes local texture. This baseline was chosen
partly because exploratory analysis had identified sharpness as the strongest
simple discriminator, and HOG and LBP are precisely gradient and texture
descriptors. Since Frank et al. (2020) locate the GAN artefact signal in the
high-frequency band, this baseline directly measures how much of the task is
solvable from exactly the signal expected to be generator-specific.

### CNN

**MobileNetV2** (Sandler et al., 2018), ImageNet-pretrained, was selected as
the primary model. The reasoning is explicitly a compute trade-off: its
depthwise separable convolutions and inverted residual structure give roughly
2.2 million parameters after replacing the classifier, which trains at a
workable rate on CPU. Xception, established as a strong baseline for this task
by Rössler et al. (2019) and common in the literature since, was not viable on
the available hardware.

**EfficientNet-B0** (Tan and Le, 2019) was included as an architecture
comparison. Its compound scaling approach — scaling depth, width and resolution
together rather than independently — makes it a meaningfully different design
rather than simply a larger network.

Transfer learning was chosen over training from scratch. With 20,000 training
images, a randomly initialised network would almost certainly underperform, and
ImageNet features provide a useful starting point for low-level texture and
edge detection.

### Two-stage training design

Training proceeds in two stages. Stage one freezes the entire backbone and
trains only the classifier head, allowing the new output layer to reach a
sensible region before gradients are allowed to disturb pretrained features.
Stage two unfreezes a configurable number of trailing feature blocks and
fine-tunes at a learning rate an order of magnitude lower.

The number of unfrozen blocks is a parameter rather than a constant because the
correct value was not known in advance. It was ultimately raised from one to
three after evidence of underfitting.

## 4.4 Experimental design

Every model is evaluated under two conditions:

**In-domain.** Held-out test split of Dataset A. Measures conventional
performance.

**Cross-dataset.** Dataset B, unseen by any model at any stage, produced by a
different manipulation method. Measures generalisation.

The critical design property is that Dataset B is never used for training,
validation, hyperparameter selection or early stopping. Any use at all would
contaminate it as a measure of transfer.

### Evaluation framework

Accuracy, precision, recall, F1 and ROC-AUC are computed for every
model-condition pair. Accuracy alone is insufficient: a model predicting a
single class on a balanced set scores 50%, and both baselines and the CNN
approach exactly this behaviour on Dataset B. Recall and the predicted-positive
rate expose that pattern where accuracy conceals it.

**Threshold diagnostic.** An accuracy near chance is ambiguous between two very
different failures — a misplaced decision threshold with the ordering intact,
which is recoverable, and a representation carrying no usable information,
which is not. The design therefore includes computing the best accuracy
achievable at any threshold, searched over the score distribution.

This must be evaluated in both polarities. If a model's AUC falls below 0.5 its
ranking is inverted, and searching only the conventional direction understates
what the representation retains. The first implementation tested one direction
only and would have mislabelled a reversed-sign result as a total failure.

## 4.5 Robustness suite design

Six perturbation families were selected to span two categories:

| Family | Settings | Category |
|---|---|---|
| JPEG compression | q = 90, 70, 50, 30, 10 | Destructive |
| Gaussian blur | k = 3, 5, 7 | Destructive |
| Downscale round-trip | 64px, 32px | Destructive |
| Gaussian noise | σ = 0.01, 0.05, 0.10 | Additive |
| Brightness shift | ±20%, ±40% | Rescaling |
| Contrast shift | ±20%, ±40% | Rescaling |

The destructive/rescaling split is the design's analytical core. If the
detector depends on high-frequency detail — as Frank et al. (2020) would
predict for GAN-generated imagery — destructive perturbations should damage it
substantially more than rescaling ones. Multiple magnitudes per
family were included so the shape of the degradation curve is visible, not just
its endpoint.

Brightness and contrast were intended as controls but are imperfect ones, since
both are multiplicative and do alter gradient magnitudes. This limitation was
measured and documented rather than discovered later.

Perturbations are applied to the source image *before* the model's normal
evaluation transform, so they simulate degradation occurring in the wild rather
than an artefact of preprocessing.

**Adversarial design.** FGSM was chosen as the standard published baseline
attack. The design specifies that the attack use the model's own predicted
label rather than the ground-truth label, since an attacker probing a deployed
detector observes only its output. Both regimes are run so the difference can
be quantified.

## 4.6 Demonstration interface design

The experiment scripts establish that the model was measured. They do not make
it usable. Three design decisions shaped the interface that does.

**One inference path, two front ends.** A command line tool (`predict.py`) and
a web interface (`app.py`) share the same preprocessing and inference code,
the interface importing from the tool. Duplicating the pipeline would let the
two diverge silently — a difference in resizing or normalisation would give
different verdicts for the same image with nothing to signal it.

**Face detection before classification.** The model was trained on tightly
cropped 256×256 faces. In a half-length portrait the face may occupy a fifth of
the frame, so after resizing the model sees mostly background. The interface
detects and crops to the largest face first, and displays the cropped region so
the user can see what was actually classified.

**The interface reports its own limits.** The evaluation established that the
detector approaches chance on unseen manipulation methods, degrades sharply
under compression, and fails toward "real" rather than toward uncertainty. An
interface returning "REAL, 100% confident" would be at its most confident
precisely when least reliable.

Three measures address this. Inputs are checked against the validated
conditions, with warnings citing the *measured* accuracy for that condition. A
standing notice states that diffusion-generated images are out of scope, since
users reach for those first. And confidence is presented as an uncalibrated
relative score, because softmax outputs are systematically overconfident.
Chapter 8 argues a detector that silently fails while appearing to work is
worse than none; applying that to this project's own deliverable was the
minimum consistency required.

**Distribution.** Weights are published as release assets rather than
committed, and a setup script retrieves them automatically, so a reviewer
reaches a working detector without the datasets or any training.

## 4.7 Reproducibility design

A single seed in `config.yaml` propagates to Python's `random`, NumPy and
PyTorch. Split manifests are committed. Model weights are saved with their
epoch and validation metrics. All figures are regenerated from saved result
files rather than hand-assembled.

One honest limitation: CPU training is not bit-for-bit deterministic across
different machines, because some backend kernels vary. What is exactly
reproducible is the data sampling, the splits and the evaluation protocol —
the elements that affect the validity of the comparisons rather than the third
decimal place of a metric.

---

*Draft word count: approximately 1,400.*
