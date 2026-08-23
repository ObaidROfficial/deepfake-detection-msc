# Chapter 4: Design

> **DRAFT v1, 11 August 2026.** Target 1,400 words.

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

Two design decisions shape this structure. **Manifests are the interface**:
`data_prep.py` produces CSV manifests of paths and labels and every downstream
stage reads those rather than discovering images independently, so the exact
image set for any experiment is a small text file that can be committed and
diffed, and the baselines and CNN are guaranteed to see identical images. **All
configuration lives in one file**: no result-affecting constant is written into
code. Motivated by requirements N5 and N6, this proved its worth when the
resolution ablation needed only a single line changed.

## 4.2 Data pipeline design

The pipeline has three stages, separated so each can be rerun without repeating
the others.

**Download** retrieves both datasets via the Kaggle API, skipping if data is
already present so an interrupted download can be resumed.

**Inspect** walks the extracted directories and infers a label for every image
from its folder path. The two datasets use different naming conventions, so
label inference checks each path component against a token set rather than
assuming a fixed structure. Files that cannot be labelled are excluded and
counted rather than silently assigned a default.

**Subset** draws the working splits, with three properties designed in.
*Stratification*: equal numbers per class, so accuracy is meaningful and 50% a
genuine floor. *Respect for official splits*: Dataset A ships its own
train/validation/test folders and each subset is drawn from the corresponding
one, since a random draw across the whole dataset would put images the authors
placed in training into the test set. *Verified disjointness*: the code asserts
no path appears in more than one split and fails immediately if it does,
because a train/test leak would invalidate every number in the report.

Paths are stored relative to the project root, so manifests are portable and do
not embed the developer's directory structure.

## 4.3 Model selection

### Baselines

Two baselines were designed, with different purposes.

**Logistic regression on flattened 32×32 greyscale pixels** is a deliberately
weak floor applying no feature engineering. Its role is diagnostic: if it
scores highly, the task is trivial and the project's premise is questionable.

**HOG with Local Binary Patterns, classified by a linear SVM**, is a genuine
classical computer vision pipeline, with HOG encoding gradient orientations and
LBP local texture. It was chosen partly because exploratory analysis identified
sharpness as the strongest simple discriminator, and these are precisely
gradient and texture descriptors. Since Frank et al. (2020) locate the GAN
artefact signal in the high-frequency band, this baseline measures how much of
the task is solvable from exactly the signal expected to be
generator-specific.

### CNN

**MobileNetV2** (Sandler et al., 2018), ImageNet-pretrained, was selected as
the primary model, explicitly as a compute trade-off: depthwise separable
convolutions and an inverted residual structure give roughly 2.2 million
parameters after replacing the classifier, which trains at a workable rate on
CPU. Xception, established for this task by Rössler et al. (2019), was not
viable on the available hardware. **EfficientNet-B0** (Tan and Le, 2019) was
included as an architecture comparison, its compound scaling making it a
meaningfully different design rather than simply a larger network.

Transfer learning was chosen over training from scratch: with 20,000 images a
randomly initialised network would almost certainly underperform, and ImageNet
features are a useful starting point for low-level texture and edge detection.

### Two-stage training design

Stage one freezes the entire backbone and trains only the classifier head,
letting the new output layer reach a sensible region before gradients disturb
pretrained features. Stage two unfreezes a configurable number of trailing
blocks and fine-tunes at a learning rate an order of magnitude lower. The
number of unfrozen blocks is a parameter rather than a constant because the
right value was not known in advance; it was ultimately raised from one to
three after evidence of underfitting.

## 4.4 Experimental design

Every model is evaluated under two conditions: **in-domain**, on the held-out
test split of Dataset A, and **cross-dataset**, on Dataset B, produced by a
different manipulation method and unseen by any model at any stage. The
critical design property is that Dataset B is never used for training,
validation, hyperparameter selection or early stopping. Any use at all would
contaminate it as a measure of transfer.

### Evaluation framework

Accuracy, precision, recall, F1 and ROC-AUC are computed for every
model-condition pair. Accuracy alone is insufficient: a model predicting a
single class on a balanced set scores 50%, which both baselines and the CNN
approach on Dataset B. Recall and the predicted-positive rate expose that
pattern where accuracy conceals it.

**Threshold diagnostic.** An accuracy near chance is ambiguous between a
misplaced threshold with the ordering intact, which is recoverable, and a
representation carrying no usable information, which is not. The design
therefore computes the best accuracy achievable at any threshold, in both
polarities. If a model's AUC falls below 0.5 its ranking is inverted, and
searching only the conventional direction understates what the representation
retains; the first implementation tested one direction and would have
mislabelled a reversed-sign result as total failure.

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
detector depends on high-frequency detail, as Frank et al. (2020) would predict
for GAN imagery, then destructive perturbations should damage it substantially
more than rescaling ones. Multiple magnitudes per family make the
shape of the degradation curve visible, not just its endpoint. Brightness and
contrast were intended as controls but are imperfect ones, both being
multiplicative and so altering gradient magnitudes. Perturbations are applied
*before* the model's normal evaluation transform, so they simulate degradation
in the wild rather than an artefact of preprocessing.

**Adversarial design.** FGSM was chosen as the standard published baseline. The
design specifies using the model's own predicted label rather than the
ground-truth label, since an attacker probing a deployed detector observes only
its output. Both regimes are run so the difference can be quantified.

## 4.6 Demonstration interface design

The experiment scripts establish that the model was measured. They do not make
it usable. Four design decisions shaped the interface that does.

**One inference path, two front ends.** A desktop application (`app.py`) and a
command line tool (`predict.py`) share the same preprocessing and inference
code, the interface importing from the tool. The graphical interface is the
primary one, since the people best placed to judge whether a detector is usable
are not necessarily comfortable at a terminal. Duplicating the pipeline would
let the two diverge silently, since a difference in resizing or normalisation
would give different verdicts for the same image with nothing to signal it.

**Face detection before classification.** The model was trained on tightly
cropped 256×256 faces. In a half-length portrait the face may occupy a fifth of
the frame, so after resizing the model sees mostly background. The interface
detects and crops to the largest face first, and displays the cropped region so
the user can see what was actually classified.

**The interface reports its own limits.** The evaluation established that the
detector approaches chance on unseen manipulation methods, degrades sharply
under compression, and fails toward "real" rather than toward uncertainty. An
interface returning "REAL, 100% confident" would be at its most confident
precisely when least reliable. Three measures address this: inputs are checked
against the validated conditions, with warnings citing the *measured* accuracy
for that condition; a standing notice states that diffusion-generated images
are out of scope, since users reach for those first; and confidence is
presented as an uncalibrated relative score, because softmax outputs are
systematically overconfident.

**Installation should not itself require expertise.** An interface a
non-developer can use is worth nothing if reaching it means cloning a
repository and running a shell script, so the design treats installation as
part of the deliverable rather than as documentation. A graphical installer
handles the environment, dependencies, weights and a shortcut, and a matching
uninstaller removes them. Weights are published as release assets rather than
committed, so a reviewer reaches a working detector without the datasets or any
training. Section 5.5 covers the build and the circular dependency it had to
work around.

## 4.7 Reproducibility design

A single seed in `config.yaml` propagates to Python's `random`, NumPy and
PyTorch. Split manifests are committed, weights are saved with their epoch and
validation metrics, and all figures are regenerated from saved result files
rather than hand-assembled.

One honest limitation: CPU training is not bit-for-bit deterministic across
machines, because some backend kernels vary. What is exactly reproducible is
the data sampling, the splits and the evaluation protocol. Those are the
elements affecting the validity of the comparisons, rather than the third
decimal place of a metric.

---

*Draft word count: approximately 1,400.*
