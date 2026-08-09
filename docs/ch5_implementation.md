# Chapter 5: Implementation

> **DRAFT v1 — 11 August 2026.** Target 1,900 words.
> Lesson 3 requires this chapter to reflect on problems experienced during
> implementation and on how the chosen methodology affected the artefact.
> Section 5.6 does that at length.

---

## 5.1 Environment

Development used Python 3.11 in a virtual environment on Windows 11. The full
dependency set is listed in `requirements.txt`, with exact installed versions
captured in `requirements-lock.txt` via `pip freeze`.

PyTorch 2.13.0 was installed from the CPU-only wheel index. This detail matters
enough to be documented in the setup guide: installing PyTorch without the CPU
index pulls approximately 2.5GB of CUDA libraries that the integrated graphics
on this machine cannot use. `torch.cuda.is_available()` returning `False` is
the expected and correct state for this project.

Thread count was capped at eight of the twelve available cores via
`torch.set_num_threads()`. Benchmarking showed diminishing returns beyond this
point due to memory bandwidth limits, and leaving headroom kept the machine
usable during multi-hour runs.

## 5.2 Data acquisition and preparation

Datasets were downloaded through the Kaggle API: 3.75GB for Dataset A and
1.68GB for Dataset B, taking approximately 46 minutes combined.

The inspection stage scanned 140,000 images in Dataset A and 190,335 in Dataset
B, labelling every one successfully from its folder path. Verification
confirmed Dataset A is exactly balanced at 70,000 per class and Dataset B is
near-balanced at 95,201 real against 95,134 fake. No duplicate paths were found
within either dataset and no overlap between them. Both are uniformly 256×256
JPEG.

Subsets were then drawn: 20,000 training, 4,000 validation and 4,000 test from
Dataset A's corresponding official folders, plus 6,000 from Dataset B for
cross-dataset evaluation. The disjointness assertion passed.

### Verifying dataset independence

Before committing compute to training, I tested whether the two datasets shared
a real-image source. If Dataset B's real images also came from FFHQ, the
cross-dataset comparison would have been substantially weakened.

Two checks were run. Perceptual hashing (dHash) over 700 real images from each
dataset found **zero** identical hashes; had both drawn from the same
70,000-image pool, roughly seven collisions would be expected. Second, because
FFHQ is landmark-aligned, averaging many of its images preserves facial
structure, whereas averaging unaligned in-the-wild crops produces a blur.
Dataset A's mean image retained 1.60 times the edge energy of Dataset B's.

Combined with the OpenForensics paper's description of "unrestricted natural
scenes", this is strong evidence of independence. It is not proof — if
OpenForensics had re-cropped FFHQ photographs at different scales, hashes would
not match regardless — and that caveat is carried into Chapter 7.

## 5.3 Baseline implementation

Both baselines share a feature-extraction and caching layer. Extraction over
20,000 images takes minutes and is repeated whenever a model is retuned, so
extracted matrices are cached to disk as `.npy` files. The cache key includes
input resolution, which prevented a real error later: after raising the CNN's
resolution from 128 to 224 pixels, a resolution-agnostic cache would have
silently reused 128px features.

Logistic regression operates on 32×32 greyscale pixels flattened to 1,024
features. HOG uses 16-pixel cells with nine orientation bins and 2×2 block
normalisation; LBP uses the uniform variant with 24 points at radius 3,
histogrammed into 26 bins. At 224×224 this yields 6,110 features per image.

Both are wrapped in a scikit-learn pipeline with `StandardScaler`. The SVM uses
`LinearSVC` with `max_iter` raised to 5,000, as the default frequently failed
to converge at this dimensionality.

Since `LinearSVC` provides no probability estimates, `decision_function` output
is used for ROC-AUC. The code detects this via `hasattr` on the pipeline, which
resolves correctly because scikit-learn pipelines only expose `predict_proba`
when the final estimator implements it.

## 5.4 CNN implementation

`model.py` holds the dataset class, transforms and model construction;
`train.py` holds the training loop. The separation lets `evaluate.py` and
`robustness.py` reconstruct the identical architecture without importing
training code.

Augmentation is deliberately mild — horizontal flip, ±10° rotation, and modest
colour jitter. Aggressive augmentation risked destroying the low-level forensic
artefacts the model needs, which would be self-defeating.

The dataset class returns a black image rather than raising on a corrupt file.
A single unreadable file should not terminate a three-hour run.

The freeze/unfreeze logic freezes the entire backbone, then in stage two
releases the trailing *n* blocks. Writing this as "freeze everything, then
selectively unfreeze" rather than conditionally toggling proved less error-prone
than the first version, which set flags twice and was harder to reason about.

Checkpointing writes every epoch, storing model state, optimiser state, epoch
number, stage and history. Best-by-validation weights are saved separately from
the most recent epoch. The resume path reconstructs the optimiser at the correct
stage — verified by simulating resume from several epochs and confirming the
optimiser rebuilds at the right boundaries.

## 5.5 Robustness implementation

Perturbations are implemented as classes with `__call__` rather than closures.
This was not a stylistic choice; the reason is in section 5.6.

FGSM required care with normalisation. Inputs reaching the model are already
ImageNet-normalised, but epsilon is meaningful in [0,1] pixel space, so the
perturbation is divided by the per-channel standard deviation to keep epsilon
interpretable as a fraction of the pixel range.

## 5.6 Problems encountered

Ten defects or misjudgements are recorded here. They are included because
Lesson 3 requires reflection on implementation problems, and because several
of them shaped the results.

**1. Benchmark overestimated training cost by seven times.** The CPU benchmark
measured full backpropagation through all parameters and projected 2.1–3.4
hours per model. Actual training keeps most of the backbone frozen, so
gradients flow through only a fraction of the network; measured throughput was
144 images per second against a projected 15–25. The benchmark was still worth
running — it established feasibility — but the methodology quotes the measured
figure. Being wrong conservatively is the right direction when planning to a
deadline.

**2. Manifests stored absolute paths.** The first implementation wrote full
Windows paths into the split CSVs. Since manifests are committed to version
control, this would have embedded the developer's directory structure and made
the splits unusable on any other machine. Fixed to store paths relative to the
project root.

**3. Oversized splits failed silently.** If the requested training size
exceeded available images, the code consumed everything and left validation and
test empty, which would only have surfaced as nonsensical evaluation output.
Now checked upfront with an explicit error naming the actual numbers.

**4. Notebook cells rendered as a single line.** The exploration notebook was
generated programmatically, and I built the `source` arrays by splitting on
newlines without retaining them. The nbformat specification requires each line
to keep its trailing newline, so Jupyter concatenated every cell into one
unreadable line and the code raised `SyntaxError`. Rebuilt using
`splitlines(keepends=True)`.

**5. Restrictive numpy pin caused a build failure.** `requirements.txt`
initially pinned `numpy<2.0.0`. On Python 3.14 this forced pip to build numpy
1.26.4 from source, which failed for lack of a C compiler; on Python 3.11 it
succeeded but held OpenCV back to an older release. Every other package in the
stack is built against the numpy 2.x ABI. The upper bound was removed.

**6. Two training runs stopped by configuration, not convergence.** The first
full run halted at eight epochs with validation loss still falling and
validation accuracy still climbing. I initially recorded 78.7% as the model's
performance. It was not — it was a snapshot of an unconverged model.
Recognising the signal (validation accuracy *above* training accuracy,
indicating underfitting rather than overfitting) led to raising the epoch
budget and letting early stopping decide. This happened twice before the budget
was generous enough.

**7. Input resolution was costing ten accuracy points.** Having ruled out
undertraining, the remaining hypothesis was that downscaling 256 to 128 pixels
was discarding the high-frequency evidence. Retraining at 224 with everything
else fixed raised in-domain accuracy from 78.7% to 88.8%. This became the
project's cleanest controlled ablation and generated the prediction that the
robustness suite subsequently confirmed. It was, in effect, a productive
mistake.

**8. Baselines and CNN were being compared at different resolutions.**
`baselines.py` extracted HOG and LBP at a hardcoded 128 pixels while the CNN
had moved to 224. Comparing them would have credited a resolution advantage to
architecture. Resolution now comes from the shared config and forms part of the
cache key.

**9. Perturbation closures could not be pickled.** The robustness suite failed
immediately on Windows with `AttributeError: Can't pickle local object
'p_jpeg.<locals>.f'`. Windows starts DataLoader workers with `spawn` rather
than `fork`, which requires pickling the dataset — including the perturbation
function — to send to each worker process. Closures returned from factory
functions cannot be pickled. Rewriting them as module-level classes with
`__call__` resolved it, since only instance attributes need serialising. The
`clean` case had passed only because it was a plain module-level function,
which made the failure look inconsistent at first.

**10. FGSM was measuring the experiment, not the model.** The initial
implementation computed gradients with respect to the ground-truth label and
reported accuracy of exactly 0.0000 with AUC of exactly 0.0000. Numbers that
clean warranted suspicion. On a binary task, stepping away from the true label
mechanically pushes every sample across the boundary regardless of the model's
actual robustness — the known label-leaking effect. Changed to use the model's
own predicted label, the realistic threat model. Accuracy became 0.0520. The
correction changed the number, not the conclusion.

Two smaller issues were also fixed: skimage warned that LBP on floating-point
images can behave unpredictably where adjacent pixels are near-equal, resolved
by computing LBP on `uint8`; and the robustness plotting grid was hardcoded at
2×3, which would have silently dropped two families once FGSM was split into
two label regimes.

## 5.7 Reflection on the methodology

The iterative approach was directly responsible for several of the above being
caught. Items 6, 7 and 10 were each found by examining a result that looked
wrong rather than by testing code in isolation — the model plateauing lower
than expected, and an adversarial result that was implausibly perfect.

The pattern worth generalising is that an anomalous number is more useful
treated as a question than as an answer. In each case the investigation
produced something better than the original: item 6 led to item 7, which
produced the project's clearest ablation; item 10 produced a defensible
adversarial figure and a methodological point worth reporting in its own right.

The cost is that this style of working is difficult to schedule. Three training
runs were needed where one was planned, and the true extent of that was not
predictable in advance. A more prescriptive plan would have been easier to
track but would have shipped the 78.7% result without questioning it.

---

*Draft word count: approximately 1,900.*
