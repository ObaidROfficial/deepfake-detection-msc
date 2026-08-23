# Chapter 5: Implementation

> **DRAFT v1, 11 August 2026.** Target 1,900 words.
> Lesson 3 requires this chapter to reflect on problems experienced during
> implementation and on how the chosen methodology affected the artefact.
> Section 5.7 does that at length.

---

## 5.1 Environment

Development used Python 3.11 in a virtual environment on Windows 11, with
dependencies listed in `requirements.txt` and exact installed versions in
`requirements-lock.txt`. PyTorch 2.13.0 was installed from the CPU-only wheel
index. Installing without it pulls roughly 2.5GB of CUDA libraries the
integrated graphics cannot use, so `torch.cuda.is_available()` returning
`False` is the correct state here. Thread count was capped at eight of twelve
cores, since benchmarking showed diminishing returns beyond that.

## 5.2 Data acquisition and preparation

Datasets were downloaded through the Kaggle API: 3.75GB for Dataset A and
1.68GB for Dataset B, about 46 minutes combined. Inspection scanned 140,000
images in Dataset A and 190,335 in Dataset B, labelling every one from its
folder path. Dataset A is exactly balanced at 70,000 per class, Dataset B
near-balanced at 95,201 real against 95,134 fake, with no duplicate paths
within either and no overlap between them. Both are uniformly 256×256 JPEG.

Subsets were drawn from Dataset A's official folders, being 20,000 training,
4,000 validation and 4,000 test, plus 6,000 from Dataset B for cross-dataset
evaluation. The disjointness assertion passed.

### Verifying dataset independence

Before committing compute to training, I tested whether the two datasets shared
a real-image source, since if Dataset B's real images also came from FFHQ the
cross-dataset comparison would be substantially weakened.

Two checks were run. Perceptual hashing (dHash) over 700 real images from each
dataset found **zero** identical hashes, where roughly seven collisions would
be expected had both drawn from the same 70,000-image pool. Second, because
FFHQ is landmark-aligned, averaging many of its images preserves facial
structure whereas averaging unaligned in-the-wild crops produces a blur.
Dataset A's mean image retained 1.60 times the edge energy of Dataset B's.

This is strong evidence of independence but not proof. Had OpenForensics
re-cropped FFHQ photographs at different scales, hashes would not match
regardless. That caveat is carried into Chapter 7.

## 5.3 Baseline implementation

Both baselines share a feature-extraction and caching layer, since extraction
over 20,000 images takes minutes and is repeated whenever a model is retuned.
The cache key includes input resolution, which prevented a real error later: on
raising the CNN's resolution from 128 to 224 pixels, a resolution-agnostic
cache would have silently reused 128px features.

Exact descriptor settings are tabulated in Appendix A.4. One implementation
detail is worth surfacing. `LinearSVC` gives no probability estimates, so
`decision_function` output is used for ROC-AUC, detected via `hasattr` on the
pipeline.

## 5.4 CNN implementation

`model.py` holds the dataset class, transforms and model construction, and
`train.py` holds the training loop. The separation lets `evaluate.py` and
`robustness.py` reconstruct the identical architecture without importing
training code.

Augmentation is deliberately mild, being horizontal flip, ±10° rotation and
modest colour jitter, since aggressive augmentation risked destroying the
low-level forensic artefacts the model depends on. The dataset class returns a
black image rather than raising on a corrupt file, because a single unreadable
file should not terminate a three-hour run.

The freeze/unfreeze logic freezes the entire backbone, then in stage two
releases the trailing *n* blocks. Writing this as "freeze everything, then
selectively unfreeze" proved less error-prone than the first version, which set
flags twice. Checkpointing writes every epoch, keeping best-by-validation
weights separate from the most recent, and the resume path reconstructs the
optimiser at the correct stage.

## 5.5 Demonstration tools

Two entry points let the trained model be used rather than only measured.
`app.py` is a desktop window showing the classified region, the verdict, both
class probabilities and any out-of-range warnings. `src/predict.py` is a
command line tool taking an image or a folder, used for batches during
development. The window imports its inference logic from `predict.py` rather
than reimplementing it, so a preprocessing difference cannot arise silently.

The window was originally built with Gradio, which serves a local web page and
opens a browser tab. That worked, but it meant importing a web framework and an
ASGI server to display one image and one number, and ten to twenty seconds
passed before anything appeared on screen. That was long enough that the
launcher had to be given a visible console so the wait did not look like a
crash. Rebuilding it in tkinter, already in the standard library and already
used by the installer, cut startup to two or three seconds and dropped a web
stack the install never used as a web application.

A second argument, raised in supervision, mattered more than the engineering
one. A browser interface asks the user to accept that a program they just
installed has opened a network port and pointed their browser at it. For most
software that passes unnoticed, but for a tool whose whole purpose is telling
the user what to trust, it invites exactly the suspicion the tool is meant to
resolve, and it is also what a good deal of actual malware looks like. A window
that opens and does nothing else makes no such request. The cost is that the
interface becomes platform-specific, which is most of what follows.

Both entry points apply face detection before classification, warn when an
input falls outside the validated conditions by citing the *measured* accuracy
for that condition rather than a generic caution, and carry a standing notice
that diffusion-generated images are out of scope. That follows from the
argument in Chapter 8 that a detector silently failing while appearing to work
is worse than none.

### Packaging for a non-technical user

Shell scripts (`setup.ps1`, `setup.sh`) install everything in one command, but
still assume the user is comfortable at a terminal and already has Python.
Neither is safe to assume of anybody assessing the work who is not a developer.
A graphical installer and uninstaller were therefore added, compiled to
standalone executables with PyInstaller.

The awkward part is a circular dependency that is not obvious until you hit it.
PyInstaller bundles a Python runtime, so the installer launches on a machine
with nothing installed. But that runtime is frozen inside the executable and
can neither create a virtual environment nor run pip, so a real interpreter is
still needed. Bundling Python's official embeddable distribution fails too,
because it deliberately omits tkinter and pip. The resolution was to have the
installer fetch the official python.org installer and run it, checking the
download against the published checksum first.

It installs 3.11.9, matching the development environment, and *refuses*
anything above 3.13. That upper bound is not caution for its own sake. This
project's own setup originally failed on Python 3.14 because numpy had no wheel
and fell back to a source build that did not compile (defect 5 below).
Accepting an interpreter that new would reproduce that failure on the user's
machine, part-way through a 700MB download, as a compiler error they could do
nothing about. Neither executable is code-signed, so SmartScreen and Gatekeeper
warn on first run, and the README documents this rather than leaving it a
surprise.

## 5.6 Robustness implementation

Perturbations are implemented as classes with `__call__` rather than closures.
That is not a stylistic choice, and the reason is defect 9 in Appendix A.5.
FGSM required care with normalisation. Inputs reaching the model are already
ImageNet-normalised but epsilon is meaningful in [0,1] pixel space, so the
perturbation is divided by the per-channel standard deviation to keep epsilon
interpretable as a fraction of the pixel range.

## 5.7 Problems encountered

Seventeen defects or misjudgements were encountered. The four that materially
shaped the *results* are discussed below. Four that shaped the tooling rather
than the findings are in Appendix A.5, and the remaining nine are tabulated.
They are recorded because Lesson 3 requires reflection on implementation
problems.

**1. Benchmark overestimated training cost by seven times.** It measured full
backpropagation through all parameters and projected 2.1 to 3.4 hours per
model, but training keeps most of the backbone frozen. Measured throughput was
144 images per second against a projected 15 to 25. Being wrong conservatively
is the right direction when planning to a deadline, but the methodology quotes
the measured figure.

The remaining nine were minor in isolation but are recorded because several
would have produced invalid results or silent failures rather than visible
errors. Defects 13 and 14 appeared only on a second machine, and defect 17 only
when the program was started the way a user starts it. Together they are the
argument for testing a deliverable somewhere other than where it was built, by
the route it will actually be used.

| # | Defect | Consequence if unfixed |
|---|---|---|
| 2 | Manifests stored absolute Windows paths | Committed splits unusable on any other machine |
| 3 | Oversized training size silently emptied val/test | Invalid results, visible only as nonsense metrics |
| 4 | Notebook cells generated without trailing newlines | Every cell concatenated to one line; `SyntaxError` |
| 5 | `numpy<2.0.0` pin forced a source build | Install failure on Python 3.14; OpenCV held back |
| 13 | OpenCV unpinned | macOS installed OpenCV 5, where `CascadeClassifier` has moved; face detection failed while Windows worked |
| 14 | Emphasis in the GUI banner had no explicit colour | Inherited white from the dark theme, rendering the bold words invisible on a light background |
| 15 | A guard in the uninstaller was written as a chained conditional expression rather than a conditional statement | Parsed as valid Python and happened to give the right answer, so it passed testing while meaning something other than intended |
| 16 | The setup scripts checked that a Python existed but not which version | On a machine with only Python 3.14 they would proceed and then fail at the numpy build, reproducing defect 5 on the user's machine instead of the developer's |
| 17 | The frozen installer handed its own `TCL_LIBRARY` down to the program it had just installed | The detector failed to start with `Can't find a usable init.tcl` when launched from the installer's own tick box, although the install was correct and the desktop shortcut worked |

Two smaller issues were also fixed. Skimage warned that LBP on floating-point
images behaves unpredictably where adjacent pixels are near-equal, resolved by
computing on `uint8`. And the robustness plotting grid was hardcoded at 2×3,
silently dropping two families once FGSM split into two label regimes.

**6. Two training runs stopped by configuration, not convergence.** The first
full run halted at eight epochs with validation loss still falling and accuracy
still climbing, and I initially recorded 78.7% as the model's performance. It
was a snapshot of an unconverged model. Recognising the signal, which was
validation accuracy *above* training accuracy and therefore underfitting rather
than overfitting, led to raising the epoch budget and letting early stopping
decide. This happened twice before the budget was generous enough.

**7. Input resolution was costing ten accuracy points.** Having ruled out
undertraining, the remaining hypothesis was that downscaling 256 to 128 pixels
discarded the high-frequency evidence. Retraining at 224 with everything else
fixed raised in-domain accuracy from 78.7% to 88.8%. This became the project's
cleanest controlled ablation and generated the prediction the robustness suite
later confirmed, so in effect it was a productive mistake.

**10. FGSM was measuring the experiment, not the model.** The initial
implementation computed gradients against the ground-truth label and reported
accuracy of exactly 0.0000 with AUC of exactly 0.0000. Numbers that clean
warranted suspicion. On a binary task, stepping away from the true label
mechanically pushes every sample across the boundary regardless of the model's
actual robustness, which is the known label-leaking effect. Using the model's
own predicted label, the realistic threat model, gave 0.0520. The correction
changed the number, not the conclusion.

## 5.8 Reflection on the methodology

The iterative approach was directly responsible for several of the above being
caught. Items 6, 7 and 10 were each found by examining a result that looked
wrong rather than by testing code in isolation, one being a model plateauing
lower than expected and the other an adversarial result that was implausibly
perfect. The pattern worth generalising is that an anomalous number is more
useful treated as a question than as an answer. Item 6 led to item 7 and the
project's clearest ablation, and item 10 produced both a defensible adversarial
figure and a methodological point worth reporting in its own right.

The cost is that this style of working is hard to schedule. Three training runs
were needed where one was planned. A more prescriptive plan would have been
easier to track but would have shipped the 78.7% result without questioning it.

---

*Draft word count: approximately 1,900.*
