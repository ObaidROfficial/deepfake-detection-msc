# Chapter 7: Critical Evaluation

> **DRAFT v1, 11 August 2026.** Target 1,700 words.
> Covers: answers to the research questions, artefact strengths and weaknesses,
> evaluation of the process, and LO7 (ethics, sustainability, social
> responsibility, global context) which Lesson 4 requires explicitly.

---

## 7.1 Answers to the research questions

**RQ1: How effective are machine learning models in detecting deepfake media?**

Within the distribution they are trained on, very effective: MobileNetV2
reached 94.3% at an AUC of 0.987, and even logistic regression on 32×32
greyscale pixels reached 73.6%. Outside it, ineffective, since all four fell to
approximately chance on an unseen manipulation method. The answer therefore
depends entirely on whether the question means "on the test set" or "in
deployment", and for the best model those two readings differ by 44.2 accuracy
points.

**RQ2: Which features or patterns are most useful in distinguishing real and
fake media?**

High-frequency detail, on three converging lines of evidence: raising
resolution from 128 to 224 pixels was worth ten accuracy points; perturbations
removing high-frequency content cost up to 38.5 points against at most 3.8 for
intensity rescaling; and sharpness was the strongest single low-level
discriminator in the exploratory analysis (AUC 0.611, against 0.547 and 0.553
for brightness and contrast). This aligns with Frank et al. (2020), who traced
GAN frequency artefacts to generator upsampling. The useful features appear to
be generator fingerprints rather than semantic indicators of manipulation.

**RQ3: How does the performance of the proposed model compare to baseline
approaches?**

In-domain, substantially better: 94.3% and 94.6% for the two CNNs against 78.6%
for HOG+LBP and 73.6% for logistic regression. Cross-dataset, no better in any
practical sense. All four sit within three points of chance and no threshold in
either polarity recovers usable accuracy. The CNNs' AUCs of 0.5593 and 0.5285
are statistically above chance, but best achievable accuracies of 54.5% and
52.2% are not usable detectors. At matched capacity, EfficientNet-B0 gained 0.3
accuracy points in-domain for 85% more training time; architectural
sophistication bought very little here.

**RQ4: How robust is the detection model against manipulated or adversarial
inputs?**

Moderately robust to mild degradation, fragile to severe degradation, and
almost completely defenceless against white-box adversarial attack. It holds up
at JPEG quality 50 and above (≥ 0.895) but falls to 0.558 when downscaled to
thumbnail size. Under FGSM at an imperceptible ε of 0.005, accuracy collapses
from 0.943 to 0.052 and the ranking inverts.

**RQ5: What are the key limitations of current deepfake detection techniques?**

The limitation documented most clearly here is that in-domain accuracy and
cross-domain reliability are not merely different but appear to be in tension.
Across four models the generalisation gap widened monotonically as in-domain
accuracy improved: 24.7, 30.7, 43.2, 44.2 points. If detectors succeed by
exploiting generator-specific artefacts, a better detector is partly one
exploiting them more thoroughly, which offers no protection against a generator
it has not seen.

## 7.2 Strengths of the artefact

The experimental design supports the conclusions drawn from it. Splits are
seeded and verified disjoint by assertion, manifests are committed so the exact
subsets can be reconstructed, all hyperparameters live in one configuration
file, and every result in Chapter 6 can be regenerated from a clean clone.

Dataset independence was verified rather than assumed, before any compute was
committed. Perceptual hashing found zero overlap across 1,400 sampled real
images, and Dataset A's mean image retains 1.6 times the edge structure of
Dataset B's.

The threshold diagnostic separating calibration failure from representation
failure adds something beyond reporting accuracy: two models can both score 49%
for entirely different reasons, and which one applies changes what conclusion
is available. Several predictions were also recorded before the experiments
that tested them. The robustness prediction was written into the module
docstring before the suite ran, which makes those genuine tests rather than
post-hoc rationalisation.

The artefact is also usable by somebody who did not build it: a graphical
installer handles the environment, dependencies and weights, and the interface
needs no command line. That is not an academic contribution, but a detector
nobody but its author can run is hard to assess, and the requirement to
communicate the model's limits (F11) means nothing if nobody can reach the
interface displaying them.

## 7.3 Weaknesses and limitations

**Scale.** Training used 20,000 of the 100,000 available images. The final
model was still improving when early stopping halted it, and in-domain accuracy
sits roughly three points below published figures for this dataset. More data
and a fully unfrozen backbone would likely close that gap.

**Two datasets, one comparison.** The cross-dataset finding rests on a single
pair, so it is suggestive rather than conclusive. A three- or four-way matrix
would establish whether the effect is general or specific to these two
manipulation types.

**Both datasets are GAN-era, and this limitation is sharper than it first
appears.** GANs construct images by repeatedly upsampling, which Frank et al.
(2020) identify as the source of the artefacts detectors exploit. Diffusion
models generate by iterative denoising instead, so that artefact family is
largely absent, and the better this model becomes at finding GAN upsampling
traces the less reason there is to expect it to find anything in a diffusion
image. Corvi et al. (2023) put exactly this question to GAN-built detectors.

Testing bore this out: a visibly synthetic diffusion-generated portrait was
classified **REAL with reported confidence of 100%**, even after the face was
correctly detected and cropped (§5.7). This is consistent with the
cross-dataset result rather than separate from it. Dataset B was a different
manipulation method within the same generative family, so a larger failure
across families is what the account predicts.

The confidence is the more troubling part. A softmax output from a
cross-entropy-trained network is not a probability of being correct, and here it
was actively misleading: a user without this report would have received an
unambiguous, confident, wrong answer. That is precisely the failure mode
Chapter 8 argues is worse than no detector at all.

**Deviation from the proposal.** FaceForensics++ requires a signed licence with
turnaround measured in weeks and DFDC is approximately 470GB of video that
cannot be decoded on a CPU-only machine in the timeframe. The substitutes are
defensible but cost comparability, since most published results use FF++.

**Imperfect controls.** Brightness and contrast served as the
detail-preserving comparison in the robustness suite, but both are
multiplicative and do alter gradient magnitudes. On a sample image, contrast
−0.4 reduced high-frequency energy to 0.60 of the original. The argument rests
on rescaling being near-invertible and structure-preserving rather than on it
leaving frequency content untouched, a weaker claim than a genuinely clean
control would support.

**Robustness tested on one architecture.** Both CNNs were trained to
completion, so the *generalisation* finding is no longer single-architecture,
but the robustness suite was run only on MobileNetV2. Whether the degradation
profile and adversarial fragility are architecture-specific remains untested,
though there is little reason to expect much difference given both collapse
identically on transfer.

**Adversarial scope.** Only FGSM, only white-box. Stronger iterative attacks
and realistic black-box transfer attacks were not attempted.

**The installer's bootstrap path remains untested.** Install, launch and
uninstall were exercised end to end on both Windows and macOS, using the
published binaries rather than the local build, and all three succeeded. What
that does not cover is the branch that matters most: a machine with no suitable
Python, where the installer downloads and installs one. Both test machines
already had 3.11, so that path is still verified only by reasoning. The testing
that was done immediately found defect 17, where the installer handed its own
Tcl paths to the program it had just installed, so the detector would not start
from the installer's launch button while working perfectly from the shortcut.
With defects 13 and 14 having appeared only on a second machine, three faults
were invisible from the development machine by the development route, which is
reason enough to treat the untested branch as a real risk.

## 7.4 Evaluation of the process

The Agile approach suited this project: several conclusions changed as evidence
accumulated, and a Waterfall plan fixed at proposal stage would have handled
that badly. Three instances shaped the outcome. The CPU benchmark overestimated
training time roughly sevenfold. The first two training runs stopped on the
epoch count rather than convergence, and I reported 78.7% before noticing
validation accuracy still climbing and above training accuracy. And the first
FGSM implementation computed gradients against the ground-truth label,
producing accuracy of exactly zero; asking why a number was *too* clean
identified label-leaking. In all three the useful step was treating an
anomalous result as a question rather than an answer.

**What I would do differently.** I would fix input resolution by experiment at
the start rather than defaulting to 128 pixels for speed; that one choice cost
ten accuracy points and two full training runs. And I would start the
literature review in week one rather than fitting it around the implementation,
because it turned out to explain results I had already observed and would have
let me predict them instead.

## 7.5 Ethical, social and professional considerations

**Ethics approval and data.** This project used only secondary data from
publicly available datasets and involved no human participants, so participant
consent, debriefing and company sign-off under the Arden University Ethics
Policy (QA 39) did not apply. Ethical approval was obtained before work began
and no personal data as defined by UK GDPR was processed. The datasets contain
images of identifiable individuals, but are distributed for research under
their own licences and were not redistributed here, since the repository
carries download instructions rather than image files.

**Dual-use risk.** Detection research is inherently dual-use, and the
robustness results identify precise conditions under which this class of
detector fails, which is of interest to defenders and evaders alike. I judged
publication appropriate because these weaknesses are already documented in the
peer-reviewed literature, the attack used is a standard published baseline, and
defenders gain more from knowing the operating limits of their tools than
attackers gain from confirming a known vulnerability. The BCS Code of Conduct's
requirement of due regard for public wellbeing supports disclosing limitations
rather than concealing them.

**Social responsibility and the deployment question.** The most socially
significant finding is the failure direction. When the evidence it relies on is
unavailable this detector does not report uncertainty; it declares images
authentic. Deployed on compressed social media content it would systematically
under-report manipulation while appearing to work normally. Given deepfakes are
used for non-consensual imagery, fraud and disinformation, a detector that
silently fails open could do more harm than none, because it confers
unwarranted confidence.

**Sustainability.** Total compute across all experiments was approximately
eight hours on a CPU-only consumer laptop, negligible against the resources
used to train the generators these models detect. The asymmetry is worth
noting: generation is far more compute-intensive than detection and also far
better funded, so the environmental cost of the arms race falls
disproportionately on the generation side.

**Global context.** Detection performance is bounded by training data, and face
datasets have well-documented demographic imbalances. Neither dataset here
provides demographic metadata, so performance disparities across groups could
not be tested. This is a genuine gap: a detector that works less well for
under-represented populations would distribute the harms of deepfakes unevenly,
and those least protected would be the people already most vulnerable to
image-based abuse.

---

*Draft word count: approximately 1,700.*
