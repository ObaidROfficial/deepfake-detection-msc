# Chapter 7: Critical Evaluation

> **DRAFT v1, 11 August 2026.** Target 1,700 words.
> Covers: answers to the research questions, artefact strengths and weaknesses,
> evaluation of the process, and LO7 (ethics, sustainability, social
> responsibility, global context) which Lesson 4 requires explicitly.

---

## 7.1 Answers to the research questions

**RQ1: How effective are machine learning models in detecting deepfake media?**

Within the distribution they are trained on, very effective, and even a
logistic regression on 32×32 greyscale pixels is well above chance. Outside it,
ineffective: all four models fell to approximately chance on an unseen
manipulation method (§6.3). The answer therefore depends entirely on whether
the question means "on the test set" or "in deployment", and for the best model
those two readings differ by 44.2 accuracy points.

**RQ2: Which features or patterns are most useful in distinguishing real and
fake media?**

This question is the one the evidence here answers least directly, and the
answer has to be stated carefully. What the experiments establish is that the detector is *sensitive* to
high-frequency content: the resolution ablation, the gap between destructive
and rescaling perturbations (§6.5), and sharpness being the strongest single
low-level discriminator in the exploratory analysis all point the same way.
Sensitivity is not the same as knowing which features were learned, and nothing
here opens the network up to look.

The spectral comparison in §6.3.2 sharpens that distinction. In the training
dataset the averaged power spectra of the two classes almost coincide, so
whatever the model exploits to reach 94.3% is not a global difference in
high-frequency energy; it must be finer than a radial average can represent.
In the unseen dataset a clear high-frequency difference does exist and runs in
the opposite direction. The defensible claim is therefore about transfer rather
than about representation: the frequency properties that distinguish the
classes are not stable across the two datasets, which is consistent with
detectors keyed to generator-specific artefacts rather than to semantic
indicators of manipulation. Establishing what the network actually uses would
need an attribution method, which §8.5 lists as further work.

**RQ3: How does the performance of the proposed model compare to baseline
approaches?**

In-domain, substantially better: roughly sixteen accuracy points separate the
two CNNs from the stronger classical baseline. Cross-dataset, no better in any
practical sense. The CNNs' AUCs are statistically above chance, but their best
achievable accuracies are not usable detectors. At matched capacity,
EfficientNet-B0 gained 0.3 accuracy points in-domain for 85% more training
time, so architectural sophistication bought very little here.

**RQ4: How robust is the detection model against manipulated or adversarial
inputs?**

Moderately robust to mild degradation, fragile to severe degradation, and
almost completely defenceless against white-box adversarial attack. It holds up
through the compression a social platform would apply but roughly halves on a
thumbnail, and under FGSM at an imperceptible perturbation the ranking inverts
(§6.5, §6.6).

**RQ5: What are the key limitations of current deepfake detection techniques?**

The limitation documented most clearly here is that in-domain accuracy and
cross-domain reliability appear to be in tension: across four models the
generalisation gap widened monotonically as in-domain accuracy improved. If
detectors succeed by exploiting generator-specific artefacts, a better detector
is partly one exploiting them more thoroughly, which offers no protection
against a generator it has not seen.

## 7.2 Strengths of the artefact

The experimental design supports the conclusions drawn from it. Splits are
seeded and verified disjoint by assertion, manifests are committed, all
hyperparameters live in one configuration file, and every result in Chapter 6
can be regenerated from a clean clone. Dataset independence was verified rather
than assumed before any compute was committed (§5.2.1).

The threshold diagnostic separating calibration failure from representation
failure adds something beyond reporting accuracy: two models can both score 49%
for entirely different reasons, and which one applies changes what conclusion
is available. The robustness prediction also went into the module docstring
before the suite ran (Figure 5.5), so it was a test rather than a post-hoc
rationalisation.

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
used to train the generators these models detect. Generation is far more
compute-intensive than detection and also far better funded, so the
environmental cost of the arms race falls disproportionately on the generation
side.

**Global context.** Detection performance is bounded by training data, and face
datasets have well-documented demographic imbalances. Neither dataset here
provides demographic metadata, so performance disparities across groups could
not be tested. This is a genuine gap: a detector that works less well for
under-represented populations would distribute the harms of deepfakes unevenly,
and those least protected would be the people already most vulnerable to
image-based abuse.

---

*Draft word count: approximately 1,700.*
