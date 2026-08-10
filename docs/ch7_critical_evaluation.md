# Chapter 7: Critical Evaluation

> **DRAFT v1 — 11 August 2026.** Target 1,700 words.
> Covers: answers to the research questions, artefact strengths and weaknesses,
> evaluation of the process, and LO7 (ethics, sustainability, social
> responsibility, global context) which Lesson 4 requires explicitly.

---

## 7.1 Answers to the research questions

**RQ1: How effective are machine learning models in detecting deepfake media?**

Within the distribution they are trained on, very effective. MobileNetV2
reached 94.3% accuracy at an AUC of 0.987, and even logistic regression on
32×32 greyscale pixels reached 73.6%. Outside that distribution, ineffective —
all three models fell to approximately chance on an unseen manipulation method.
The answer therefore depends entirely on whether the question means "on the
test set" or "in deployment", and the difference between those two readings is
43 accuracy points.

**RQ2: Which features or patterns are most useful in distinguishing real and
fake media?**

High-frequency detail. Three separate pieces of evidence converge on this.
Raising input resolution from 128 to 224 pixels, with everything else held
constant, was worth ten accuracy points. Perturbations that remove or mask
high-frequency content cost up to 38.5 points, while intensity rescaling of
the same perceptual magnitude cost at most 3.8. And exploratory analysis found
that sharpness was the strongest single low-level discriminator within Dataset
A, with an AUC of 0.611 against 0.547 and 0.553 for brightness and contrast.

This aligns with Frank et al. (2020), who traced GAN frequency artefacts to the
upsampling operations present in generator architectures. The useful features
appear to be generator fingerprints rather than semantic indicators of
manipulation.

**RQ3: How does the performance of the proposed model compare to baseline
approaches?**

In-domain, substantially better: 94.3% for MobileNetV2 and 94.6% for
EfficientNet-B0, against 78.6% for HOG+LBP and 73.6% for logistic regression.
Cross-dataset, no better in any practical sense. All four sit within three
points of chance, and no threshold in either polarity recovers usable accuracy
for any of them. The CNNs' cross-dataset AUCs of 0.5593 and 0.5285 are
statistically above chance and better than the pixel baseline, but best
achievable accuracies of 54.5% and 52.2% do not constitute usable detectors.

The comparison between the two CNNs is also informative. Configured to matched
trainable capacity — 1,208,642 against 1,131,954 parameters, within 6.4% —
EfficientNet-B0 gained 0.3 accuracy points in-domain for 85% more training time
(4.44 hours against 2.40), and achieved a worse best validation loss. On this
task, at this scale, architectural sophistication bought very little.

**RQ4: How robust is the detection model against manipulated or adversarial
inputs?**

Moderately robust to mild degradation, fragile to severe degradation, and
almost completely defenceless against white-box adversarial attack. It holds up
at JPEG quality 50 and above (≥ 0.895) but falls to 0.558 when downscaled to
thumbnail size. Under FGSM at an imperceptible ε of 0.005, accuracy collapses
from 0.943 to 0.052 and the ranking inverts.

**RQ5: What are the key limitations of current deepfake detection techniques?**

The limitation this project documents most clearly is that in-domain accuracy
and cross-domain reliability are not merely different — they appear to be in
tension. Across three model families, the generalisation gap widened
monotonically as in-domain accuracy improved: 24.7, then 30.7, then 43.2
points. If detectors succeed by exploiting generator-specific artefacts, a
better detector is partly one that exploits them more thoroughly, and that
offers no protection against a generator it has not seen.

## 7.2 Strengths of the artefact

The experimental design supports the conclusions drawn from it. Splits are
seeded and verified disjoint by assertion, manifests are committed so the exact
subsets can be reconstructed, and all hyperparameters live in a single
configuration file rather than being scattered through code. Every result in
Chapter 6 can be regenerated from a clean clone.

Dataset independence was verified rather than assumed. Before committing
compute to training, I checked whether Dataset B shared a real-image source
with Dataset A, since that would have invalidated the cross-dataset design.
Perceptual hashing found zero overlap across 1,400 sampled images, and Dataset
A's mean image retains 1.6 times the edge structure of Dataset B's, consistent
with FFHQ's landmark alignment and its absence in OpenForensics.

The threshold diagnostic distinguishing calibration failure from representation
failure adds something beyond reporting accuracy. Two models can both score
49% for entirely different reasons, and knowing which applies changes what
conclusion is available.

Several predictions were recorded before the experiments that tested them. The
robustness prediction was written into the module docstring before the suite
ran. This makes those results genuine tests rather than post-hoc rationalisation.

## 7.3 Weaknesses and limitations

**Scale.** Training used 20,000 of the 100,000 available images. The final
model was still improving when early stopping halted it on validation loss, and
in-domain accuracy sits roughly three points below published figures for this
dataset. More data and a fully unfrozen backbone would likely close that gap.

**Two datasets, one comparison.** The cross-dataset finding rests on a single
pair. A three- or four-way matrix would establish whether the effect is general
or specific to this pair of manipulation types. As it stands the finding is
suggestive rather than conclusive.

**Both datasets are GAN-era, and this limitation is sharper than it first
appears.** GANs construct images by repeatedly upsampling, and Frank et al.
(2020) identified that upsampling as the source of the artefacts detectors
exploit. Diffusion models generate by iterative denoising instead, so that
artefact family is largely absent. The better this model becomes at finding GAN
upsampling traces, the less reason there is to expect it to find anything in a
diffusion image.

Testing during development bore this out: a visibly synthetic
diffusion-generated portrait was classified **REAL with reported confidence of
100%**, even after the face was correctly detected and cropped (§5.7). This is
consistent with the cross-dataset result rather than separate from it — Dataset
B was a different manipulation method within the same generative family, so a
larger failure across families is what the account predicts.

The more troubling aspect is the confidence. The model reported certainty on an
input it had no basis for judging. A softmax output from a cross-entropy-trained
network is not a probability of being correct, and here it was actively
misleading. A user without this report would have received an unambiguous,
confident, wrong answer — which is precisely the failure mode Chapter 8 argues
is worse than no detector at all.

**Deviation from the proposal.** The project specified FaceForensics++ and the
DFDC corpus. Neither was usable: FF++ requires a signed licence agreement with
turnaround measured in weeks, and DFDC is approximately 470GB of video that
cannot be decoded on a CPU-only machine within the timeframe. The substitutes
are defensible but they cost comparability — most published results use FF++,
so the numbers here cannot be placed directly alongside them.

**Imperfect controls.** Brightness and contrast were used as the
detail-preserving comparison in the robustness suite, but both are
multiplicative and do alter gradient magnitudes; measured on a sample image,
contrast −0.4 reduced high-frequency energy to 0.60 of the original. The
argument rests on rescaling being near-invertible and structure-preserving
rather than on it leaving the frequency content untouched, which is a weaker
claim than a genuinely clean control would support.

**Robustness tested on one architecture.** Both CNNs were trained to
completion, so the *generalisation* finding is no longer single-architecture.
The robustness suite, however, was run only on MobileNetV2. Whether the
degradation profile and adversarial fragility are architecture-specific
remains untested, though there is little reason to expect a large difference
given both models collapse identically on transfer.

**Adversarial scope.** Only FGSM, only white-box. Stronger iterative attacks
and realistic black-box transfer attacks were not attempted.

## 7.4 Evaluation of the process

The Agile approach suited this project. Several conclusions changed as evidence
accumulated, and a Waterfall plan fixed at proposal stage would have handled
that badly.

Three specific instances are worth recording because they shaped the outcome.
The initial CPU benchmark overestimated training time by roughly a factor of
seven, because it measured full backpropagation while the actual training keeps
most of the backbone frozen. Being wrong conservatively was the right direction
to be wrong when planning to a deadline, but the methodology quotes the measured
figure rather than the projection.

The first two full training runs were stopped by the epoch count rather than by
convergence, and I reported a 78.7% result before noticing that validation
accuracy was still climbing and sitting above training accuracy. Recognising
that this indicated undertraining rather than a ceiling was what led to the
resolution experiment, which produced the project's clearest ablation.

The first FGSM implementation computed gradients against the ground-truth
label, producing accuracy of exactly zero. Investigating why a number was
*too* clean identified the label-leaking effect and produced a defensible
figure.

The pattern in all three is that the useful step was treating an anomalous
result as a question rather than an answer.

**What I would do differently.** I would fix input resolution by experiment at
the start rather than defaulting to 128 pixels for speed; that single choice
cost ten accuracy points and two full training runs. I would set epoch budgets
generously from the beginning and let early stopping decide. And I would start
the literature review in week one rather than treating it as something to fit
around the implementation, because it turned out to explain results I had
already observed and would have let me predict them instead.

## 7.5 Ethical, social and professional considerations

**Ethics approval and data.** This project used only secondary data from
publicly available datasets and involved no human participants, so participant
consent, debriefing and company sign-off procedures under the Arden University
Ethics Policy (QA 39) did not apply. Ethical approval was obtained before work
began. No personal data as defined by UK GDPR was collected or processed. The
datasets contain images of identifiable individuals, but they are distributed
for research purposes under their own licences, were not redistributed as part
of this work, and download instructions rather than image files are included in
the repository.

**Dual-use risk.** Detection research is inherently dual-use. The robustness
results in particular identify precise conditions under which this class of
detector fails, and that information is of interest both to defenders and to
those seeking to evade detection. I judged publication appropriate on the basis
that these weaknesses are already documented in the peer-reviewed literature,
that the attack used is a standard published baseline rather than a novel
method, and that defenders benefit more from knowing the operating limits of
their tools than attackers gain from confirmation of a known vulnerability.
The BCS Code of Conduct's requirement to have due regard for public health,
privacy and wellbeing supports disclosure of limitations rather than their
concealment.

**Social responsibility and the deployment question.** The most
socially significant finding is the failure direction. When the evidence it
relies on is unavailable, this detector does not report uncertainty — it
declares images authentic. A system deployed on compressed social media content
would systematically under-report manipulation while appearing to function
normally. Given deepfakes are used for non-consensual imagery, fraud and
political disinformation, a detector that silently fails open could do more
harm than no detector at all, because it confers unwarranted confidence. Any
deployment should report calibrated confidence and flag inputs outside the
conditions it was validated on.

**Sustainability.** Training was performed on a CPU-only consumer laptop.
Total compute across all experiments was approximately eight hours, which is
negligible against the resources used to train the generators these models
detect. There is a broader asymmetry worth noting: generation is far more
compute-intensive than detection, but generation is also far better funded, and
the environmental cost of the arms race falls disproportionately on the
generation side.

**Global context.** Detection performance is bounded by training data, and
face datasets have well-documented demographic imbalances. Neither dataset used
here provides demographic metadata, so I could not test for performance
disparities across groups. This is a genuine gap: a detector that works less
well for under-represented populations would distribute the harms of deepfakes
unevenly, and the people least protected would be those already most
vulnerable to image-based abuse. Auditing detection performance across
demographic groups is important further work that this project was not able to
undertake.

---

*Draft word count: approximately 1,700.*
