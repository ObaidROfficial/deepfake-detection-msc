# Chapter 8: Conclusion and Further Work

> **DRAFT v1, 11 August 2026.** Target 900 words.

---

## 8.1 Objectives revisited

**Objective 1, structured literature review.** Achieved. Chapter 2 covers
generation techniques, detection approaches, benchmark datasets and the
generalisation and robustness literature, drawing on sixteen sources verified
against the original papers. Frank et al.'s (2020) finding that GAN frequency
artefacts originate in generator upsampling proved central, supplying a
mechanism that explained results first observed empirically. Yan et al. (2023)
shaped the methodology by establishing that accuracy figures are not reliably
comparable across papers, which is why every model here is evaluated under one
fixed protocol.

**Objective 2, prepare balanced, reproducible datasets.** Achieved and
exceeded. Two datasets using genuinely different manipulation methods were
prepared with seeded, verified-disjoint splits and committed manifests, and
their independence was verified empirically before training, not in the
original plan, but the cross-dataset argument would have failed had they shared
a real-image source.

**Objective 3, implement a CNN with baselines.** Achieved: MobileNetV2 with
two-stage transfer learning, plus two classical baselines chosen to serve
different diagnostic purposes rather than as a formality.

**Objective 4, evaluate and compare.** Achieved, with a caveat: the comparison
produced a conditional answer, the CNN being substantially better in-domain and
no better outside it.

**Objective 5, test robustness.** Achieved. Twenty-two degradation settings
plus adversarial attack in two label regimes.

**Objective 6, document the work.** This report. All six were met, with 2 and
5 going beyond what was specified.

## 8.2 Key findings

**Improvements in in-domain accuracy did not transfer.** Across four models
in-domain accuracy rose 21 points (73.6% → 94.6%) while cross-dataset accuracy
never left the neighbourhood of chance, so the generalisation gap widened
monotonically: 24.7, 30.7, 43.2, 44.2 points. Better representation learning
was absorbed entirely by the source domain.

**The failure is not architecture-specific.** MobileNetV2 and EfficientNet-B0
are structurally different networks matched on trainable capacity to within
6.4%. Both reach approximately 94.5% in-domain and both land within a point of
chance on unseen manipulations, so a single unlucky architecture choice does
not explain it.

**The failure is one of representation, not calibration.** Across every
threshold in both polarities, the best achievable accuracy on the unseen
dataset was 0.5453. No operating point recovers usable performance, so
recalibration cannot fix this.

**The discriminative signal is high-frequency, and this explains both
failures.** Consistent with Frank et al. (2020), raising input resolution from
128 to 224 pixels with all else constant was worth 10.1 accuracy points, and
perturbations removing or masking high-frequency content cost up to 38.5 points
against at most 3.8 for intensity rescaling of comparable perceptual magnitude.
One mechanism accounts for the resolution effect, the robustness profile and
the transfer failure, and generated predictions recorded before the experiments
that tested them.

**The detector fails in the dangerous direction.** When high-frequency evidence
is unavailable, whether through domain shift or heavy compression, the model does not
express uncertainty. It classifies almost everything as real, predicting "fake"
for only 7.2% of a balanced unseen set.

**White-box adversarial robustness is effectively absent.** Accuracy falls from
94.3% to 5.2% under an imperceptible FGSM perturbation, ranking inverted.

## 8.3 Contribution

The contribution is not a new architecture but a structured account of *how* a
conventional detector fails outside its training distribution, under documented
resource constraints.

Three elements distinguish it from simply reporting a cross-dataset accuracy
drop. The threshold diagnostic separates calibration failure from
representation failure, which are usually conflated. Generalisation and
robustness are examined together and shown to share a mechanism, where the
literature typically treats them separately. And the mechanism was used
predictively rather than only descriptively, in that the robustness prediction was
recorded in the code before the suite ran.

The project also demonstrates that meaningful deepfake detection research is
achievable on consumer CPU hardware. Total compute across all experiments was
approximately eight hours.

## 8.4 Limitations

The main limitations, discussed fully in Chapter 7, are: a single cross-dataset
pair rather than a matrix; both datasets GAN-era with no diffusion coverage;
20,000 of 100,000 available training images; in-domain accuracy roughly three
points below published figures; robustness measured on one architecture only;
adversarial testing limited to white-box FGSM; and imperfect controls in the
robustness suite, since brightness and contrast are multiplicative and do alter
gradient magnitudes.

## 8.5 Further work

**Extend the cross-dataset matrix.** The clearest next step: three or four
datasets spanning GAN synthesis, face swapping, face reenactment and diffusion
generation, evaluated pairwise, would establish whether the widening-gap
pattern is general or specific to the pair tested here.

**Test whether the high-frequency dependence can be broken deliberately.**
Training with aggressive JPEG and downscaling augmentation should force the
model toward more transferable features, at some cost to in-domain accuracy.
This is the most promising of these proposals because it is not speculative:
Wang et al. (2020) attribute their cross-generator generalisation largely to
pre-processing and augmentation rather than to the classifier, so the lever is
one the literature has already shown to work. Whether the trade is worthwhile
at this scale is directly testable and the pipeline already supports it.

**Frequency-domain input representations.** Training on DCT or wavelet
representations rather than pixels is a natural extension given where these
results point.

**Adversarial training and stronger attacks.** FGSM is the weakest standard
attack; iterative attacks such as PGD and realistic black-box transfer attacks
would give a fuller picture.

**Demographic auditing.** Neither dataset provides demographic metadata, so
performance disparities across groups could not be tested. Given documented
imbalances in face datasets, and that image-based abuse falls
disproportionately on already-vulnerable groups, this is important rather than
optional.

**Calibrated confidence output.** The most immediately useful practical change.
A detector reporting calibrated uncertainty, and flagging inputs outside its
validated conditions, would be far safer to deploy than one silently returning
"real" when it has no evidence either way.

**Engineering directions for the artefact.** Separately from the research
questions above, four changes would widen who can use the detector. Accepting
video and aggregating across frames is the largest, since §1.7 scopes this work
to stills and manipulated media mostly circulates as video. Mobile applications
would put the tool where images are received rather than where they are
analysed, though the model would need converting to a mobile runtime first.
Both are helped by shrinking the distribution, currently dominated by a PyTorch
install of several hundred megabytes rather than by the 8.7MB model, and by
fetching weights on demand as the installer already does. None of this would
change a number in Chapter 6, only how many people ever see them.

## 8.6 Entrepreneurial and practical potential

The artefact is a research prototype, and the honest assessment is that its
cross-dataset and adversarial results argue against deploying this class of
detector as a standalone authenticity verdict.

The potential lies in the evaluation methodology rather than the model.
Organisations procuring detection tools are generally shown in-domain benchmark
figures. A standardised harness reporting cross-method transfer, degradation
curves and adversarial robustness alongside headline accuracy would let buyers
see the operating envelope rather than the best case. That a 94.3% detector
delivers roughly 56% on a thumbnail is exactly the kind of information that
should inform procurement and appears on no specification sheet.

More broadly, the results support the view that provenance-based approaches,
meaning cryptographic signing at capture, are likely to prove more durable than
artefact-based detection, which is structurally always one generator behind.

---

*Draft word count: approximately 900.*
