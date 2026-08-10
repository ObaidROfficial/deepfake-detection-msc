# Chapter 8: Conclusion and Further Work

> **DRAFT v1 — 11 August 2026.** Target 900 words.

---

## 8.1 Objectives revisited

**Objective 1 — structured literature review.** Achieved. Chapter 2 covers
generation techniques, the progression of detection approaches, benchmark
datasets, and the generalisation and robustness literature, drawing on eleven
sources verified against the original papers. Frank et al.'s (2020) finding
that GAN frequency artefacts originate in generator upsampling proved central,
supplying a mechanism that explained results initially observed empirically.
Yan et al. (2023) also shaped the methodology, by establishing that accuracy
figures are not reliably comparable across papers — which is why every model
here is evaluated under one fixed protocol rather than against quoted
benchmarks.

**Objective 2 — prepare balanced, reproducible datasets.** Achieved and
exceeded in one respect. Two datasets using genuinely different manipulation
methods were prepared with seeded, verified-disjoint splits and committed
manifests. Their independence was verified empirically before training, which
was not part of the original plan but proved necessary — the cross-dataset
argument would have failed had they shared a real-image source.

**Objective 3 — implement a CNN with baselines.** Achieved. MobileNetV2 with
two-stage transfer learning, plus two classical baselines chosen to serve
different diagnostic purposes rather than as a formality.

**Objective 4 — evaluate and compare.** Achieved, with a caveat worth stating.
The comparison produced a conditional answer rather than a clean one: the CNN
is substantially better in-domain and no better outside it.

**Objective 5 — test robustness.** Achieved. Twenty-two degradation settings
plus adversarial attack in two label regimes.

**Objective 6 — document the work.** This report.

All six objectives were met. Objectives 2 and 5 went beyond what was specified.

## 8.2 Key findings

**Improvements in in-domain accuracy did not transfer.** Across four models,
in-domain accuracy rose 21 points (73.6% → 94.6%) while cross-dataset accuracy
never left the neighbourhood of chance. The generalisation gap widened
monotonically: 24.7, 30.7, 43.2, 44.2 points. Better representation learning
was absorbed entirely by the source domain.

**The failure is not architecture-specific.** MobileNetV2 and EfficientNet-B0
are structurally different networks, trained at trainable capacity matched to
within 6.4%. Both reach approximately 94.5% in-domain and both land within a
point of chance on unseen manipulations. A single unlucky architecture choice
does not explain the result.

**The failure is one of representation, not calibration.** Testing every
threshold in both polarities, the best achievable accuracy on the unseen
dataset was 0.5453. No operating point recovers usable performance, so this
cannot be fixed by recalibration.

**The discriminative signal is high-frequency, and this explains both
failures.** Consistent with Frank et al. (2020), who trace GAN artefacts to
generator upsampling, raising input resolution from 128 to 224 pixels with all
else constant was worth 10.1 accuracy points. Perturbations that remove or mask
high-frequency content cost up to 38.5 points; intensity rescaling of
comparable perceptual magnitude cost at most 3.8. One mechanism accounts for
the resolution effect, the robustness profile and the transfer failure, and it
was used to generate predictions recorded before the experiments that tested
them.

**The detector fails in the dangerous direction.** When high-frequency
evidence is unavailable — whether through cross-domain shift or heavy
compression — the model does not express uncertainty. It classifies almost
everything as real, predicting "fake" for only 7.2% of a balanced unseen set.

**White-box adversarial robustness is effectively absent.** Accuracy falls from
94.3% to 5.2% under an imperceptible FGSM perturbation, with the ranking
inverted.

## 8.3 Contribution

The contribution is not a new architecture. It is a structured account of
*how* a conventional detector fails outside its training distribution,
conducted under documented resource constraints.

Three elements distinguish it from simply reporting a cross-dataset accuracy
drop. The threshold diagnostic separates calibration failure from
representation failure, which are usually conflated. Generalisation and
robustness are examined together and shown to share a mechanism, where the
literature typically treats them separately. And the mechanism was used
predictively rather than only descriptively — the robustness prediction was
recorded in the code before the suite ran.

The project also demonstrates that meaningful deepfake detection research is
achievable on consumer CPU hardware. Total compute across all experiments was
approximately eight hours.

## 8.4 Limitations

The main limitations, discussed fully in Chapter 7, are: a single
cross-dataset pair rather than a matrix; both datasets GAN-era with no
diffusion coverage; 20,000 of 100,000 available training images; in-domain
accuracy roughly three points below published figures; one architecture trained
to completion; adversarial testing limited to white-box FGSM; and imperfect
control conditions in the robustness suite, since brightness and contrast are
multiplicative and do alter gradient magnitudes.

## 8.5 Further work

**Extend the cross-dataset matrix.** The clearest next step. Three or four
datasets spanning GAN synthesis, face swapping, face reenactment and diffusion
generation, evaluated pairwise, would establish whether the widening-gap
pattern is general or specific to the pair tested here.

**Test whether the high-frequency dependence can be broken deliberately.**
Training with aggressive JPEG and downscaling augmentation should force the
model toward more transferable features, at some cost to in-domain accuracy.
Whether that trade is worthwhile is directly testable and the pipeline already
supports it.

**Frequency-domain input representations.** Training on DCT or wavelet
representations rather than pixels is a natural extension given where these
results point.

**Adversarial training and stronger attacks.** FGSM is the weakest standard
attack; iterative attacks such as PGD and realistic black-box transfer attacks
would give a fuller picture.

**Demographic auditing.** Neither dataset provides demographic metadata, so
performance disparities across groups could not be tested. Given that
face datasets have documented imbalances and that image-based abuse falls
disproportionately on already-vulnerable groups, this is important rather than
optional.

**Calibrated confidence output.** The most immediately useful practical
change. A detector that reports calibrated uncertainty, and flags inputs
outside its validated conditions, would be far safer to deploy than one that
silently returns "real" when it has no evidence either way.

## 8.6 Entrepreneurial and practical potential

The artefact is a research prototype, and the honest assessment is that its
cross-dataset and adversarial results argue against deploying this class of
detector as a standalone authenticity verdict.

The potential lies in the evaluation methodology rather than the model.
Organisations procuring detection tools are generally shown in-domain benchmark
figures. A standardised harness reporting cross-method transfer, degradation
curves and adversarial robustness alongside headline accuracy would let buyers
see the operating envelope rather than the best case. That a 94.3% detector
delivers roughly 56% on a thumbnail-resized image is exactly the kind of
information that should inform procurement and does not appear on any
specification sheet.

More broadly, the results support the view that provenance-based approaches —
cryptographic signing at capture — are likely to prove more durable than
artefact-based detection, which is structurally always one generator behind.

---

*Draft word count: approximately 900.*
