# Chapter 2: Literature Review and Fact Finding

> **DRAFT v1 — 11 August 2026.** Target 2,600 words. Current count noted at the
> end. Citations marked `[VERIFY]` need checking against the actual paper
> before submission — see §2.8.

---

## 2.1 Introduction to the review

This chapter reviews the technical and academic background to the project. It
covers how deepfakes are produced, how detection has developed, which datasets
the field relies on, and the two problems that motivate this dissertation:
poor generalisation across manipulation methods, and fragility under everyday
image degradation.

The search was carried out using IEEE Xplore, the ACM Digital Library,
ScienceDirect and Google Scholar, with the arXiv preprint server used to reach
recent work not yet formally published. Search terms included "deepfake
detection", "GAN artefact detection", "cross-dataset generalisation" and
"adversarial attacks on deepfake detectors". Priority was given to
peer-reviewed papers published between 2019 and 2026, since the field moves
quickly and older work often describes generators that are no longer
representative.

## 2.2 How deepfakes are generated

The term deepfake covers several different techniques and it is worth
separating them, because as this project shows later, the technique used has a
large effect on what a detector actually learns.

Generative Adversarial Networks, introduced by Goodfellow et al. (2014)
`[VERIFY]`, train two networks against each other: a generator producing
images and a discriminator trying to tell them from real ones. The approach
improved rapidly and by the time of StyleGAN (Karras et al., 2019) `[VERIFY]`
GANs could synthesise entire human faces at high resolution that most people
cannot identify as artificial. These are *fully synthetic* images — no real
person is depicted at all.

A second family performs *face swapping*. Rather than generating a whole
image, an autoencoder or similar model replaces the face region in an existing
photograph or video with someone else's, leaving the background and body
untouched. This is what most people mean colloquially by a deepfake, and it is
the technique behind most of the misuse cases reported in the media.

More recently diffusion models have started to displace GANs for image
synthesis. This matters for the present project because both datasets used
here are GAN-era, which is a limitation acknowledged in Chapter 7.

The distinction between full synthesis and localised face swapping turns out
to be central. A detector trained on one is not obviously being trained for the
other, even though both are labelled "fake".

## 2.3 Detection approaches

Early detection work used handcrafted features. Researchers looked for
inconsistencies a generator was likely to get wrong — irregular blinking,
inconsistent head pose, mismatched lighting, or blending boundaries around a
swapped face region. These methods are interpretable, which is an advantage,
but they tend to break as generators improve and specifically address whatever
flaw was being exploited.

The field then moved almost entirely to convolutional neural networks. Rather
than specifying what to look for, a CNN is trained end to end on labelled real
and fake images and learns its own features. Architectures borrowed from
general image classification — XceptionNet, ResNet, EfficientNet and
MobileNet — have all been applied, usually initialised from ImageNet weights
and fine-tuned. Reported accuracies on standard benchmarks are very high,
frequently above 95%, and this is the approach adopted for the artefact in this
project.

More recently transformer architectures have been applied to the problem.
There is some evidence that transformers generalise across datasets better
than CNNs do `[VERIFY — a Springer paper reports ~11.3% degradation for
transformers against >15% for CNNs, but the protocol needs checking before
this figure is quoted]`. Transformers were not used here because of the
CPU-only hardware constraint discussed in Chapter 3.

A third strand works in the frequency domain rather than on pixels directly,
and it is worth its own section because it explains a result found later in
this project.

## 2.4 Frequency-domain artefacts

Frank et al. (2020), published at ICML, carried out what appears to be the
first systematic frequency-domain analysis of GAN-generated images. Their
finding is that GAN images contain severe and easily identified artefacts in
frequency space, and that these are consistent across different network
architectures, datasets and resolutions.

The important part of their contribution is the explanation rather than the
observation. They demonstrate that the artefacts are caused by the
**upsampling operations** present in all current GAN architectures — the steps
that progressively increase spatial resolution as an image is generated. Frank
et al. describe this as a structural and fundamental problem in the way GANs
produce images, not an incidental flaw of one model.

This matters a great deal for the present work. If the discriminative signal
between real and GAN-generated faces is concentrated in the high-frequency
band, then two consequences follow directly. First, anything that reduces
image resolution should reduce detection accuracy, because it discards exactly
the band carrying the evidence. Second, any processing that removes or masks
high-frequency detail — compression, blurring, downscaling — should damage a
detector far more than processing that leaves the fine structure intact.

Both of these were tested in this project and both were confirmed, which is
discussed in Chapters 6 and 7. Frank et al. therefore provides the theoretical
grounding for results I initially observed empirically without expecting them.

A related line of work looks at GAN "fingerprints" — generator-specific
patterns that allow an image to be attributed to the model that produced it.
The counterpart to this is that such fingerprints can be deliberately removed,
which limits how much a detector should rely on them `[VERIFY — GANprintR and
Yu et al. need full citations]`.

## 2.5 Benchmark datasets

Several datasets have become standard in this field.

**FaceForensics++** (Rössler et al., 2019) `[VERIFY]` is the most widely used
benchmark. It provides video sequences manipulated by several different methods
at several compression levels, which makes it valuable for studying exactly the
robustness questions this project is concerned with. Access requires a signed
agreement with the hosting institution.

**The Deepfake Detection Challenge dataset** (Dolhansky et al., 2020)
`[VERIFY]` was released by Facebook for a Kaggle competition and is very large
— several hundred gigabytes of video.

**Celeb-DF** improved on earlier datasets by producing visually higher-quality
forgeries, on the argument that earlier benchmarks contained obvious artefacts
that made the task unrealistically easy `[VERIFY]`.

**OpenForensics** (Le et al., 2021), published at ICCV, takes a different
approach again. Rather than single-face videos it targets multi-face forgery
detection and segmentation "in-the-wild", localising manipulated faces among
multiple people in unrestricted natural scenes. The authors argue this is
substantially harder than conventional deepfake recognition. This dataset
provides the cross-dataset evaluation set for the present project.

The **Flickr-Faces-HQ (FFHQ)** dataset (Karras et al., 2019) `[VERIFY]` is not
a deepfake dataset but supplies the real images for the training set used here,
and is the dataset StyleGAN was trained on.

An observation worth making is that these datasets differ not only in
generation method but in capture conditions, resolution, compression history
and alignment. A model trained on one and tested on another faces a shift on
several axes at once, and it is not straightforward to attribute a
generalisation failure to any single one of them.

## 2.6 The generalisation problem

This is the central issue the project addresses.

Detectors report very high accuracy when trained and tested on the same
dataset, but accuracy falls sharply when they are applied to manipulations
produced by a method they were not trained on. Since a real-world detector will
inevitably encounter generation techniques that did not exist when it was
trained, in-domain accuracy on its own is a poor indicator of practical value.

Recent work has attempted to quantify this systematically. DeepfakeBench
`[VERIFY — arXiv:2307.01426]` provides a unified benchmark allowing detectors
to be compared under standardised conditions, which addresses a real problem in
the literature: papers report results under differing protocols, making
cross-paper comparison unreliable. A 2025 systematic review focusing
specifically on model generalisation `[VERIFY — ScienceDirect
S2543925125000075]` and a 2026 study evaluating fourteen benchmark datasets
`[VERIFY — arXiv:2508.06248]` both indicate this remains an open problem rather
than a solved one.

The explanations offered generally reduce to the same idea: detectors learn
artefacts specific to the generator in their training data rather than any
general notion of manipulation. Frank et al.'s finding that upsampling
artefacts are structural to GANs supports this — if the signal is a
generator-specific fingerprint, a change of generator removes it.

What is less clear from the literature, and what this project examines
directly, is whether improving in-domain performance helps or hurts transfer.
If detectors succeed by exploiting generator-specific artefacts, then a better
detector may simply be one that exploits them more thoroughly, in which case
better in-domain accuracy would be associated with *worse* relative transfer
rather than better. Chapter 6 tests this.

## 2.7 Adversarial robustness and real-world degradation

Two distinct threats are usually treated separately in the literature and both
are relevant here.

The first is ordinary degradation. Images shared through social platforms are
recompressed, resized and re-encoded, often repeatedly. Since the forensic
evidence sits in the high-frequency band, and compression is designed
specifically to discard high-frequency information as perceptually
unimportant, there is a direct conflict between how images are distributed and
how detectors work.

An analysis of recent advances in deepfake detection presented at IEEE
Security and Privacy 2024 `[VERIFY — arXiv:2404.16212]` observes that most
detection papers conduct only basic robustness evaluation — typically blurring,
JPEG compression, downsampling and noise — and that even state-of-the-art
detectors show significant drops on compressed or resized inputs relative to
pristine ones. This both justifies including such a suite in the present
project and suggests going beyond it.

The second threat is deliberate adversarial attack. Work presented at WACV 2021
`[VERIFY]` demonstrated that deepfake detectors can be defeated by adversarially
crafted inputs, and further work published in ACM Digital Threats: Research and
Practice `[VERIFY — 10.1145/3464307]` examined robust attacks against detection
systems. The Fast Gradient Sign Method is the standard baseline attack in this
area and is the one used in Chapter 7.

An important qualification, which the present project observes in its own
results, is that adversarial evaluation is easy to get wrong in ways that
inflate apparent attack strength. This is discussed in Chapter 7.

## 2.8 Research gap and contribution

Three observations emerge from this review.

First, cross-dataset generalisation is a recognised weakness, but it is most
often reported as a single accuracy figure, without analysis of *how* the
failure occurs. Whether a detector's learned representation becomes merely
uninformative, or actively misleading, or simply mis-calibrated, are different
failures with different implications, and they are rarely distinguished.

Second, robustness evaluation and generalisation evaluation are usually
conducted separately. If both failures stem from the same cause — reliance on
high-frequency generator artefacts — then they should be studied together,
because the same mechanism predicts both.

Third, most published work uses substantial GPU resources. There is limited
discussion of what is achievable under constrained conditions, which is
relevant to practitioners without access to such hardware.

This project addresses the first two directly. It evaluates three model
families of increasing sophistication under a single protocol, measures
in-domain and cross-dataset performance for each, and applies a threshold
diagnostic that distinguishes calibration failure from representation failure
rather than reporting accuracy alone. It then applies a systematic
degradation suite and a white-box adversarial attack to the same model, and
tests whether one mechanism accounts for both the generalisation failure and
the robustness profile.

The contribution is therefore not a new detection architecture. It is a
structured examination of *how* and *why* a conventional detector fails outside
its training distribution, conducted under documented resource constraints.

---

## Draft notes

**Word count of main body (2.1–2.8): approximately 1,700.** Target is 2,600, so
roughly 900 words still to add. The thinnest sections are §2.3 (detection
approaches — needs specific named studies with reported figures) and §2.5
(datasets — needs the actual sizes and composition of each).

### Citations requiring verification before submission

Everything marked `[VERIFY]`. Only two sources have been read and confirmed
in full:

| Source | Status |
|---|---|
| Frank et al. (2020), *Leveraging Frequency Analysis for Deep Fake Image Recognition*, ICML 2020, arXiv:2003.08685 | **Verified** — upsampling claim confirmed from the paper |
| Le et al. (2021), *OpenForensics*, ICCV 2021, arXiv:2107.14480 | **Verified** — in-the-wild multi-face framing confirmed |

All others are currently based on search results or general knowledge and
**must not go into the submitted reference list until the paper has been
opened and checked**. Lesson 5 warns specifically about reference lists that
appear assembled rather than read, and the Documentation criterion is worth 10%.

### Next actions

1. Obtain and read the `[VERIFY]` sources; correct details, add page/volume data
2. Expand §2.3 with named studies and their reported accuracies
3. Expand §2.5 with dataset sizes and composition
4. Add 15–20 further sources — currently well short of the 45–60 target
5. Import everything to Zotero, set Arden Harvard output, cross-check every in-text citation against the reference list
