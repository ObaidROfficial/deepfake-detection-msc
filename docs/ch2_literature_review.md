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
improved rapidly, and by the time of StyleGAN (Karras, Laine and Aila, 2019)
GANs could synthesise entire human faces at 1024×1024 that most people cannot
identify as artificial. These are *fully synthetic* images — no real person is
depicted at all. StyleGAN is directly relevant here because it produced the
fake half of the training dataset used in this project.

An architectural detail of StyleGAN matters for the results later. Like all
current GAN architectures it builds an image by progressively **upsampling**
from a low-resolution representation, and as section 2.4 sets out, that
upsampling is the origin of the frequency artefacts detectors learn to exploit.

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

### Handcrafted features

Early detection work used handcrafted features, looking for inconsistencies a
generator was likely to get wrong — irregular blinking, inconsistent head pose,
mismatched lighting, or visible blending boundaries around a swapped face
region. These methods have the advantage of being interpretable: the reason for
a decision can be stated. Their weakness is that each targets a specific flaw,
so they degrade quickly once generators improve enough to address it. The
history of the field is largely a sequence of such cues being identified and
then closed.

The classical computer vision descriptors used as a baseline in this project —
Histogram of Oriented Gradients and Local Binary Patterns — belong to this
tradition, though they are applied here as general texture and gradient
descriptors rather than as detectors of any specific artefact.

### Convolutional neural networks

The field then moved almost entirely to convolutional neural networks. Rather
than specifying what to look for, a CNN is trained end to end on labelled real
and fake images and learns its own features. Architectures developed for
general image classification are typically borrowed, initialised from ImageNet
weights and fine-tuned on the detection task.

Rössler et al. (2019) established XceptionNet as a strong baseline for this
task in the FaceForensics++ paper, and it remains a common reference point.
Reported accuracies on standard benchmarks are consistently high, frequently
above 95%.

Two architectures are used in this project, both chosen for reasons discussed
in Chapter 4.

**MobileNetV2** (Sandler et al., 2018, CVPR) is built around an inverted
residual structure in which the input and output of each block are narrow
bottleneck layers and the intermediate layer is an expanded representation
filtered by lightweight depthwise convolutions. The authors also argue against
placing non-linearities in the narrow layers, on the grounds that this destroys
representational capacity. The practical consequence is a network with far
fewer parameters and operations than a conventional CNN of comparable accuracy,
which is what makes it viable on CPU.

**EfficientNet-B0** (Tan and Le, 2019, ICML) introduces compound scaling:
rather than increasing depth, width or input resolution independently, all
three are scaled together according to a fixed ratio. The building block is a
mobile inverted bottleneck similar to MobileNetV2's, but the network is larger.
Its inclusion here allows an architecture comparison at matched trainable
capacity.

It is worth noting that neither architecture was designed for forensic
analysis. Both are general-purpose classifiers, and the assumption underlying
their use — that features learned for object recognition transfer usefully to
detecting generator artefacts — is rarely examined explicitly in this
literature. The results in Chapter 6 bear on that assumption.

### Transformers

More recently transformer architectures have been applied to the problem, and
there is some evidence they generalise across datasets better than CNNs
`[VERIFY — a Springer paper reports ~11.3% degradation for transformers against
>15% for CNNs; the protocol behind those figures must be checked before they
are quoted, since "degradation" is defined inconsistently across this
literature]`. Transformers were not used here because of the CPU-only hardware
constraint discussed in Chapter 3; attention over image patches is
substantially more expensive than depthwise convolution at the resolutions
required.

### Frequency-domain approaches

A third strand works in the frequency domain rather than on pixels directly. It
is treated separately in section 2.4 because it supplies the mechanism that
explains several of this project's results.

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

**FaceForensics++** (Rössler et al., 2019) is the most widely used benchmark. It
was presented at ICCV 2019 and provides video sequences manipulated by four
distinct techniques — DeepFakes, Face2Face, FaceSwap and NeuralTextures — at
several compression levels. That combination is what makes it valuable for the
questions this project asks: it allows both cross-method and cross-compression
evaluation within one dataset. The authors report that domain-specific
knowledge improves detection accuracy even under strong compression, and that
automated detectors clearly outperform human observers. Access requires a
signed agreement with the hosting institution.

It is worth being explicit that this project would have benefited from
FaceForensics++ precisely because it varies manipulation *method* and
compression *level* independently, which is exactly the two-factor design the
results in Chapters 6 and 7 approximate using two separate datasets and a
synthetic degradation suite. Chapter 3 explains why it could not be used.

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

The **Flickr-Faces-HQ (FFHQ)** dataset (Karras et al., 2019) is not a deepfake
dataset but supplies the real images for the training set used in this project,
and is the dataset StyleGAN was trained on. It contains 70,000 images at
1024×1024 and was introduced alongside the StyleGAN architecture at CVPR 2019,
offering greater variation in accessories, age, ethnicity and background than
the CelebA-HQ dataset it largely displaced.

Two properties of FFHQ matter here. Its images are **landmark-aligned**, which
this project exploited when verifying that the two datasets do not share a
real-image source (Chapter 5). And its stated improvement in demographic
variation over CelebA-HQ is relative rather than absolute — it provides no
demographic labels, which is why the demographic audit discussed in Chapter 7
could not be performed.

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

Recent work has attempted to quantify this systematically. Yan et al. (2023)
introduced **DeepfakeBench**, which they describe as the first comprehensive
benchmark for the field, integrating 15 detection methods and 9 datasets behind
a unified data-management system with standardised evaluation protocols. Their
motivation is worth quoting in substance: they argue that the absence of a
standardised benchmark leads to *"unfair performance comparisons and
potentially misleading results"*, because data-processing pipelines, experimental
settings and evaluation metrics all differ between papers.

This is a significant observation for the present project, and it cuts two
ways. It means published accuracy figures cannot be compared across papers with
confidence, which is part of why this dissertation evaluates all of its own
models under one fixed protocol rather than quoting numbers from elsewhere. It
also means the ~97% figures commonly cited for the dataset used here should be
treated as indicative rather than as a precise target.

A 2025 systematic review focusing specifically on model generalisation
`[VERIFY — ScienceDirect S2543925125000075]` and a 2026 study evaluating
fourteen benchmark datasets `[VERIFY — arXiv:2508.06248]` both indicate this
remains an open problem rather than a solved one.

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

An analysis of recent advances in deepfake detection presented at IEEE Security
and Privacy 2024 examines eight state-of-the-art detectors and argues they are
"far from being ready for deployment". Two developments drive that conclusion.
Lightweight methods for customising large generative models let an attacker
produce many bespoke generators, and the authors show existing defences fail to
generalise well to such user-customised models. Separately, vision foundation
models can be misused to craft adversarial deepfakes; they demonstrate an attack
that evades defences **without adding any adversarial noise**, instead
manipulating image content semantically.

The first point reinforces the case for cross-dataset evaluation: if the number
of distinct generators in circulation is growing rapidly, in-domain accuracy
becomes an ever weaker proxy for deployed performance.

The second sets a limit on this project's own adversarial testing that should
be stated plainly. FGSM, used in Chapter 6, perturbs pixels within a small norm
budget. An attack operating through semantic manipulation rather than additive
noise would not be captured by any measure used here, and would not be
addressed by the adversarial training suggested as further work in Chapter 8.
The adversarial results in this dissertation are therefore a lower bound on
vulnerability rather than a full characterisation of it.

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
| Frank, J., Eisenhofer, T., Schönherr, L., Fischer, A., Kolossa, D. and Holz, T. (2020) *Leveraging Frequency Analysis for Deep Fake Image Recognition*. ICML 2020. arXiv:2003.08685 | **Verified** — upsampling claim confirmed from the paper |
| Le, T.-N., Nguyen, H.H., Yamagishi, J. and Echizen, I. (2021) *OpenForensics: Large-Scale Challenging Dataset For Multi-Face Forgery Detection And Segmentation In-The-Wild*. ICCV 2021. arXiv:2107.14480 | **Verified** — in-the-wild multi-face framing confirmed |
| Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J. and Nießner, M. (2019) *FaceForensics++: Learning to Detect Manipulated Facial Images*. ICCV 2019, Seoul. arXiv:1901.08971 | **Verified** — authors, venue, four manipulation methods, compression levels |
| Karras, T., Laine, S. and Aila, T. (2019) *A Style-Based Generator Architecture for Generative Adversarial Networks*. CVPR 2019, pp. 4401–4410 | **Verified** — authors, venue, pages; FFHQ 70k at 1024×1024 |
| Yan, Z., Zhang, Y., Yuan, X., Lyu, S. and Wu, B. (2023) *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection*. arXiv:2307.01426 | **Verified** — authors, 15 methods / 9 datasets, "unfair performance comparisons" claim |
| Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.-C. (2018) *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR 2018, pp. 4510–4520 | **Verified** — authors, venue, pages; inverted residual / linear bottleneck claims |
| Tan, M. and Le, Q.V. (2019) *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML 2019, pp. 6105–6114 | **Verified** — authors, venue, pages; compound scaling |
| *An Analysis of Recent Advances in Deepfake Image Detection in an Evolving Threat Landscape*. IEEE S&P 2024. arXiv:2404.16212 | **Verified** — 8 detectors, user-customised generators, no-noise semantic attack. **Author list still to be confirmed** |

All others are currently based on search results or general knowledge and
**must not go into the submitted reference list until the paper has been
opened and checked**. Lesson 5 warns specifically about reference lists that
appear assembled rather than read, and the Documentation criterion is worth 10%.

### Still outstanding

| Source | Needed for |
|---|---|
| Source | Needed for | Priority |
|---|---|---|
| Goodfellow et al. (2014), *Generative Adversarial Nets*, NeurIPS | §2.2 | High — foundational, cited in the proposal |
| Dolhansky et al. (2020), DFDC dataset | §2.5 + Ch3 deviation argument | High — the deviation rests on it |
| Li et al. (2020), Celeb-DF | §2.5 | Medium |
| arXiv:2508.06248, *Deepfake Detection that Generalizes Across Benchmarks* | §2.6 | Medium |
| ScienceDirect S2543925125000075, systematic review on generalisation | §2.6 | Medium |
| Springer 10.1007/s00138-026-01809-w, transformer vs CNN | §2.3 | **Care needed — verify the protocol before quoting 11.3% / 15%** |
| WACV 2021, *Adversarial Deepfakes* | §2.7 | Medium |
| ACM DTRAP 10.1145/3464307, robust attacks on detectors | §2.7 | Medium — ACM venue, good for the professional-body requirement |
| Yu et al., GAN fingerprints; GANprintR | §2.4 | Low — need full citations before use |
| arXiv:2210.14571, diffusion model detection | §2.4 + Ch7 limitation | Medium |
| Xception / MobileNetV2 / EfficientNet original papers | §2.3, Ch4 justification | High — architectures are used, should be cited |

### Next actions

1. Obtain and read the `[VERIFY]` sources; correct details, add page/volume data
2. Expand §2.3 with named studies and their reported accuracies
3. Expand §2.5 with dataset sizes and composition
4. Add 15–20 further sources — currently well short of the 45–60 target
5. Import everything to Zotero, set Arden Harvard output, cross-check every in-text citation against the reference list
