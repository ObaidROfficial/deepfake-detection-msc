# Chapter 2: Literature Review and Fact Finding

> **DRAFT v2 — 11 August 2026.** All sources cited in this chapter have been
> verified against the original papers. Claims that could not be sourced to a
> verified reference have been removed or generalised rather than attributed to
> a paper that was not read — see §2.8.

---

## 2.1 Introduction to the review

This chapter reviews the background to the project: how deepfakes are
produced, how detection has developed, which datasets the field relies on, and
the two problems motivating this dissertation — poor generalisation across
manipulation methods, and fragility under everyday image degradation.

Searches used IEEE Xplore, the ACM Digital Library, ScienceDirect and Google
Scholar, with arXiv for recent work not yet formally published. Terms included
"deepfake detection", "GAN artefact detection", "cross-dataset generalisation"
and "adversarial attacks on deepfake detectors". Priority went to peer-reviewed
work from 2019 onwards, since older papers often describe generators no longer
representative of the threat.

## 2.2 How deepfakes are generated

The term deepfake covers several different techniques and it is worth
separating them, because as this project shows later, the technique used has a
large effect on what a detector actually learns.

Generative Adversarial Networks, introduced by Goodfellow et al. (2014),
train two networks against each other: a generator producing
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
mismatched lighting, or blending boundaries around a swapped face. These are
interpretable, but each targets a specific flaw and degrades once generators
address it. The history of the field is largely a sequence of such cues being
identified and then closed.

The classical descriptors used as a baseline here — Histogram of Oriented
Gradients and Local Binary Patterns — belong to this tradition, though they are
applied as general texture and gradient descriptors rather than as detectors of
any specific artefact.

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
there is some evidence they generalise across datasets better than CNNs do.
Specific comparative figures are avoided here: "degradation" is defined
inconsistently across this literature, and quoting a percentage without
verifying the protocol behind it would be unsafe. The qualitative direction of
the finding is nonetheless worth noting, since it suggests the generalisation
failure documented in Chapter 6 may be partly architectural rather than wholly
intrinsic to the task.

Transformers were not used here because of the CPU-only hardware constraint
discussed in Chapter 3; attention over image patches is substantially more
expensive than depthwise convolution at the resolutions required.

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

A related line of work treats these artefacts as generator "fingerprints" —
patterns specific enough to attribute an image to the model that produced it.
The corollary is uncomfortable for detection: a signal distinctive enough to
identify a generator is, by construction, a signal that changes when the
generator does. This project's cross-dataset results in Chapter 6 are
consistent with that reading.

## 2.5 Benchmark datasets

Several datasets have become standard in this field.

**FaceForensics++** (Rössler et al., 2019, ICCV) is the most widely used
benchmark. It provides sequences manipulated by four distinct techniques —
DeepFakes, Face2Face, FaceSwap and NeuralTextures — at several compression
levels. Although distributed as video, it is commonly used frame-by-frame for
still-image detection, which is how it would have been used here. The authors
report that domain-specific knowledge improves detection accuracy even under
strong compression, and that automated detectors clearly outperform human
observers. Access requires a signed agreement with the hosting institution.

This project would have benefited from FaceForensics++ specifically because it
varies manipulation *method* and compression *level* independently — the
two-factor design that Chapters 6 and 7 approximate using two separate datasets
plus a synthetic degradation suite. Chapter 3 explains why it could not be used.

**The Deepfake Detection Challenge dataset** (Dolhansky et al., 2020) was
released by Facebook for a Kaggle competition and contains over 100,000 clips
from 3,426 paid actors, produced by several Deepfake, GAN-based and non-learned
methods. It is a video corpus, which places it outside the scope of this
project in two ways: the manipulations are face swaps within video sequences
rather than fully synthetic stills, and the full corpus runs to several hundred
gigabytes, which Chapter 3 discusses as the practical reason it could not be
used.

One detail is worth noting regardless of scope. All subjects consented to
having their likenesses modified, in contrast to datasets assembled from
scraped web imagery — a point returned to in the ethical discussion in
Chapter 3.

**Celeb-DF** (Li et al., 2020) contains 5,639 high-quality forgeries of
celebrities produced by an improved synthesis process. Its motivating argument
is directly relevant to this project: the authors contend that existing
datasets suffer from low visual quality and do not resemble the deepfakes
actually circulating online, making detection benchmarks unrealistically easy.
That is a criticism of benchmark validity rather than of any particular
detector, and it reinforces the case made throughout this chapter that
in-domain accuracy is a weak proxy for deployed performance.

**OpenForensics** (Le et al., 2021), published at ICCV, takes a different
approach again. Rather than single-face videos it targets multi-face forgery
detection and segmentation "in-the-wild", localising manipulated faces among
multiple people in unrestricted natural scenes. The authors argue this is
substantially harder than conventional deepfake recognition. This dataset
provides the cross-dataset evaluation set for the present project.

The **Flickr-Faces-HQ (FFHQ)** dataset (Karras, Laine and Aila, 2019) is not a deepfake
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

Recent surveys and multi-dataset evaluations continue to treat cross-dataset
generalisation as an open problem rather than a solved one, which is itself
informative: the difficulty has been recognised for several years without a
settled remedy emerging.

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

The second threat is deliberate adversarial attack. It is established that
deepfake detectors can be defeated by adversarially crafted inputs, and the
IEEE Security and Privacy analysis above extends this to attacks requiring no
added noise at all. The Fast Gradient Sign Method is the standard baseline
attack in this area and is the one used in Chapter 6.

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

This project addresses the first two directly. It evaluates four models of
increasing sophistication under a single protocol, measures in-domain and
cross-dataset performance for each, and applies a threshold
diagnostic that distinguishes calibration failure from representation failure
rather than reporting accuracy alone. It then applies a systematic
degradation suite and a white-box adversarial attack to the same model, and
tests whether one mechanism accounts for both the generalisation failure and
the robustness profile.

The contribution is therefore not a new detection architecture. It is a
structured examination of *how* and *why* a conventional detector fails outside
its training distribution, conducted under documented resource constraints.

---

## Verified reference list

Every source below has been checked against the original paper: authors,
venue, year, and the specific claim attributed to it.

| Source | Verified detail |
|---|---|
| Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. and Bengio, Y. (2014) *Generative Adversarial Nets*. NeurIPS, pp. 2672–2680 | Authors, venue, pages |
| Karras, T., Laine, S. and Aila, T. (2019) *A Style-Based Generator Architecture for Generative Adversarial Networks*. CVPR, pp. 4401–4410 | Authors, venue, pages; FFHQ 70k at 1024×1024 |
| Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J. and Nießner, M. (2019) *FaceForensics++: Learning to Detect Manipulated Facial Images*. ICCV | Four manipulation methods, compression levels, Xception baseline |
| Frank, J., Eisenhofer, T., Schönherr, L., Fischer, A., Kolossa, D. and Holz, T. (2020) *Leveraging Frequency Analysis for Deep Fake Image Recognition*. ICML | Upsampling as the origin of frequency artefacts |
| Li, Y., Yang, X., Sun, P., Qi, H. and Lyu, S. (2020) *Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics*. CVPR, pp. 3207–3216 | 5,639 videos; benchmark-realism argument |
| Dolhansky, B., Bitton, J., Pflaum, B., Lu, J., Howes, R., Wang, M. and Canton Ferrer, C. (2020) *The DeepFake Detection Challenge (DFDC) Dataset*. arXiv:2006.07397 | 100,000+ clips, 3,426 consenting actors |
| Le, T.-N., Nguyen, H.H., Yamagishi, J. and Echizen, I. (2021) *OpenForensics*. ICCV | In-the-wild multi-face forgery framing |
| Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L.-C. (2018) *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR, pp. 4510–4520 | Inverted residual / linear bottleneck design |
| Tan, M. and Le, Q.V. (2019) *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. ICML, pp. 6105–6114 | Compound scaling |
| Yan, Z., Zhang, Y., Yuan, X., Lyu, S. and Wu, B. (2023) *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection*. arXiv:2307.01426 | 15 methods, 9 datasets; comparability argument |
| *An Analysis of Recent Advances in Deepfake Image Detection in an Evolving Threat Landscape*. IEEE S&P 2024. arXiv:2404.16212 | 8 detectors; user-customised generators; no-noise semantic attack. **Author list still to confirm** |

### Claims deliberately generalised rather than attributed

Three passages originally named sources found through database searches but
not read in full. Rather than cite them, the claims were either generalised or
supported from a verified reference instead:

- Transformer-versus-CNN degradation percentages — removed; the figures are quoted inconsistently across the literature and the protocol was not verified
- GAN fingerprinting and fingerprint-removal work — the general claim is retained, supported by Frank et al. (2020), without naming specific papers
- Two recent generalisation surveys — reduced to a general statement about the field rather than a specific attribution

This is a deliberate trade of breadth for accuracy. Lesson 5 warns against
reference lists that appear assembled rather than read, and eleven verified
sources are more defensible than twenty of uncertain provenance.

### Outstanding

- Confirm the author list for arXiv:2404.16212
- Consider adding sources for Chapters 3, 5 and 8, which currently carry none
- Import to Zotero and export in Arden Harvard format
- Cross-check every in-text citation against the final reference list
