# Chapter 2: Literature Review and Fact Finding

> **DRAFT v2 — 11 August 2026.** All sources cited in this chapter have been
> verified against the original papers. Claims that could not be sourced to a
> verified reference have been removed or generalised rather than attributed to
> a paper that was not read — see §2.8.

---

## 2.1 Introduction to the review

This chapter reviews the background to the project: how deepfakes are produced,
how detection has developed, which datasets the field relies on, and the two
problems motivating this dissertation — poor generalisation across manipulation
methods, and fragility under everyday image degradation. The search strategy
and inclusion criteria are set out in section 3.1.

## 2.2 How deepfakes are generated

The term deepfake covers several techniques, and separating them matters
because the technique used has a large effect on what a detector learns.

Generative Adversarial Networks, introduced by Goodfellow et al. (2014), train
two networks against each other: a generator producing images and a
discriminator trying to tell them from real ones. By StyleGAN (Karras, Laine
and Aila, 2019) GANs could synthesise entire faces at 1024×1024 that most
people cannot identify as artificial. These are *fully synthetic* — no real
person is depicted. StyleGAN produced the fake half of this project's training
dataset. One architectural detail matters later: like all current GAN
architectures it builds an image by progressively **upsampling** from a
low-resolution representation, and section 2.4 identifies that upsampling as
the origin of the artefacts detectors exploit.

A second family performs *face swapping*, replacing the face region in an
existing photograph with someone else's and leaving background and body
untouched. This is what most people mean colloquially by a deepfake and lies
behind most reported misuse. More recently diffusion models have begun
displacing GANs for synthesis, which matters here because both datasets used
are GAN-era — a limitation taken up in Chapter 7.

The distinction between full synthesis and localised face swapping turns out to
be central: a detector trained on one is not obviously being trained for the
other, even though both are labelled "fake".

## 2.3 Detection approaches

### Handcrafted features

Early work used handcrafted features, looking for inconsistencies a generator
was likely to get wrong — irregular blinking, inconsistent head pose,
mismatched lighting, blending boundaries around a swapped face. These are
interpretable, but each targets a specific flaw and degrades once generators
address it; the history of the field is largely a sequence of such cues being
identified and then closed. The classical descriptors used as a baseline here,
Histogram of Oriented Gradients and Local Binary Patterns, belong to this
tradition, though they are applied as general texture and gradient descriptors
rather than as detectors of any specific artefact.

### Convolutional neural networks

The field then moved almost entirely to convolutional neural networks, trained
end to end on labelled images so they learn their own features. Architectures
developed for general image classification are typically borrowed, initialised
from ImageNet weights and fine-tuned. Rössler et al. (2019) established
XceptionNet as a strong baseline in the FaceForensics++ paper, and reported
accuracies on standard benchmarks are consistently above 95%.

Two architectures are used here, for reasons discussed in Chapter 4.
**MobileNetV2** (Sandler et al., 2018) is built around an inverted residual
structure in which each block's input and output are narrow bottlenecks and the
intermediate layer is an expanded representation filtered by lightweight
depthwise convolutions; the authors also argue against non-linearities in the
narrow layers, which destroy representational capacity. The result is far fewer
parameters than a conventional CNN of comparable accuracy, which is what makes
it viable on CPU. **EfficientNet-B0** (Tan and Le, 2019) introduces compound
scaling, increasing depth, width and input resolution together by a fixed ratio
rather than independently. Its building block is a similar mobile inverted
bottleneck but the network is larger, allowing a comparison at matched
trainable capacity.

Neither architecture was designed for forensic analysis. Both are
general-purpose classifiers, and the assumption underlying their use — that
features learned for object recognition transfer usefully to detecting
generator artefacts — is rarely examined explicitly. Chapter 6 bears on it.

### Transformers

Transformer architectures have also been applied, with some evidence they
generalise across datasets better than CNNs. Comparative figures are avoided
here because "degradation" is defined inconsistently in this literature and
quoting a percentage without verifying the protocol behind it would be unsafe.
The qualitative direction still matters: it suggests the generalisation failure
in Chapter 6 may be partly architectural rather than intrinsic to the task.
Transformers were not used because of the CPU-only constraint — attention over
image patches is substantially more expensive than depthwise convolution at the
resolutions required.

### Frequency-domain approaches

A third strand works in the frequency domain rather than on pixels directly,
treated separately in section 2.4 because it supplies the mechanism explaining
several of this project's results.

## 2.4 Frequency-domain artefacts

Frank et al. (2020) carried out what appears to be the first systematic
frequency-domain analysis of GAN-generated images, finding severe and easily
identified artefacts in frequency space that are consistent across network
architectures, datasets and resolutions.

The important part is the explanation rather than the observation. They
demonstrate that the artefacts are caused by the **upsampling operations**
present in all current GAN architectures — the steps progressively increasing
spatial resolution as an image is generated — and describe this as structural
and fundamental to how GANs produce images rather than an incidental flaw of
one model.

This matters a great deal here. If the discriminative signal is concentrated in
the high-frequency band, two consequences follow. Anything reducing image
resolution should reduce accuracy, because it discards the band carrying the
evidence; and any processing that removes high-frequency detail — compression,
blurring, downscaling — should damage a detector far more than processing that
leaves fine structure intact. Both were tested here and both confirmed
(Chapters 6 and 7). Frank et al. therefore supplies the theoretical grounding
for results I first observed empirically without expecting them.

A related line of work treats these artefacts as generator "fingerprints",
specific enough to attribute an image to the model that produced it. The
corollary is uncomfortable: a signal distinctive enough to identify a generator
is, by construction, one that changes when the generator does. Chapter 6's
cross-dataset results are consistent with that reading.

## 2.5 Benchmark datasets

Several datasets have become standard in this field.

**FaceForensics++** (Rössler et al., 2019) is the most widely used benchmark,
providing sequences manipulated by four techniques — DeepFakes, Face2Face,
FaceSwap and NeuralTextures — at several compression levels. Although
distributed as video it is commonly used frame-by-frame for still-image
detection, which is how it would have been used here. The authors report that
domain-specific knowledge improves accuracy even under strong compression and
that automated detectors clearly outperform human observers. Access requires a
signed agreement with the hosting institution.

This project would have benefited from it specifically because it varies
manipulation *method* and compression *level* independently — the two-factor
design Chapters 6 and 7 approximate using two datasets plus a synthetic
degradation suite. Chapter 3 explains why it could not be used.

**The Deepfake Detection Challenge dataset** (Dolhansky et al., 2020), released
by Facebook for a Kaggle competition, contains over 100,000 clips from 3,426
paid actors produced by several Deepfake, GAN-based and non-learned methods.
Being a video corpus places it outside this project's scope twice over: the
manipulations are face swaps within sequences rather than synthetic stills, and
the corpus runs to several hundred gigabytes. One detail is worth noting
regardless — all subjects consented to having their likenesses modified, unlike
datasets assembled from scraped web imagery, a point Chapter 3's ethical
discussion returns to.

**Celeb-DF** (Li et al., 2020) contains 5,639 high-quality celebrity forgeries
from an improved synthesis process. Its motivating argument is directly
relevant: the authors contend existing datasets are of low visual quality and
do not resemble deepfakes actually circulating, making benchmarks
unrealistically easy. That criticises benchmark validity rather than any
detector, reinforcing this chapter's case that in-domain accuracy is a weak
proxy for deployed performance.

**OpenForensics** (Le et al., 2021) targets multi-face forgery detection and
segmentation "in-the-wild", localising manipulated faces among several people
in unrestricted scenes, which the authors argue is substantially harder than
conventional deepfake recognition. It provides the cross-dataset evaluation set
here.

The **Flickr-Faces-HQ (FFHQ)** dataset (Karras, Laine and Aila, 2019) is not a
deepfake dataset but supplies the real images for the training set, and is what
StyleGAN was trained on. Its 70,000 images at 1024×1024 offer greater variation
in accessories, age, ethnicity and background than the CelebA-HQ dataset it
largely displaced. Two properties matter here: its images are
**landmark-aligned**, which this project exploited when verifying the two
datasets do not share a real-image source (Chapter 5); and its improvement in
demographic variation is relative rather than absolute, since it provides no
demographic labels — which is why the audit discussed in Chapter 7 could not be
performed.

These datasets differ not only in generation method but in capture conditions,
resolution, compression history and alignment. A model trained on one and
tested on another faces a shift on several axes at once, so attributing a
generalisation failure to any single one is not straightforward.

## 2.6 The generalisation problem

This is the central issue the project addresses. Detectors report very high
accuracy when trained and tested on the same dataset, but accuracy falls
sharply on manipulations produced by a method they were not trained on. Since a
real-world detector will inevitably meet techniques that did not exist when it
was trained, in-domain accuracy alone is a poor indicator of practical value.

Yan et al. (2023) attempt to quantify this with **DeepfakeBench**, the field's
first comprehensive benchmark by their description, integrating 15 detection
methods and 9 datasets behind standardised protocols. They argue the absence of
such a benchmark produces *"unfair performance comparisons and potentially
misleading results"*, because data pipelines, experimental settings and metrics
all differ between papers. This cuts two ways here: published accuracies cannot
be confidently compared across papers, which is part of why this dissertation
evaluates its own models under one fixed protocol rather than quoting figures
from elsewhere; and the ~97% commonly cited for the dataset used here is
indicative rather than a target.

The explanations offered reduce to the same idea: detectors learn artefacts
specific to the generator in their training data rather than any general notion
of manipulation. Frank et al.'s finding that upsampling artefacts are
structural to GANs supports this — a generator-specific fingerprint disappears
when the generator changes.

There is an important counterexample. Wang et al. (2020) trained a classifier
on a single generator (ProGAN) and found it generalised "surprisingly well" to
eleven other CNN-based generators, concluding that CNN-generated images of that
era shared systematic flaws. Two details matter: the result depended on careful
pre-processing, post-processing and augmentation rather than on the classifier,
and all eleven generators were convolutional. Chai et al. (2020) reach a
compatible conclusion using a patch-based classifier with a deliberately
limited receptive field, showing detectability concentrates in local texture.
Together these suggest transfer within a generative family is achievable and
that augmentation is the lever — a useful benchmark against which to read
Chapter 6, where no such augmentation was applied.

Across the family boundary the picture reverts. Corvi et al. (2023) asked
whether detectors developed for GAN imagery transfer to diffusion-generated
images, specifically under the compression and resizing social platforms
impose. Both datasets used here are GAN-era, so their question is the one a
user pointing this detector at a Midjourney image is implicitly asking.
Chapter 7 returns to it.

What is less clear, and what this project examines directly, is whether
improving in-domain performance helps or hurts transfer. If detectors succeed
by exploiting generator-specific artefacts, a better detector may simply be one
exploiting them more thoroughly, in which case better in-domain accuracy would
accompany *worse* relative transfer. Chapter 6 tests this.

## 2.7 Adversarial robustness and real-world degradation

Two distinct threats are usually treated separately, and both are relevant here.

The first is ordinary degradation. Images shared through social platforms are
recompressed, resized and re-encoded, often repeatedly. Since the forensic
evidence sits in the high-frequency band and compression is designed
specifically to discard high-frequency information as perceptually
unimportant, there is a direct conflict between how images are distributed and
how detectors work.

The second is deliberate adversarial attack, and the published figures are
stark. Carlini and Farid (2020) took a classifier achieving an AUC of 0.95
across almost all existing generators and, with full access, reduced it to
0.0005 by flipping the lowest bit of each pixel. Perturbing 1% of the image
area gave 0.08, and a black-box attack with no access to the classifier still
reached 0.22. A detector defeated this comprehensively is not weakened but
inverted. The Fast Gradient Sign Method used in Chapter 6 is the standard
baseline in this area and considerably weaker than any of these.

Abdullah et al. (2024) extend both threats. Examining eight state-of-the-art
detectors, they argue these are "far from being ready for deployment" for two
reasons: lightweight methods for customising large generative models let an
attacker produce many bespoke generators, against which existing defences
generalise poorly; and vision foundation models can be misused to craft
adversarial deepfakes **without adding any adversarial noise at all**, instead
manipulating image content semantically.

The first reinforces the case for cross-dataset evaluation: if the number of
generators in circulation is growing rapidly, in-domain accuracy becomes an
ever weaker proxy for deployed performance. The second limits this project's
adversarial testing. FGSM perturbs pixels within a small norm budget, so an
attack operating through semantic manipulation would be captured by no measure
used here, nor addressed by the adversarial training proposed in Chapter 8;
these results are a lower bound on vulnerability rather than a characterisation
of it. A further qualification, met in this project's own results, is that
adversarial evaluation is easy to get wrong in ways that inflate apparent
attack strength — see Chapter 7.

## 2.8 Research gap and contribution

Three observations emerge from this review. First, cross-dataset generalisation
is a recognised weakness but is usually reported as a single accuracy figure
without analysis of *how* the failure occurs — a representation that has become
uninformative, one that has become actively misleading, and one that is merely
mis-calibrated are different failures with different implications, and are
rarely distinguished. Second, robustness and generalisation are usually
evaluated separately; if both stem from reliance on high-frequency generator
artefacts they should be studied together, because the same mechanism predicts
both. Third, most published work uses substantial GPU resources, with limited
discussion of what is achievable under constrained conditions.

This project addresses the first two directly, evaluating four models under a
single protocol, measuring in-domain and cross-dataset performance for each,
and applying a threshold diagnostic that separates calibration failure from
representation failure. It then applies a degradation suite and a white-box
adversarial attack to the same model and tests whether one mechanism accounts
for both. The contribution is therefore not a new architecture but a structured
examination of *how* and *why* a conventional detector fails outside its
training distribution, under documented resource constraints.
