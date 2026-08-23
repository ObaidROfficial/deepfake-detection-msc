# Chapter 2: Literature Review and Fact Finding

> **DRAFT v2, 11 August 2026.** All sources cited in this chapter have been
> verified against the original papers. Claims that could not be sourced to a
> verified reference have been removed or generalised rather than attributed to
> a paper that was not read. See §2.8.

---

## 2.1 Introduction to the review

This chapter reviews the background to the project: how deepfakes are produced,
how detection has developed, which datasets the field relies on, and the two
problems motivating this dissertation. Those two problems are poor
generalisation across manipulation methods, and fragility under everyday image
degradation. The search strategy and inclusion criteria are set out in section
3.1.

## 2.2 How deepfakes are generated

The term deepfake covers several techniques, and separating them matters
because the technique used has a large effect on what a detector learns.

Generative Adversarial Networks, introduced by Goodfellow et al. (2014), train
a generator and a discriminator against each other. By StyleGAN (Karras, Laine
and Aila, 2019) they could synthesise entire faces at 1024×1024 that most
people cannot identify as artificial. These are *fully synthetic*: no real
person is depicted. StyleGAN produced the fake half of this project's training
data. One architectural detail matters later. Like all current GAN
architectures it builds an image by progressively **upsampling** from a
low-resolution representation, and section 2.4 identifies that upsampling as
the origin of the artefacts detectors exploit.

A second family performs *face swapping*, replacing the face region in an
existing photograph and leaving background and body untouched. This is what
most people mean colloquially by a deepfake and lies behind most reported
misuse. More recently diffusion models have begun displacing GANs for
synthesis, which matters here because both datasets used are GAN-era. Chapter 7
takes that limitation up.

The distinction between full synthesis and localised face swapping turns out to
be central: a detector trained on one is not obviously being trained for the
other, even though both are labelled "fake".

## 2.3 Detection approaches

### Handcrafted features

Early work used handcrafted features, looking for inconsistencies a generator
was likely to get wrong, such as irregular blinking, inconsistent head pose,
mismatched lighting and blending boundaries around a swapped face. These are
interpretable, but each targets a specific flaw and degrades once generators
address it. The classical descriptors used as a baseline here, Histogram of
Oriented Gradients and Local Binary Patterns, belong to this tradition, though
applied as general texture and gradient descriptors rather than as detectors of
any specific artefact.

### Convolutional neural networks

The field then moved almost entirely to convolutional neural networks, trained
end to end on labelled images so they learn their own features. Architectures
developed for general image classification are typically borrowed, initialised
from ImageNet weights and fine-tuned. Rössler et al. (2019) established
XceptionNet as a strong baseline in the FaceForensics++ paper, and reported
accuracies on standard benchmarks are consistently above 95%.

Two architectures are used here. **MobileNetV2** (Sandler et al., 2018) is
built on inverted residual blocks with lightweight depthwise convolutions and
has far fewer parameters than a conventional CNN of comparable accuracy.
**EfficientNet-B0** (Tan and Le, 2019) scales depth, width and input resolution
together by a fixed ratio rather than independently. Chapter 4 sets out why
these two.

Neither was designed for forensic analysis. Both are general-purpose
classifiers, and the assumption underlying their use is rarely examined
explicitly: that features learned for object recognition transfer usefully to
detecting generator artefacts. Chapter 6 bears on it.

### Transformers

Transformer architectures have also been applied, with some evidence they
generalise across datasets better than CNNs. Comparative figures are avoided
here because "degradation" is defined inconsistently in this literature. The
qualitative direction still matters: it suggests the generalisation failure in
Chapter 6 may be partly architectural rather than intrinsic to the task.
Transformers were not used because of the CPU-only constraint, attention over
image patches being far more expensive than depthwise convolution.

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
present in all current GAN architectures, and describe this as structural to
how GANs produce images rather than an incidental flaw of one model.

This matters a great deal here. If the discriminative signal sits in the
high-frequency band, two consequences follow. Anything reducing image
resolution should reduce accuracy, because it discards the band carrying the
evidence. And any processing that removes high-frequency detail, such as
compression, blurring or downscaling, should damage a detector far more than
processing that leaves fine structure intact. Both were tested here and both
confirmed (Chapters 6 and 7). Frank et al. therefore supplies the theoretical
grounding for results I first observed empirically without expecting them.

A related line of work treats these artefacts as generator "fingerprints",
specific enough to attribute an image to the model that produced it. The
corollary is uncomfortable: a signal distinctive enough to identify a generator
is, by construction, one that changes when the generator does.

## 2.5 Benchmark datasets

Several datasets have become standard in this field.

**FaceForensics++** (Rössler et al., 2019) is the most widely used benchmark,
providing sequences manipulated by four techniques at several compression
levels, and access requires a signed agreement with the hosting institution.
This project would have benefited from it specifically because it varies
manipulation *method* and compression *level* independently. That is the
two-factor design Chapters 6 and 7 approximate using two datasets plus a
synthetic degradation suite. Chapter 3 explains why it could not be used.

**The Deepfake Detection Challenge dataset** (Dolhansky et al., 2020) contains
over 100,000 clips from 3,426 paid actors. Being a video corpus of several
hundred gigabytes places it outside this project's scope twice over. One detail
is worth noting regardless: all subjects consented to having their likenesses
modified, unlike datasets assembled from scraped web imagery, a point Chapter
3's ethical discussion returns to.

**Celeb-DF** (Li et al., 2020) contains 5,639 high-quality celebrity forgeries
from an improved synthesis process. Its motivating argument is directly
relevant: the authors contend existing datasets are of low visual quality and
do not resemble deepfakes actually circulating, making benchmarks
unrealistically easy. That criticises benchmark validity rather than any
detector, reinforcing the case that in-domain accuracy is a weak proxy for
deployed performance.

**OpenForensics** (Le et al., 2021) targets multi-face forgery detection and
segmentation "in-the-wild", localising manipulated faces among several people
in unrestricted scenes, which the authors argue is substantially harder than
conventional deepfake recognition. It provides the cross-dataset evaluation set
here.

The **Flickr-Faces-HQ (FFHQ)** dataset (Karras, Laine and Aila, 2019) is not a
deepfake dataset but supplies the real images for the training set, and is what
StyleGAN was trained on. Two properties matter here. Its images are
**landmark-aligned**, which this project exploited when verifying the two
datasets do not share a real-image source (Chapter 5). And its improved
demographic variation is relative rather than absolute, since it provides no
demographic labels, which is why the audit discussed in Chapter 7 could not be
performed.

These datasets differ not only in generation method but in capture conditions,
resolution, compression history and alignment. A model trained on one and
tested on another faces a shift on several axes at once, so attributing a
generalisation failure to any single axis is not straightforward.

## 2.6 The generalisation problem

This is the central issue the project addresses. Detectors report very high
accuracy when trained and tested on the same dataset, but accuracy falls
sharply on manipulations produced by a method they were not trained on. Since a
real-world detector will inevitably meet techniques that did not exist when it
was trained, in-domain accuracy alone is a poor indicator of practical value.

Yan et al. (2023) attempt to quantify this with **DeepfakeBench**, integrating
15 detection methods and 9 datasets behind standardised protocols. They argue
the absence of such a benchmark produces *"unfair performance comparisons and
potentially misleading results"*, because data pipelines, settings and metrics
all differ between papers. That is part of why this dissertation evaluates its
own models under one fixed protocol rather than quoting figures from elsewhere,
and why the ~97% commonly cited for the dataset used here is treated as
indicative rather than as a target.

The explanations offered reduce to the same idea: detectors learn artefacts
specific to the generator in their training data rather than any general notion
of manipulation. Frank et al.'s finding that upsampling artefacts are
structural to GANs supports this, because a generator-specific fingerprint
disappears when the generator changes.

There is an important counterexample. Wang et al. (2020) trained a classifier
on a single generator (ProGAN) and found it generalised "surprisingly well" to
eleven other CNN-based generators, though the result depended on careful
pre-processing and augmentation rather than on the classifier, and all eleven
generators were convolutional. Chai et al. (2020) reach a compatible conclusion
using a patch-based classifier with a deliberately limited receptive field,
showing detectability concentrates in local texture. Together these suggest
transfer within a generative family is achievable and that augmentation is the
lever. That is a useful benchmark against which to read Chapter 6, where no
such augmentation was applied.

Across the family boundary the picture reverts. Corvi et al. (2023) asked
whether detectors developed for GAN imagery transfer to diffusion-generated
images, under the compression and resizing social platforms impose. Both
datasets used here are GAN-era, so their question is the one a user pointing
this detector at a Midjourney image is implicitly asking.

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
0.0005 by flipping the lowest bit of each pixel. A black-box attack with no
access still reached 0.22. A detector defeated this comprehensively is not
weakened but inverted. The Fast Gradient Sign Method used in Chapter 6 is the
standard baseline in this area and considerably weaker than any of these.

Abdullah et al. (2024) extend both threats. Examining eight state-of-the-art
detectors, they argue these are "far from being ready for deployment" for two
reasons. Lightweight customisation of large generative models lets an attacker
produce many bespoke generators, against which existing defences generalise
poorly. And vision foundation models can craft adversarial deepfakes **without
adding any adversarial noise at all**, manipulating image content semantically.

The first reinforces the case for cross-dataset evaluation: if the number of
generators in circulation is growing rapidly, in-domain accuracy becomes an
ever weaker proxy for deployed performance. The second limits this project's
adversarial testing. FGSM perturbs pixels within a small norm budget, so a
semantic attack would be captured by no measure used here, nor addressed by the
adversarial training proposed in Chapter 8. These results are a lower bound on
vulnerability rather than a characterisation of it. A further qualification,
met in this project's own results, is that adversarial evaluation is easy to
get wrong in ways that inflate apparent attack strength, as Chapter 7 describes.

## 2.8 Research gap and contribution

Three observations emerge from this review. First, cross-dataset generalisation
is a recognised weakness but is usually reported as a single accuracy figure
without analysis of *how* the failure occurs. A representation that has become
uninformative, one that has become actively misleading, and one that is merely
mis-calibrated are different failures with different implications, and they are
rarely distinguished. Second, robustness and generalisation are usually
evaluated separately, but if both stem from reliance on high-frequency
generator artefacts they should be studied together. Third, most published work
uses substantial GPU resources, with limited discussion of what is achievable
under constrained conditions.

This project addresses the first two directly, evaluating four models under a
single protocol and applying a threshold diagnostic that separates calibration
failure from representation failure. It then applies a degradation suite and a
white-box adversarial attack to the same model and tests whether one mechanism
accounts for both. The contribution is therefore not a new architecture but a
structured examination of *how* and *why* a conventional detector fails outside
its training distribution, under documented resource constraints.
