# Chapter 1: Introduction and Project Framing

> **DRAFT v1, 11 August 2026.** Target 1,300 words.
> Written after the results chapters, per the sequence recommended in Lesson 5.

---

## 1.1 Background

Synthetic media has moved from a research curiosity to a practical problem in
under a decade. Generative Adversarial Networks, introduced by Goodfellow et
al. (2014), made it possible to produce images corresponding to nothing that
ever existed, and by 2019 StyleGAN (Karras, Laine and Aila, 2019) could
synthesise faces at 1024×1024 that most people cannot reliably identify as
artificial. A parallel line of work made replacing one person's face with
another's in an existing photograph straightforward.

The techniques have legitimate applications in film, accessibility and
education, but the harmful uses are well documented: non-consensual intimate
imagery, impersonation fraud, political disinformation. The tools have also
become progressively more accessible, so the barrier is no longer technical
skill or computing resources. Detection has not kept pace, and a persistent gap
separates benchmark performance from behaviour on media encountered in the
real world.

## 1.2 Problem statement

Published deepfake detectors routinely report accuracies above 95% on standard
benchmarks such as FaceForensics++ (Rössler et al., 2019). Three things
complicate reading those figures as evidence of practical capability.

**Comparability.** Yan et al. (2023) argue that the absence of a standardised
benchmark produces "unfair performance comparisons and potentially misleading
results", because data pipelines, experimental settings and metrics differ
between papers. Headline accuracies are therefore not straightforwardly
comparable across studies.

**Generalisation.** A detector is trained on manipulations produced by methods
that existed when the training set was assembled, and a deployed detector will
encounter methods that did not. If detection depends on artefacts specific to
particular generators, accuracy on familiar manipulations may say very little
about unfamiliar ones.

**Robustness.** Benchmark images are pristine, whereas images circulating in
the world are compressed, resized and re-encoded, often repeatedly. If the
evidence a detector relies on is fine-grained, and compression is designed
specifically to discard fine-grained detail as perceptually unimportant, then
distribution itself may degrade detectability.

These are usually studied separately. This project treats them together, on the
hypothesis that they share a cause.

## 1.3 Rationale

A single accuracy figure says very little about whether a detector would be
useful. That detectors fail outside their training distribution is already
established. The interesting question is *how* and *why* they fail.

The distinction matters practically. A detector whose ranking is correct but
whose threshold is misplaced can be recalibrated. One whose representation
carries no relevant information cannot, and one whose ranking is actively
inverted is worse than useless. Accuracy alone cannot separate them. There is
also a resourcing rationale, in that most published work assumes substantial GPU
capacity whereas this project ran entirely on a CPU-only consumer laptop.

## 1.4 Aim

To design, develop and evaluate a machine learning artefact capable of
detecting deepfake media, and to critically assess its performance not only on
data resembling its training set but on unseen manipulation methods and
degraded inputs.

## 1.5 Objectives

1. Conduct a structured review of deepfake detection techniques, machine
   learning approaches, and the adversarial and generalisation threats
   identified in the literature.
2. Identify and prepare balanced, reproducible datasets of real and
   manipulated media from open-source repositories, using two distinct
   generation methods so that cross-method transfer can be measured.
3. Design and implement a convolutional neural network for deepfake detection
   using PyTorch, together with classical machine learning baselines for
   comparison.
4. Evaluate all models using accuracy, precision, recall, F1 and ROC-AUC, both
   in-domain and on an unseen dataset, and compare the proposed model against
   the baselines.
5. Test robustness against realistic input degradation and adversarial
   perturbation, and analyse the resulting limitations.
6. Document the development process, findings and critical evaluation in a
   final report, and deliver a working artefact.

## 1.6 Research questions

1. How effective are machine learning models in detecting deepfake media?
2. Which features or patterns are most useful in distinguishing real and fake
   media?
3. How does the performance of the proposed model compare to baseline
   approaches?
4. How robust is the detection model against manipulated or adversarial
   inputs?
5. What are the key limitations of current deepfake detection techniques?

## 1.7 Scope and limitations

The project addresses still images and extracted video frames, not temporal
analysis of video sequences, since approaches exploiting inconsistency across
frames are a substantial subfield of their own. Two datasets are used, giving
one cross-method comparison. A larger matrix would be more general but was not
feasible in the timeframe. Both datasets are GAN-era, and since diffusion
models are increasingly displacing GANs, the conclusions may not extend to
diffusion-generated content, particularly as the mechanism identified here is
specific to GAN architecture. Adversarial testing is limited to the Fast
Gradient Sign Method under white-box conditions, and black-box and iterative
attacks were not attempted.

The approved proposal specified FaceForensics++ and the DFDC corpus, and
neither proved usable. Substitutes were selected and the supervisor notified.
Chapter 3 discusses this fully and Chapter 7 acknowledges the cost to
comparability with published work.

## 1.8 Report structure

**Chapter 2** reviews the literature on generation techniques, detection
approaches, benchmark datasets, generalisation and robustness, and identifies
the research gap.

**Chapter 3** sets out the research and development methodology, the
requirements, the deviation from the approved proposal, ethical and legal
considerations, and the risk register.

**Chapter 4** describes the design of the pipeline, the model selection
rationale, and the experimental and evaluation framework.

**Chapter 5** documents the implementation, including the demonstration tools
(a graphical interface, a command line classifier and the installer that
packages them) and a detailed account of the defects and misjudgements
encountered and how they were resolved.

**Chapter 6** presents the results: in-domain performance, cross-dataset
transfer, the resolution ablation, the robustness suite and the adversarial
evaluation.

**Chapter 7** evaluates the work critically, answering each research question,
assessing the artefact's strengths and weaknesses, reflecting on the process,
and addressing the ethical, social, sustainability and global dimensions of the
topic.

**Chapter 8** concludes and identifies further work.

---

*Draft word count: approximately 1,300.*
