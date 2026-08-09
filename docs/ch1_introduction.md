# Chapter 1: Introduction and Project Framing

> **DRAFT v1 — 11 August 2026.** Target 1,300 words.
> Written after the results chapters, per the sequence recommended in Lesson 5.

---

## 1.1 Background

Synthetic media has moved from a research curiosity to a practical problem in
under a decade. Generative Adversarial Networks, introduced in 2014, made it
possible to produce images that do not correspond to anything that ever
existed, and by 2019 architectures such as StyleGAN could synthesise human
faces that most people cannot reliably identify as artificial. A parallel line
of work made it straightforward to replace one person's face with another's in
existing photographs and video.

The term deepfake now covers both. The techniques have legitimate applications
in film production, accessibility and education, but the harmful uses are
well documented: non-consensual intimate imagery, financial fraud through
impersonation, and political disinformation. The tools required have become
progressively more accessible, so the barrier to producing convincing forged
media is no longer technical skill or computing resources.

This creates a detection problem, and detection has not kept pace with
generation. Research systems report high accuracy on benchmark datasets, but
there is a persistent and widely acknowledged gap between benchmark performance
and behaviour on media encountered in the real world.

## 1.2 Problem statement

Published deepfake detectors routinely report accuracies above 95% on standard
benchmarks. Two things complicate reading those figures as evidence of
practical capability.

The first is generalisation. A detector is trained on manipulations produced by
methods that existed when the training set was assembled. A deployed detector
will encounter methods that did not. If detection depends on artefacts specific
to particular generators, then accuracy on familiar manipulations may say very
little about performance on unfamiliar ones.

The second is robustness. Benchmark images are pristine. Images that circulate
in the world are compressed, resized and re-encoded, often repeatedly. If the
evidence a detector relies on is fine-grained, and compression is designed
specifically to discard fine-grained detail as perceptually unimportant, then
distribution itself may degrade detectability.

These are usually studied separately. This project treats them together, on the
hypothesis that they may share a cause.

## 1.3 Rationale

The rationale for the project is that reporting a single accuracy figure tells
you very little about whether a detector would be useful, and that the
interesting question is not *whether* detectors fail outside their training
distribution — this is established — but *how* they fail and *why*.

The distinction matters practically. A detector whose ranking is correct but
whose threshold is misplaced can be fixed by recalibration. One whose learned
representation carries no relevant information cannot. One whose ranking is
actively inverted is worse than useless. These are different problems with
different remedies, and reporting accuracy alone cannot separate them.

There is also a resourcing rationale. Most published work in this area assumes
substantial GPU capacity. This project was carried out entirely on a CPU-only
consumer laptop, and documenting what is and is not achievable under that
constraint has some value for practitioners without access to better hardware.

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

The project addresses **still images and extracted video frames**, not
temporal analysis of video sequences. Approaches exploiting inconsistency
across frames are a substantial subfield in their own right and were outside
what could be attempted here.

Two datasets are used, providing one cross-method comparison. A larger matrix
of generation methods would give a more general result but was not feasible
within the timeframe.

Both datasets are **GAN-era**. Diffusion models are increasingly displacing
GANs for image synthesis, and the conclusions here may not extend to
diffusion-generated content — particularly since the mechanism identified is
specific to GAN architecture.

Adversarial testing is limited to the Fast Gradient Sign Method under
white-box conditions, where the attacker has full access to model gradients.
Black-box and iterative attacks were not attempted.

The approved proposal specified FaceForensics++ and the DFDC corpus. Neither
proved usable: the former requires a licence agreement with a turnaround
measured in weeks, and the latter is approximately 470GB of video that cannot
be processed on the available hardware. Substitutes were selected and the
supervisor notified. This is discussed fully in Chapter 3, and the cost to
comparability with published work is acknowledged in Chapter 7.

## 1.8 Summary of findings

The results are stated here briefly so the reader knows where the report is
heading.

Four models were evaluated: logistic regression on raw pixels, a HOG and LBP
pipeline with a linear SVM, and two fine-tuned convolutional networks —
MobileNetV2 and EfficientNet-B0 — configured to matched trainable capacity.
In-domain accuracy rose from 73.6% to 94.6%. On an unseen manipulation method,
all four scored between 47.9% and 51.1%, indistinguishable from chance on a
balanced set. The generalisation gap therefore widened monotonically, from 24.7
points to 44.2, as the models improved. Because the two CNNs are structurally
very different and fail identically, the result is not attributable to a single
architecture choice.

A threshold diagnostic confirmed this is a representation failure rather than a
calibration failure: no decision threshold in either polarity recovers usable
accuracy.

Robustness testing showed the detector is highly sensitive to any perturbation
that removes or masks high-frequency detail, and largely insensitive to changes
that preserve it. Under white-box adversarial attack at an imperceptible
perturbation size, accuracy falls from 94.3% to 5.2% and the model's ranking
inverts.

A single mechanism accounts for both sets of results, and it was used to make
predictions that were recorded before the experiments testing them.

## 1.9 Report structure

**Chapter 2** reviews the literature on generation techniques, detection
approaches, benchmark datasets, generalisation and robustness, and identifies
the research gap.

**Chapter 3** sets out the research and development methodology, the
requirements, the deviation from the approved proposal, ethical and legal
considerations, and the risk register.

**Chapter 4** describes the design of the pipeline, the model selection
rationale, and the experimental and evaluation framework.

**Chapter 5** documents the implementation, including a detailed account of
the defects and misjudgements encountered and how they were resolved.

**Chapter 6** presents the results: in-domain performance, cross-dataset
transfer, the resolution ablation, the robustness suite and the adversarial
evaluation.

**Chapter 7** evaluates the work critically — answering each research question,
assessing the artefact's strengths and weaknesses, reflecting on the process,
and addressing the ethical, social, sustainability and global dimensions of the
topic.

**Chapter 8** concludes and identifies further work.

---

*Draft word count: approximately 1,300.*
