# Chapter 2 — Literature Review: source collection

Working bibliography and thematic notes. **Not report prose.**

Started 7 August 2026. Target 45–60 sources, majority peer-reviewed, 2019–2026.

> **Verification rule for this file.** Every entry must be read (at least
> abstract + methods) before it goes into the report. Anything listed here from
> a search snippet alone is marked `[UNVERIFIED]` and must be checked against
> the actual paper. Lesson 5 explicitly warns against reference lists that look
> assembled rather than read, and markers do check.

---

## Structure of the chapter (2,600 words)

| § | Topic | Words | Purpose |
|---|---|---|---|
| 2.1 | How deepfakes are generated — GANs, autoencoders, diffusion | 400 | Establishes *why* artefacts exist at all |
| 2.2 | Detection approaches: handcrafted → CNN → transformer | 600 | The technical lineage; positions our baselines and MobileNetV2 |
| 2.3 | Frequency-domain and artefact-based detection | 400 | Directly relevant — our HOG/LBP finding is a spatial analogue |
| 2.4 | Benchmark datasets compared | 350 | Justifies dataset choice + the FF++ deviation |
| 2.5 | **The generalisation problem** | 500 | Core of the argument — where our contribution sits |
| 2.6 | Adversarial robustness and real-world degradation | 350 | Motivates the robustness suite |
| 2.7 | Research gap | 200 | Explicit statement of what this project adds |

---

## §2.5 Generalisation — the chapter's centre of gravity

This section carries the project's argument and should be written first.

**Key claim to support:** deepfake detectors report near-perfect in-domain
accuracy and degrade sharply on unseen generation methods. Our baselines
demonstrate this concretely (75.3% → 48.9%, see `ch6_baseline_findings.md`).

### Sources

- **DeepfakeBench** (Yan et al., 2023) — `arXiv:2307.01426`. Unified benchmark;
  standardised comparison across detectors and datasets. Likely the single most
  citable source for "cross-dataset performance collapses". Code:
  github.com/SCLBD/DeepfakeBench. `[UNVERIFIED — read in full]`
- **Deepfake Detection that Generalizes Across Benchmarks** — `arXiv:2508.06248`.
  Evaluates 14 benchmark datasets 2019–2025. Reports that tuning only
  LayerNorm parameters (0.03% of weights) plus L2-normalised hyperspherical
  feature manifold improves cross-dataset AUROC over more complex methods.
  Very relevant to §2.5 and §2.7. `[UNVERIFIED]`
- **DeepFake video detection: Insights into model generalisation — a systematic
  review** — ScienceDirect `S2543925125000075`. Systematic review, explicitly
  on generalisation. `[UNVERIFIED]`
- **Cross-dataset video deepfake detection using Transformer and CNN
  architectures** — Springer, Machine Vision and Applications
  `10.1007/s00138-026-01809-w`. Reports transformers degrade ~11.33%
  cross-dataset vs >15% for CNNs. **Useful specific figure — but verify the
  protocol before quoting, as "degradation" is defined differently across
  papers.** `[UNVERIFIED]`

### Connection to our result — worth foregrounding

Most of this literature reports generalisation failure as a *drop in accuracy*.
Our HOG+LBP result is more specific: AUC falls **below** chance (0.3986), and
inverting the decision rule recovers 57.9%. That is a *sign reversal*, not
merely information loss.

**To check:** does any of this literature report below-chance cross-dataset AUC
explicitly? If it is under-discussed, that sharpens the contribution claim in
§2.7. If it is well known, we cite it and position our result as a
demonstration rather than a novelty. Either outcome is fine — but the claim
must not be made without checking.

---

## §2.3 Frequency-domain detection

Directly relevant: our finding is that *spatial* gradient/texture descriptors
show sign reversal across datasets. The frequency-domain literature is the
established parallel.

- **Frank et al. (2020), Leveraging Frequency Analysis for Deep Fake Image
  Recognition** — PMLR v119. Foundational. Grid-like frequency artefacts arise
  from generator **upsampling layers**. Explains *why* GAN images differ in
  high-frequency structure — mechanistically underpins our sharpness
  observation. `[UNVERIFIED — priority read]`
- **Yu et al.** — introduced GAN "fingerprints" for attributing synthetic
  images. `[UNVERIFIED — need full citation]`
- **GANprintR** — removes GAN fingerprints, defeating fingerprint-based
  identification. Good for the limitations discussion. `[UNVERIFIED]`
- **Rethinking the Up-Sampling Operations in CNN-based Generative Networks** —
  `arXiv:2312.10461`. `[UNVERIFIED]`
- **Towards the Detection of Diffusion Model Deepfakes** — `arXiv:2210.14571`.
  Needed because diffusion models are displacing GANs; both our datasets are
  GAN-era, which is a limitation to state honestly in Ch7. `[UNVERIFIED]`
- **Anisotropic multiresolution analyses for deepfake detection** —
  `arXiv:2210.14874`. Wavelet packet transform features. `[UNVERIFIED]`

---

## §2.6 Robustness and real-world degradation

Motivates the robustness suite (JPEG, noise, blur, downscale, FGSM).

- **An Analysis of Recent Advances in Deepfake Image Detection in an Evolving
  Threat Landscape** — `arXiv:2404.16212`, also IEEE S&P 2024 (Virginia Tech).
  Peer-reviewed venue, strong citation. Notes most detection papers perform
  only *basic* robustness evaluation (blur, JPEG, downsample, noise) — which
  is a direct justification for our suite and for going further with FGSM.
  `[UNVERIFIED — priority read]`
- **Adversarial Deepfakes: Evaluating Vulnerability of Deepfake Detectors to
  Adversarially Crafted Fake Videos** — WACV 2021, UCSD. Peer-reviewed.
  `[UNVERIFIED]`
- **Exposing Vulnerabilities of Deepfake Detection Systems with Robust
  Attacks** — ACM Digital Threats: Research and Practice, `10.1145/3464307`.
  ACM venue — good for the Arden requirement to use professional body sources.
  `[UNVERIFIED]`
- Claim to verify and cite: strong compression suppresses high-frequency
  forensic traces. This is *precisely* the mechanism our JPEG robustness test
  probes, and it predicts our detector should degrade with quality. Another
  pre-registered prediction if we state it before running.

---

## §2.4 Datasets

- **FaceForensics++** (Rössler et al., 2019) — must cite even though not used;
  the deviation in Ch3 depends on establishing it as the field standard.
- **DFDC** (Dolhansky et al., 2020) — same reason.
- **Celeb-DF** (Li et al., 2020).
- **OpenForensics** (Le et al., ICCV 2021) — `arXiv:2107.14480`. **Dataset B's
  actual source.** Must be cited properly, not via the Kaggle mirror.
- **FFHQ / StyleGAN** (Karras et al., 2019) — origin of Dataset A's real images
  and the generator behind its fakes.
- **Deepfake-Eval-2024** — in-the-wild social media deepfakes, 2024. Useful for
  arguing that lab benchmarks overstate performance. `[UNVERIFIED]`

### Open question flagged in ch5_findings

Does OpenForensics draw its *real* images from FFHQ? If so, Datasets A and B
share a real-image source and the "different sources" claim weakens. **Must be
resolved from the OpenForensics paper before Ch3 is written.**

---

## Immediate next actions

1. [ ] Resolve the OpenForensics ↔ FFHQ overlap question — affects Ch3 and Ch6 claims
2. [ ] Read Frank et al. (2020) properly — mechanism behind our sharpness result
3. [ ] Read arXiv:2404.16212 — robustness protocol justification
4. [ ] Read DeepfakeBench — cross-dataset numbers to compare ours against
5. [ ] Check whether below-chance cross-dataset AUC is reported anywhere
6. [ ] Set up Zotero, import everything, configure Arden Harvard output
7. [ ] Replace every `[UNVERIFIED]` tag with a verified full citation

**Count so far:** ~18 candidates. Need 45–60. Weakest areas: §2.1 (generation
methods) and §2.2 (detection architectures) — both need dedicated searches.
