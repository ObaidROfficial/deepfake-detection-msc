# Chapter 6 — Baseline and CNN results

## 6.0 DEFINITIVE RESULT — all models at matched 224px resolution

Final configuration: 224px, `unfreeze_blocks: 3`, early stopping fired at
epoch 15 (best epoch 12, val loss 0.1390). 2.4 hours. **Properly converged** —
training accuracy (0.9503) now exceeds validation (0.9450), the opposite of the
underfitting signal seen in every earlier run.

Baselines re-extracted at matched 224px so the comparison is resolution-fair.

| Model | A (in-domain) | A AUC | B (cross) | B AUC | Best-any-threshold on B | Gap |
|---|---|---|---|---|---|---|
| Logistic Regression (32px pixels) | 0.7358 | 0.8121 | 0.4883 | 0.5063 | 0.5180 | 24.7 |
| HOG + LBP → SVC (224px) | 0.7860 | 0.8706 | 0.4790 | 0.4751 | 0.5317 | 30.7 |
| **MobileNetV2 (224px, converged)** | **0.9430** | **0.9869** | 0.5110 | 0.5593 | 0.5453 | **43.2** |

### The central finding

```
in-domain:      73.6%  ->  78.6%  ->  94.3%      (+20.7 points)
cross-dataset:  48.8%  ->  47.9%  ->  51.1%      (+2.3 points, chance throughout)
gap:            24.7   ->  30.7   ->  43.2       (monotonically WIDENING)
```

**Twenty points of in-domain improvement bought 2.3 points cross-dataset, and
the generalisation gap grew monotonically from 24.7 to 43.2 points.** Every
advance in representation quality was absorbed entirely by the source domain.

94.3% is now within ~3 points of the ~97% typically reported for this dataset,
so the in-domain result is credible and the comparison is being made from a
competent model rather than a crippled one. That matters: a weak model failing
to generalise proves little.

### An honest qualification

It would overstate the case to say "no signal transfers at all". The CNN's
cross-dataset AUC is **0.5593** — at n=6,000 the standard error on AUC is about
0.0075, so this sits roughly **7.9 standard errors above chance** and is
statistically real. It is also better than either baseline (0.5063, 0.4751).

So better representations *do* transfer marginally better. But the effect is
practically negligible: the best accuracy achievable at any threshold is
**0.5453** against a 0.50 floor. A detector operating at 54.5% on a balanced
set has no deployment value.

The correct claim is therefore: **improvements in in-domain accuracy yield
statistically detectable but practically useless gains in cross-dataset
transfer, while the absolute generalisation gap widens.** That is more
defensible than a blanket "CNNs do not generalise" and it is what the data
supports.

---

## 6.0-old HEADLINE RESULT (128px run, superseded — retained for the process narrative)

MobileNetV2 retrained to convergence: early stopping fired at epoch 16, best
weights from epoch 13 (val loss 0.4583). 47 minutes.

| Model | A (in-domain) acc | A AUC | B (cross) acc | B AUC | Best-any-threshold on B |
|---|---|---|---|---|---|
| Logistic Regression | 0.7358 | 0.8121 | 0.4883 | 0.5063 | 0.5180 |
| HOG + LBP → SVC | 0.7532 | 0.8313 | 0.4892 | 0.3986 | 0.5790 |
| **MobileNetV2 (converged)** | **0.7873** | **0.8715** | **0.5003** | 0.4819 | 0.5217 |

### The finding

**In-domain accuracy rises monotonically across the three models. Cross-dataset
accuracy does not move at all.**

```
in-domain:     73.6%  ->  75.3%  ->  78.7%     (+5.1 points)
cross-dataset: 48.8%  ->  48.9%  ->  50.0%     (+1.2 points, i.e. chance throughout)
```

Every gain from better representation learning is absorbed entirely by the
in-domain task. None of it transfers. A pretrained CNN fine-tuned to
convergence generalises to an unseen generation method no better than logistic
regression on 32×32 greyscale pixels.

The threshold diagnostic confirms this is not a calibration artefact: the best
accuracy achievable at **any** threshold in **either** polarity is 0.5217.
There is no decision rule that recovers usable performance. This is a
representation failure — the learned features carry essentially no information
about real-vs-fake on Dataset B.

This is the project's central claim and it is now supported by three
independent model families, one of them trained to convergence with early
stopping.

### Secondary observation

Training to convergence bought **+2.1 points in-domain** (76.6 → 78.7) and
**nothing** cross-dataset (50.25 → 50.03, both chance). The undertrained run
and the converged run tell the same story about generalisation. Worth stating,
because it means the finding is robust to how long the model trains.

---

## 6.0c RESOLUTION ABLATION — hypothesis confirmed

The §6.0b prediction was that 128px input was discarding the high-frequency
artefacts GAN detection depends on. Retrained at 224px, **changing only input
resolution**. This is the project's one clean controlled comparison.

| | 128px | 224px | Δ |
|---|---|---|---|
| In-domain accuracy | 0.7873 | **0.8880** | **+0.1007** |
| In-domain ROC-AUC | 0.8715 | **0.9600** | **+0.0885** |
| Cross-dataset accuracy | 0.5003 | 0.5135 | +0.0132 |
| Cross-dataset ROC-AUC | 0.4819 | 0.5305 | +0.0486 |
| Best-any-threshold on B | 0.5217 | 0.5260 | +0.0043 |
| **Generalisation gap** | 28.7 pts | **37.5 pts** | **+8.8** |

**Confirmed.** Input resolution alone is worth **+10.1 accuracy points and
+0.089 AUC** in-domain. This supports the frequency-domain account directly:
the discriminative signal for GAN-generated faces lives in high-frequency
detail (Frank et al., 2020 — grid artefacts from generator upsampling), and
downscaling 256 → 128 destroys it.

**This makes the central finding stronger, not weaker.** A ten-point in-domain
improvement bought **1.3 points** cross-dataset — still chance, and the
best-achievable-at-any-threshold barely moved (0.5217 → 0.5260).

> **The generalisation gap widened from 28.7 to 37.5 points.** Making the model
> better at the in-domain task made it *relatively worse* at transferring.
> The extra high-frequency information it learned to exploit is precisely the
> information that is generator-specific and does not survive a change of
> manipulation method.

That is a sharper claim than "detectors generalise poorly", and it is directly
evidenced: **the features that drive in-domain gains are the same features that
fail to transfer.**

### Second prediction, now testable

If high-frequency detail carries the signal, the robustness suite should show
large accuracy drops under **JPEG compression** and **downscaling**, which
attack exactly that band. Recorded before running the suite.

---

## 6.0b OPEN ISSUE — in-domain accuracy is well below the literature

78.7% against roughly **97%** commonly reported on this dataset. That gap needs
explaining before Chapter 6 is written, or a marker will ask.

The model is no longer undertrained: at the best epoch train accuracy (0.7844)
and validation accuracy (0.7827) are within 0.002, and by epoch 15–16 training
had begun to pull ahead (0.7938 vs 0.7715). It is fitting the training set
about as well as it fits validation. More epochs will not help.

**Leading hypothesis: input resolution.** Images are downscaled 256 → **128px**
before training. GAN detection depends heavily on fine high-frequency artefacts
(Frank et al., 2020 — grid-like frequency traces from generator upsampling
layers). Halving the resolution discards exactly that information. Published
results on this dataset generally train at 224px on the full 100k training set
with more of the backbone unfrozen.

This hypothesis is attractive because it is *internally consistent with our own
robustness suite*: downscaling is one of the perturbations we test, and if
resolution carries the signal we should also see large accuracy drops under the
downscale perturbations. Two independent predictions from one mechanism.

**Test:** retrain at 224px. Compute rises ~3× (224²/128² = 3.06), so roughly
2.5 hours — affordable given the corrected compute budget.

Either outcome is reportable:

- Accuracy jumps toward 90%+ → confirms high-frequency artefacts carry the signal, ties directly to the frequency-domain literature, and strengthens the interpretation of the robustness results.
- Accuracy stays near 79% → resolution is not the constraint; capacity (`unfreeze_blocks`) or training-set size is. Test those next.

### Confound to fix if resolution changes

The baselines extract HOG/LBP at 128px and pixels at 32px, independent of
`subset.image_size`. If the CNN moves to 224px, the comparison table above
becomes resolution-unfair. **Re-run the baselines at matched resolution before
the comparison goes in the report**, or state the difference explicitly.

---

## Chapter 6 — Baseline results

Working notes from `src/baselines.py`. Run 7 August 2026.
Train subsample 10,000 (Dataset A). Test 4,000 (A, in-domain). Cross-test 6,000 (B, unseen).

---

## 6.1 Results

| Model | Evaluated on | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (32×32 pixels) | A (in-domain) | 0.7358 | 0.7363 | 0.7345 | 0.7354 | 0.8121 |
| Logistic Regression | **B (cross-dataset)** | 0.4883 | 0.4513 | 0.1080 | 0.1743 | **0.5063** |
| HOG + LBP → LinearSVC | A (in-domain) | 0.7540 | 0.7550 | 0.7520 | 0.7535 | 0.8314 |
| HOG + LBP → LinearSVC | **B (cross-dataset)** | 0.4895 | 0.2261 | 0.0087 | 0.0167 | **0.3985** |

Generalisation drop A→B: **24.7 points** (LogReg), **26.5 points** (HOG+LBP).

Extraction: pixels ~47s for 20k images; HOG+LBP ~3.6 min. SVM fit 37s.

---

## 6.2 In-domain reading

Both baselines land around **73–75% accuracy, AUC 0.81–0.83**. That is a useful
place to be:

- Well above the 50% chance floor, so the task is learnable from low-level
  features and the pipeline is wired correctly.
- Well below the 95%+ that CNNs report on this dataset in the literature, so
  there is real headroom for the deep models to demonstrate value. A baseline
  that already scored 95% would make the whole project pointless.

HOG+LBP beats raw pixels by only ~2 points (75.4% vs 73.6%). Gradient and
texture descriptors add surprisingly little over downscaled greyscale. Worth a
sentence in the evaluation — it suggests the in-domain signal is coarse and
fairly global rather than fine-grained texture.

---

## 6.3 Cross-dataset collapse — the main result

**Both baselines fall to chance on Dataset B.** 48.8% and 49.0% on a balanced
set is indistinguishable from guessing.

Two details make this more than a simple failure:

**(a) HOG+LBP scores AUC 0.3985 — below chance.**
An AUC meaningfully under 0.50 means the ranking is *systematically inverted*:
images the model scores as more-fake are, on average, more likely to be real.
At n=6,000 (3,000 per class) the standard error on AUC is roughly 0.0075, so
0.3985 sits about 13 standard errors below 0.50. This is not noise.

**(b) Recall collapses to near zero** — 0.108 for LogReg, **0.0087** for
HOG+LBP. The models classify almost every Dataset B image as *real*. Accuracy
lands near 49% simply because the set is balanced and they are answering "real"
to everything.

**(c) The more texture-dependent model degrades further.** LogReg on raw pixels
lands at chance (0.5063). HOG+LBP, which explicitly encodes gradients and
texture, goes *below* chance (0.3985). More reliance on texture, worse
cross-domain behaviour — a dose-response pattern consistent with the shortcut
hypothesis.

---

## 6.4a RESOLVED — the two baselines fail differently

Testing both threshold polarities settles the open question. **They do not fail
the same way**, and the difference is the most informative result so far.

| Model | Best acc., normal rule | Best acc., inverted rule | Verdict |
|---|---|---|---|
| Logistic Regression (pixels) | 0.5180 | 0.5130 | **Representation failure** — nothing transfers |
| HOG + LBP → SVC | 0.4993 | **0.5790** | **Signal survives, sign reversed** |

**Raw pixels carry nothing.** Neither threshold direction beats 0.518. Whatever
the pixel classifier learned about Dataset A is simply absent in Dataset B.

**HOG+LBP retains real signal with the sign flipped.** Inverting the decision
rule recovers 57.9% — modest, but at n=6,000 the standard error on accuracy is
about 0.0064, so 0.579 sits roughly 12 standard errors above chance. It is
small but not noise.

This is the substantive finding: **gradient and texture descriptors capture
something about facial forgery that does transfer across generation methods —
but the direction of the relationship reverses.** It corroborates the sharpness
check in §6.4 directly: Dataset A reals are sharper than fakes (AUC 0.619),
Dataset B fakes are slightly sharper than reals (AUC 0.480). Same feature
family, opposite sign.

A plausible reading, offered as interpretation rather than established fact:
Dataset A's fakes are whole StyleGAN renders, which are smoother than real
photographs. Dataset B's fakes are face-swapped regions blended into genuine
photographs, where the blending and re-sharpening of the manipulated region
introduces *additional* local high-frequency structure. Under that account the
sign reversal is a property of the manipulation type, not an artefact.

### Correction to the earlier §6.4 reasoning

The "global sharpness shortcut" explanation drafted before this diagnostic was
wrong in mechanism and should not be used. HOG applies L2-Hys block
normalisation, which largely removes global contrast and illumination scaling —
so the classifier is not keyed on whole-image sharpness at all. It responds to
*local* gradient structure. The global sharpness statistic was a useful
predictor of *that a gap existed*, but it is not the feature the model uses,
and the report should not claim otherwise.

Two separate things are going wrong at once, and they should be reported
separately:

1. **Sign reversal** — the feature/label relationship inverts between datasets (evidenced above).
2. **Boundary misplacement** — 98.1% of Dataset B falls on one side of the learned threshold, independent of the sign problem.

---

## 6.4 Mechanism — original reasoning (superseded by 6.4a, retained for the process narrative)

**Confirmed.** Chapter 5 predicted the collapse from the sharpness domain gap
(A vs B AUC 0.822; B averages roughly half the gradient variance of A). The
collapse happened, and it is severe. The prediction was recorded before
training, so this is a genuine test.

Direct check of whether the real/fake sharpness relationship survives the
domain change (n=500 per dataset, AUC = P(random real ranks above random fake)):

| | sharpness | high-freq | brightness |
|---|---|---|---|
| Dataset A (test) | 0.619 | 0.567 | 0.548 |
| Dataset B (crosstest) | 0.480 | 0.472 | 0.457 |

Median sharpness: A real 0.00246 / fake 0.00198 — **reals sharper**.
Dataset B real 0.00095 / fake 0.00107 — **fakes very slightly sharper**.

So the cue that works in A is absent in B, and what little signal exists points
the opposite way. A model keyed on it has nothing valid to use.

**Not explained.** The tidy version of this story does not survive scrutiny and
should not be written up as if it did. If the learned rule were simply
"blurrier ⇒ fake", then Dataset B — which is globally about half as sharp —
should be classified overwhelmingly as *fake*, giving recall near 1.0. The
opposite happens: recall is near **zero**, so the boundary maps almost all of
Dataset B to *real*.

Global sharpness alone therefore does not account for the direction of the
failure. The models operate in 1,024 and 1,790 dimensions respectively, and
the interaction between a high-dimensional decision boundary and the domain
shift is not reducible to one summary statistic.

**This is an open question, not a settled finding.** Chapter 7 should say so
plainly rather than assert a mechanism the evidence does not support.

### Decision-boundary diagnostic (run 7 Aug, partial)

Distinguishing two failure modes that look identical in an accuracy table:
*calibration* failure (ordering intact, threshold misplaced — recoverable) vs
*representation* failure (features carry no transferable signal — not
recoverable).

| Model | % of B on one side of threshold | Best acc. at any threshold (forward rule only) |
|---|---|---|
| Logistic Regression | 88.0% | 0.5180 |
| HOG + LBP → SVC | 98.1% | 0.4993 |

Reference points from synthetic controls: a pure calibration failure yields
~0.78; a fully scrambled representation yields ~0.51.

**Caveat — this diagnostic was incomplete on first run.** It only tested the
rule *"score above threshold ⇒ fake"*. With HOG+LBP's AUC at 0.3986 the ranking
is inverted, so the correct rule to test is the reverse one, and testing only
the forward direction understates how much information the representation
retains. Re-running with both polarities. Numbers above stand for the forward
rule; the inverted figure is pending and is the one that matters for HOG+LBP.

### Follow-up checks worth running

- [x] Inspect the decision-function distribution on A vs B — done, see above
- [ ] Re-run diagnostic testing **both** threshold polarities
- [ ] Refit with sharpness normalised per image. If cross-dataset performance improves, that isolates sharpness as causal.
- [ ] Repeat once the CNN is trained. Does a CNN show the same near-zero recall, or does it fail differently?

---

## 6.5 What this establishes for the report

1. A quantified baseline for the CNN to beat: **75.4% / AUC 0.831** in-domain.
2. Direct evidence that **classical low-level features do not generalise across
   deepfake generation methods** — the central claim of the project, demonstrated
   before any deep learning is involved.
3. A concrete prediction to carry into the CNN experiments: if MobileNetV2 also
   collapses toward chance on Dataset B, the problem is the domain gap rather
   than model capacity. If it holds up substantially better, that is a genuine
   finding about representation learning and worth foregrounding.
4. An honest open question about *why* the failure takes the specific form it
   does. Recording this is worth more than a confident but unsupported
   explanation.

---

## 6.6 Note on a warning during the run

skimage emitted:

> `UserWarning: Applying local_binary_pattern to floating-point images may give
> unexpected results when small numerical differences between adjacent pixels
> are present. It is recommended to use this function with images of integer dtype.`

LBP is currently computed on float images in `[0, 1]`. The comparison is
`neighbour >= centre`, so on floats, near-equal pixels can flip on
floating-point noise. Applied identically to both datasets, so it does not
invalidate the comparison, but it adds avoidable noise to the LBP histogram.

**Action:** switch LBP to `uint8` input and re-run. Cheap (features are cached
per-kind, so only the LBP path recomputes) and removes a reviewer objection.
