# Chapter 6: Testing, Results and Evaluation

> **DRAFT v1 — 11 August 2026.** Target 2,000 words. Report prose, drawn from
> `ch5_findings.md`, `ch6_baseline_findings.md` and `ch7_robustness_findings.md`.
> Figure references correspond to files in `results/figures/`.

---

## 6.1 Test plan

Three questions were tested, each with a separate protocol.

The first is straightforward performance: how accurately does each model
classify held-out images drawn from the same dataset it was trained on? This
uses the 4,000-image test split of Dataset A, which was never seen during
training or validation.

The second is generalisation: how does each model perform on a dataset produced
by a different manipulation method? This uses the 6,000-image cross-test split
of Dataset B (OpenForensics), which no model saw at any point.

The third is robustness: how does the best model behave when its inputs are
degraded in ways that happen routinely to images shared online, and when they
are attacked deliberately?

Metrics reported are accuracy, precision, recall, F1 and ROC-AUC. Because both
test sets are exactly class-balanced by construction, accuracy is meaningful
here and the 50% chance level is a genuine floor. ROC-AUC is reported alongside
because it is threshold-independent, which turns out to matter considerably in
the cross-dataset results.

## 6.2 In-domain results

Table 6.1 gives performance on the Dataset A test split. All models were
evaluated at matched input resolution so the comparison is fair.

**Table 6.1 — In-domain performance (Dataset A, n = 4,000)**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (pixels) | 0.7358 | 0.7363 | 0.7345 | 0.7354 | 0.8121 |
| HOG + LBP → Linear SVC | 0.7860 | 0.7889 | 0.7810 | 0.7849 | 0.8706 |
| MobileNetV2 | 0.9430 | 0.9461 | 0.9395 | 0.9428 | 0.9869 |
| EfficientNet-B0 | **0.9457** | 0.9148 | 0.9830 | **0.9477** | **0.9910** |

The ordering is as expected and the gaps are meaningful. Logistic regression on
flattened 32×32 greyscale pixels reaches 73.6%, which is well above chance and
confirms the task is learnable from very coarse information. The classical
computer vision pipeline adds around five points. Both convolutional networks
add a further sixteen.

Accuracies of 94.3% and 94.6% at AUCs of 0.987 and 0.991 are competent results,
roughly three points below the figures typically reported for this dataset. That
is a reasonable position given the constraints described in Chapter 5 — a
subsample of 20,000 of the 100,000 available training images, CPU-only training,
and only the trailing feature blocks unfrozen. The point worth making is that
the generalisation results which follow come from models that work, not from
weak ones. A poor detector failing to generalise would prove very little.

The two networks differ in how they distribute their errors. EfficientNet-B0
has markedly higher recall (0.9830 against 0.9395) and lower precision (0.9148
against 0.9461), predicting "fake" for 53.7% of a balanced set against
MobileNetV2's 49.6%. It is the more suspicious of the two, catching more
forgeries at the cost of more false accusations.

### Architecture comparison and matched capacity

The two networks were configured to have comparable trainable capacity rather
than the same nominal setting. MobileNetV2's feature extractor is a sequence of
19 small blocks; EfficientNet-B0's is 9 much larger stages, so unfreezing the
same number would have released 16% of one network against roughly 79% of the
other, and any difference in results would partly reflect adaptation capacity
rather than architecture. Unfreezing the last 3 blocks of MobileNetV2 and the
last 2 stages of EfficientNet-B0 gives **1,208,642** and **1,131,954**
trainable parameters respectively — within 6.4% of each other.

On that basis EfficientNet-B0's advantage is marginal: 0.3 accuracy points and
0.004 AUC. It took **4.44 hours** to train against MobileNetV2's **2.40**, and
achieved a *worse* best validation loss (0.1469 against 0.1390). For this task,
at this scale, on this hardware, the more sophisticated architecture cost 85%
more compute for no meaningful gain.

### The effect of input resolution

An early version of the model was trained at 128×128 pixels and reached only
78.7% accuracy. Suspecting that downscaling was discarding the evidence rather
than the model lacking capacity, the experiment was repeated at 224×224 with
every other setting held constant.

**Table 6.2 — Effect of input resolution (MobileNetV2)**

| | 128px | 224px | Change |
|---|---|---|---|
| In-domain accuracy | 0.7873 | 0.8880 | +0.1007 |
| In-domain ROC-AUC | 0.8715 | 0.9600 | +0.0885 |

Resolution alone was worth ten accuracy points. This is consistent with Frank
et al. (2020), who showed that the artefacts distinguishing GAN-generated
images from real photographs are concentrated in the high-frequency band and
arise from the upsampling operations used in generator architectures. Halving
the resolution removes precisely that information.

This result also generated a prediction about robustness which is tested in
section 6.5, and I note it here because the prediction was recorded before that
suite was run rather than fitted afterwards.

## 6.3 Cross-dataset results

Table 6.3 gives the same models evaluated on Dataset B without any retraining
or adaptation.

**Table 6.3 — Cross-dataset performance (Dataset B, n = 6,000)**

| Model | Accuracy | Recall | F1 | ROC-AUC | Predicted "fake" |
|---|---|---|---|---|---|
| Logistic Regression | 0.4883 | 0.1080 | 0.1743 | 0.5063 | 12.0% |
| HOG + LBP → SVC | 0.4790 | 0.0650 | 0.1109 | 0.4751 | 8.6% |
| MobileNetV2 | 0.5110 | 0.0827 | 0.1446 | 0.5593 | 7.2% |
| EfficientNet-B0 | 0.5038 | 0.0717 | 0.1262 | 0.5285 | 6.8% |

Every model collapses to approximately chance. Setting these against the
in-domain figures produces the central result of this project:

```
                    LogReg   HOG+LBP   MobileNetV2   EfficientNet-B0
in-domain            73.6%     78.6%         94.3%             94.6%
cross-dataset        48.8%     47.9%         51.1%             50.4%
generalisation gap    24.7      30.7          43.2              44.2
```

In-domain accuracy rises by 21 points across the four models. Cross-dataset
accuracy varies by 3 points and never leaves the neighbourhood of chance. The
absolute generalisation gap grows monotonically throughout. Whatever the more
capable models learned was absorbed entirely by the source domain.

**The failure is not architecture-specific.** MobileNetV2 and EfficientNet-B0
are structurally different networks — depthwise separable convolutions against
compound-scaled MBConv blocks — trained at matched capacity to within 6.4%, and
they fail identically. Both reach ~94.5% in-domain and both land within a
point of chance on unseen manipulations. This removes the most obvious
objection to the finding, which is that a single unlucky architecture choice
might explain it.

The recall figures are also revealing. All four models classify the
overwhelming majority of Dataset B as real — EfficientNet-B0 predicts "fake"
for only 6.8% of a set that is 50% fake, and MobileNetV2 for 7.2%. Faced with
unfamiliar manipulations these detectors do not become uncertain; they default
to declaring images authentic. For a deployed system this is the worse of the
two failure directions.

Notably, EfficientNet-B0's in-domain bias runs the *other* way — it predicts
"fake" for 53.7% of the in-domain test set, more often than MobileNetV2. So the
tendency to declare everything real on Dataset B is not a property of the model
being conservative in general; it is specific to the domain shift.

### Distinguishing failure modes

An accuracy figure near chance is ambiguous. It could mean the learned
representation orders real and fake correctly but the decision threshold is
misplaced, which would be recoverable by re-calibration. Or it could mean the
representation carries no usable information at all, which would not be.

To separate these, the best accuracy achievable at *any* threshold was computed
for each model, testing both polarities of the decision rule. Testing both
directions matters because an AUC below 0.5 indicates an inverted ranking, and
searching only the conventional direction would understate the information
retained.

**Table 6.4 — Best achievable accuracy on Dataset B at any threshold**

| Model | Conventional rule | Inverted rule | Best | Interpretation |
|---|---|---|---|---|
| Logistic Regression | 0.5180 | 0.5130 | 0.5180 | No transferable signal |
| HOG + LBP → SVC | 0.5060 | 0.5317 | 0.5317 | Weak signal, reversed sign |
| MobileNetV2 | 0.5453 | 0.4967 | 0.5453 | Negligible signal |
| EfficientNet-B0 | 0.5220 | 0.4997 | 0.5220 | Negligible signal |

No threshold in either direction recovers usable performance for any model.
This is a representation failure rather than a calibration failure, and it
cannot be fixed by adjusting an operating point.

It would nevertheless be an overstatement to claim that nothing transfers. Both
CNNs achieve cross-dataset AUCs above chance — 0.5593 and 0.5285 — which at
this sample size are several standard errors from 0.5 and genuinely non-zero.
Better representations do transfer marginally better than the pixel baseline.
But best achievable accuracies of 54.5% and 52.2% on a balanced problem have no
practical value, so the accurate summary is that improvements in in-domain
accuracy produce statistically detectable but practically useless gains in
transfer, while the absolute gap widens.

It is also worth noting that the *smaller* network transferred slightly better
(AUC 0.5593 against 0.5285) despite being marginally worse in-domain. With two
architectures this is not enough to claim a pattern, but it is at least not
evidence that greater capacity helps generalisation.

The HOG and LBP result is worth a separate note. Its cross-dataset AUC of
0.4751 is *below* chance, meaning its ranking is partially inverted. Analysis of
the underlying image statistics showed why: within Dataset A real images are
sharper than fakes, whereas within Dataset B the relationship runs weakly the
other way. Gradient and texture descriptors capture something real about
forgery in both datasets, but the direction of the relationship depends on the
manipulation method. Fully synthetic GAN faces are smoother than photographs;
face-swapped regions blended into genuine photographs are not.

## 6.4 Model comparison against project objectives

Objective 4 required comparing the developed model against baseline methods.
MobileNetV2 outperforms both baselines in-domain by a clear margin (94.3%
against 78.6% and 73.6%). On unseen manipulations it does not meaningfully
outperform either. The honest answer to research question 3 is therefore
conditional: the proposed model is substantially better within its training
distribution and no better outside it.

## 6.5 Robustness results

The robustness suite applied twenty-two perturbation settings to the Dataset A
test split and re-scored the trained MobileNetV2 model on each. Clean baseline
accuracy is 0.9430.

**Table 6.5 — Worst-case accuracy by perturbation family**

| Family | Strongest setting | Accuracy | Drop |
|---|---|---|---|
| Downscale | 32px round-trip | 0.5580 | 0.385 |
| Gaussian noise | σ = 0.10 | 0.6797 | 0.263 |
| JPEG compression | quality 10 | 0.7288 | 0.214 |
| Gaussian blur | k = 7 | 0.7943 | 0.149 |
| Contrast | ±40% | 0.9048 | 0.038 |
| Brightness | ±40% | 0.9105 | 0.032 |

The prediction recorded in section 6.2 was that perturbations destroying
high-frequency information should damage the detector far more than those
merely rescaling intensities. Grouping the families accordingly:

- Destructive (JPEG, blur, downscale): mean worst-case accuracy **0.6937**
- Rescaling (brightness, contrast): mean worst-case accuracy **0.9076**

The gap is 21 points and the prediction is supported. A ±40% change in
brightness or contrast — a large and clearly visible change — costs at most
3.8 accuracy points. Downscaling to 32 pixels and back costs 38.5.

Gaussian noise fits the same account despite being additive rather than
destructive. It does not remove fine detail, it buries it, and the effect on
the detector is comparable. The unifying statement is that this detector
depends on the high-frequency band and fails whether that band is removed or
masked, while tolerating changes that preserve the spatial pattern of
gradients.

This is the second independent confirmation of a single mechanism. The
resolution ablation in section 6.2 and the robustness profile here were
predicted by the same explanation and both came out as expected.

At the strongest downscaling setting, F1 falls to 0.2432 while accuracy is
0.5580 — the model again stops predicting "fake" almost entirely, mirroring
its behaviour on Dataset B. The same failure signature appears whenever the
high-frequency evidence is unavailable, whatever the cause.

### Practical operating range

At JPEG quality 50 and above, which covers typical social media
re-encoding, accuracy remains at or above 0.895. Below that it degrades
quickly. A detector reporting 94.3% on pristine images would deliver
approximately 56% on a thumbnail-resized image. Benchmark accuracy on clean
data materially overstates deployed performance, and this gap should be
reported alongside headline figures rather than omitted.

## 6.6 Adversarial results

The Fast Gradient Sign Method was applied as a white-box attack on a
500-image subset.

**Table 6.6 — FGSM results**

| ε | Accuracy | ROC-AUC |
|---|---|---|
| 0.005 | 0.0520 | 0.0747 |
| 0.01 | 0.0520 | 0.0696 |
| 0.03 | 0.0820 | 0.0620 |

Accuracy falls from 0.9430 to 0.0520 at a perturbation small enough to be
visually imperceptible. The detector is not merely degraded but inverted: an
AUC of 0.0747 means its ranking is close to exactly backwards.

The residual 5.2% has a specific explanation. Clean accuracy is 94.3%, so
approximately 5.7% of samples were already misclassified. The attack steps away
from the model's own predicted label, so for samples that were already wrong it
moves them toward the correct answer. The residual accuracy is essentially the
population that was incorrect to begin with, which confirms the attack is
behaving as intended.

The attack saturates immediately. Accuracy at ε = 0.005 and ε = 0.01 is
identical and there is no gradual degradation curve; the smallest perturbation
tested already flips every confident prediction.

An important methodological point applies here. An initial implementation
computed the gradient with respect to the ground-truth label rather than the
model's prediction, producing accuracy of exactly 0.0000. On a binary task this
mechanically guarantees inversion regardless of the model's actual robustness —
the attack is given the answer. This is the known label-leaking effect. The
figures above use the model's predicted label, which is the realistic threat
model since an attacker probing a deployed system observes its output rather
than the truth. The correction changed the number from 0.0% to 5.2% but not the
conclusion.

Finally, the scope of this result should be stated plainly. FGSM is a white-box
attack requiring access to model gradients. It does not demonstrate that the
detector is useless in general; it demonstrates that it is unsuitable wherever
an adversary can obtain model access. Black-box transfer attacks are
considerably harder and were not tested here.

---

*Draft word count: approximately 1,950.*

**Figures to insert:** 6.1 class balance, 6.2 sample grids, 6.3 domain gap,
6.4 training curves, 6.5 confusion matrices, 6.6 ROC/PR curves, 6.7 robustness
degradation panel, 6.8 worst-case bar chart.
