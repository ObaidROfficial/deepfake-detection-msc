# Chapter 7 — Robustness findings (RQ4)

MobileNetV2 @224px, converged. Perturbations applied to the 4,000-image
in-domain test set. Clean baseline accuracy **0.9430**. Run 10 Aug, 33.7 min.

---

## 7.1 Results

| Family | Setting | Accuracy | F1 | ROC-AUC | Drop |
|---|---|---|---|---|---|
| — | **clean** | **0.9430** | 0.9428 | 0.9869 | — |
| JPEG | q=90 | 0.9310 | 0.9333 | 0.9837 | 0.012 |
| JPEG | q=70 | 0.9113 | 0.9157 | 0.9770 | 0.032 |
| JPEG | q=50 | 0.8952 | 0.9019 | 0.9692 | 0.048 |
| JPEG | q=30 | 0.8650 | 0.8748 | 0.9512 | 0.078 |
| JPEG | **q=10** | 0.7288 | 0.7213 | 0.8094 | **0.214** |
| Noise | σ=0.01 | 0.9267 | 0.9289 | 0.9814 | 0.016 |
| Noise | σ=0.05 | 0.8125 | 0.8287 | 0.9126 | 0.131 |
| Noise | **σ=0.10** | 0.6797 | 0.7059 | 0.7552 | **0.263** |
| Blur | k=3 | 0.8880 | 0.8946 | 0.9631 | 0.055 |
| Blur | k=5 | 0.8518 | 0.8553 | 0.9286 | 0.091 |
| Blur | **k=7** | 0.7943 | 0.7823 | 0.8813 | **0.149** |
| Downscale | 64px | 0.7977 | 0.7897 | 0.8860 | 0.145 |
| Downscale | **32px** | 0.5580 | 0.2432 | 0.6963 | **0.385** |
| Brightness | −0.4 | 0.9270 | 0.9260 | 0.9797 | 0.016 |
| Brightness | +0.4 | 0.9105 | 0.9083 | 0.9717 | 0.032 |
| Contrast | −0.4 | 0.9048 | 0.8992 | 0.9748 | 0.038 |
| Contrast | +0.4 | 0.9230 | 0.9213 | 0.9790 | 0.020 |

### Worst case per family

| Family | Worst accuracy | Drop |
|---|---|---|
| Downscale | 0.5580 | 0.385 |
| Noise | 0.6797 | 0.263 |
| JPEG | 0.7288 | 0.214 |
| Blur | 0.7943 | 0.149 |
| Contrast | 0.9048 | 0.038 |
| Brightness | 0.9105 | 0.032 |

---

## 7.2 Prediction confirmed

Recorded in the module docstring **before** the suite was run: perturbations
that destroy high-frequency information irreversibly should hurt substantially
more than those that merely rescale intensities.

```
destructive (jpeg, blur, downscale) : mean worst-case accuracy 0.6937
rescaling   (brightness, contrast)  : mean worst-case accuracy 0.9076
                                      gap 0.214
```

**Supported, and by a wide margin.** Brightness and contrast shifts of ±40%
cost at most 3.8 accuracy points. Downscaling to 32px costs 38.5.

This is the second independent confirmation of the same mechanism. The
128→224px ablation showed resolution was worth +10.1 accuracy points; the
robustness suite now shows that removing high-frequency content is what breaks
the detector. **One explanation, two predictions, both borne out.**

### Noise fits the account too

Gaussian noise is *additive*, not destructive, yet σ=0.10 costs 26.3 points —
more than JPEG q=10. This is consistent rather than contradictory: noise does
not remove the fine detail, it swamps it. The unifying statement is that the
detector depends on the high-frequency band, and it fails whether that band is
**removed** (JPEG, blur, downscale) or **buried** (noise). Only intensity
rescaling, which preserves the relative spatial pattern of gradients, is
tolerated.

### An echo of the cross-dataset failure

At downscale 32px, accuracy is 0.5580 but F1 collapses to **0.2432** — the
model stops predicting "fake" almost entirely. That is the same behaviour seen
on Dataset B (recall 0.083, predicted-fake 7.2%). When the high-frequency
evidence disappears, this detector does not become uncertain; it defaults to
"real". Worth noting: in a deployment context, a detector that silently fails
*open* under compression is worse than one that flags low confidence.

---

## 7.3 Practical implication

The operating range matters more than the headline number:

- JPEG q≥50 (typical social-media quality): accuracy stays ≥ 0.895. Usable.
- JPEG q=10, or resizing to thumbnail scale: accuracy 0.729 and 0.558. Not usable.

A detector reporting 94.3% on pristine images would deliver roughly **56%** on
a thumbnail-resized image — barely better than guessing. Benchmark accuracy on
clean data materially overstates deployed performance.

---

## 7.4 Adversarial robustness (FGSM)

### Both label regimes

| ε | Predicted-label (realistic) | | Ground-truth label (label leaking) | |
|---|---|---|---|---|
| | Accuracy | AUC | Accuracy | AUC |
| 0.005 | 0.0520 | 0.0747 | 0.0000 | 0.0000 |
| 0.01 | 0.0520 | 0.0696 | 0.0000 | 0.0000 |
| 0.03 | 0.0820 | 0.0620 | 0.0300 | 0.0030 |

**Report the predicted-label column.** An attacker probing a deployed detector
observes its output, not the ground truth. Computing the gradient against the
true label hands the attack the answer and mechanically guarantees total
inversion on a binary task — the well-known *label-leaking* effect.

### Result

Clean accuracy **0.9430** → **0.0520** under FGSM at ε=0.005, a perturbation
small enough to be visually imperceptible. The detector is not merely degraded,
it is **inverted**: ROC-AUC of 0.0747 means the ranking is almost perfectly
backwards.

### Why the floor sits at ~5% rather than 0%

This is not noise, and the explanation is worth stating because it confirms the
attack is behaving correctly. Clean accuracy is 94.3%, so roughly **5.7%** of
samples were *already misclassified* before any attack. FGSM steps away from
the model's predicted label — for those already-wrong samples, stepping away
from the wrong prediction moves them **toward** the correct one. The 5.2%
residual accuracy is almost exactly the population that was wrong to begin
with.

### The attack saturates immediately

Accuracy is identical at ε=0.005 and ε=0.01 (both 0.0520), and marginally
*higher* at ε=0.03 (0.0820). There is no gradual degradation curve. The
smallest perturbation tested already flips essentially every confident
prediction; larger perturbations add nothing and may distort the input enough
that a few predictions revert.

### Honest framing of the correction

The bug changed the **number**, not the conclusion. Both regimes show
catastrophic failure; the corrected figure is 5.2% rather than 0.0%. It would
overstate the significance of the fix to imply the original finding was wrong —
it was right for the wrong reason, and now it is right for the right one, with
a defensible number attached.

### Threat model — important caveat

FGSM is a **white-box** attack requiring full access to the model architecture
and gradients. That is a strong assumption. This result does not show the
detector is useless in practice; it shows it is unsuitable in any setting where
an adversary can obtain model access. Black-box transfer attacks are
substantially harder and are not tested here. Chapter 7 should state this
scope limit explicitly rather than implying general defeat.

---

## 7.5 Open items
- [ ] Repeat the suite on Dataset B — does an already-failing model degrade further, or is it already at the floor?
- [ ] Compare against EfficientNet-B0 once trained; is the robustness profile architecture-specific?
- [ ] Consider whether augmentation during training (JPEG/noise) would improve robustness — a natural "further work" item, testable if time allows
