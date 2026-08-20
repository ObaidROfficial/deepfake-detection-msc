# Front Matter

> **DRAFT v1 — 11 August 2026.** Title page, abstract, acknowledgements and
> abbreviations. Contents page and lists of figures/tables are generated
> automatically during `.docx` assembly and are not drafted here.

---

## Title page

<div align="center">

**AI-Based Deepfake Detection Using Machine Learning
with Robustness Evaluation**

Muhammad Obaid Ur Rehman

stu3000

MSc Advanced Computing Project (COM7014)

Arden University

Supervisor: Tony Stark

Submitted October 2026

</div>

---

## Abstract

*Target 300 words. Currently ~300.*

Machine learning detectors for deepfake imagery routinely report accuracies
above 95% on benchmark datasets, yet there is a persistent gap between such
figures and performance on media encountered outside the laboratory. This
project investigates that gap directly, asking not whether detectors fail
outside their training distribution but how and why they do so.

Four models of increasing sophistication were evaluated under a single fixed
protocol: logistic regression on raw pixels, a HOG and LBP pipeline with a
linear SVM, and two fine-tuned convolutional networks, MobileNetV2 and
EfficientNet-B0, configured to matched trainable capacity. Each was trained on
GAN-generated faces and tested both in-domain and on an unseen dataset produced
by a different manipulation method. All training was performed on CPU-only
consumer hardware.

In-domain accuracy rose from 73.6% to 94.6% across the four models.
Cross-dataset accuracy never left the neighbourhood of chance, and the
generalisation gap widened monotonically from 24.7 to 44.2 percentage points.
A threshold diagnostic testing both decision polarities established that this
is a representation failure rather than a calibration failure: no operating
point recovers usable accuracy. Because the two convolutional networks are
structurally dissimilar yet fail identically, the result is not attributable to
a single architecture choice.

A controlled ablation found input resolution alone worth 10.1 accuracy points,
supporting the account that the discriminative signal lies in high-frequency
detail arising from generator upsampling. That account generated a prediction,
recorded before testing, which the robustness suite confirmed: perturbations
destroying high-frequency information cost up to 38.5 accuracy points against
at most 3.8 for intensity rescaling. Under white-box adversarial attack at an
imperceptible perturbation, accuracy fell to 5.2%.

The detector fails toward classifying images as authentic rather than toward
expressing uncertainty, a failure direction with direct implications for
deployment.

---

## Acknowledgements

I would like to thank my supervisor, Tony Stark, for guidance throughout this
project.

I am grateful to the maintainers of the datasets used here — NVIDIA for the
Flickr-Faces-HQ dataset, and Le et al. for OpenForensics — and to the
maintainers of PyTorch, scikit-learn and scikit-image, without which a
project of this scope would not have been feasible on the hardware available.

---

## Abbreviations

| | |
|---|---|
| AUC | Area Under the Curve |
| BCS | British Computer Society |
| CNN | Convolutional Neural Network |
| CPU | Central Processing Unit |
| CUDA | Compute Unified Device Architecture |
| DCT | Discrete Cosine Transform |
| DFDC | DeepFake Detection Challenge |
| EULA | End User Licence Agreement |
| FFHQ | Flickr-Faces-HQ |
| FF++ | FaceForensics++ |
| FGSM | Fast Gradient Sign Method |
| GAN | Generative Adversarial Network |
| GDPR | General Data Protection Regulation |
| GPU | Graphics Processing Unit |
| GUI | Graphical User Interface |
| HOG | Histogram of Oriented Gradients |
| JPEG | Joint Photographic Experts Group |
| LBP | Local Binary Patterns |
| PGD | Projected Gradient Descent |
| PR | Precision–Recall |
| RAM | Random Access Memory |
| RGB | Red, Green, Blue |
| ROC | Receiver Operating Characteristic |
| RQ | Research Question |
| SVM | Support Vector Machine |

---

## Note on the abstract

Written last, per the sequence recommended in Lesson 5. It deliberately leads
with the negative result rather than the in-domain accuracy, because the
cross-dataset collapse is the finding and the 94.6% figure is context for it.
An abstract foregrounding "94.6% accuracy achieved" would misrepresent what
this project actually establishes.
