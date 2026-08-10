# Chapter 3: Requirements and Methodology

> **DRAFT v1 — 11 August 2026.** Target 1,600 words.

---

## 3.1 Research methodology

This project uses a secondary research approach combined with experimental
software development. No primary data was collected and no human participants
were involved.

The literature review followed a structured search of IEEE Xplore, the ACM
Digital Library, ScienceDirect and Google Scholar, supplemented by arXiv for
recent preprints. Searches were restricted mainly to 2019 onwards because
generator technology has changed substantially and older detection results
often describe systems that no longer represent the threat.

The experimental component follows a comparative design. Rather than proposing
a novel architecture, the project evaluates four models of increasing sophistication
under a single fixed protocol, so that differences in
performance can be attributed to the models rather than to differences in how
they were tested. Two evaluation conditions are applied to every model: an
in-domain test set drawn from the same dataset as training, and a cross-dataset
test set built from a different manipulation method entirely.

This design was chosen because the research questions are comparative. RQ3 asks
how the proposed model compares to baselines, and RQ5 asks about the
limitations of current techniques. Neither can be answered by training a single
model and reporting its accuracy.

## 3.2 Development methodology

An Agile, iterative approach was adopted rather than a Waterfall sequence.
Dawson (2015) characterises the choice as depending on how well requirements
can be specified at the outset, and that framing is what settles it here.

The project's later stages were genuinely dependent on findings from earlier
ones in ways that could not have been specified in advance. The decision to run a resolution ablation, for example, came from
noticing that a converged model had plateaued below expectation; the robustness
suite's central prediction came from the result of that ablation. A Waterfall
plan fixed at proposal stage would have specified neither.

Work proceeded in short cycles: implement a component, run it, examine the
output, and let the result determine the next task. Each cycle produced
something runnable, which meant errors surfaced quickly. Chapter 5 documents
several cases where an anomalous result led directly to finding a defect.

The trade-off is that an iterative approach makes scheduling less predictable,
and this did cost time — three separate training runs were needed before the
model was properly converged. A more rigid plan would have been easier to
track but would have produced a weaker result, because the questions worth
asking were not visible at the outset.

## 3.3 Deviation from the approved proposal

The approved proposal specified FaceForensics++ and the Kaggle Deepfake
Detection Challenge dataset. Neither was used. This section documents why, as
the change materially affects how the results should be interpreted.

**FaceForensics++** requires a signed End User Licence Agreement submitted to
the hosting institution and manually approved before download. Turnaround is
typically one to two weeks with no guarantee of falling inside the project
window. It is also distributed as raw video, requiring frame extraction and
face cropping before use — an expensive preprocessing stage without GPU
acceleration.

**The DFDC corpus** is approximately 470GB of video. This exceeded available
storage and bandwidth, and decoding that volume on a CPU-only machine was not
achievable in the time available.

**The binding constraint** was hardware. The project machine has twelve CPU
cores, 16GB of RAM and integrated graphics with no CUDA support. This is stated
plainly rather than minimised, because it drove several subsequent decisions.

Replacement datasets were selected against four criteria: pre-extracted face
crops so no video decoding is required; open access with no licence delay; two
genuinely distinct generation methods so the cross-dataset comparison is
meaningful; and sufficient volume to allow subsampling.

The consequences are acknowledged. FaceForensics++ is the field's standard
benchmark, so results here cannot be placed directly alongside the bulk of
published work. The substitutes also skew toward fully synthetic faces and
in-the-wild face swaps rather than the range of manipulation types FF++
covers. Both points are revisited in Chapter 7.

The supervisor was notified of this change before it was implemented.

## 3.4 Requirements

### Functional requirements

| ID | Requirement | Met |
|---|---|---|
| F1 | Download and prepare datasets reproducibly from a documented source | Yes |
| F2 | Produce class-balanced, disjoint train/validation/test splits from a fixed seed | Yes |
| F3 | Implement at least one classical baseline for comparison | Yes — two implemented |
| F4 | Implement a CNN classifier using transfer learning | Yes |
| F5 | Report accuracy, precision, recall, F1 and ROC-AUC | Yes |
| F6 | Evaluate every model on an unseen dataset without retraining | Yes |
| F7 | Apply a systematic input-degradation suite to the trained model | Yes — 22 settings |
| F8 | Implement an adversarial attack | Yes — FGSM, two label regimes |
| F9 | Save trained weights for reuse without retraining | Yes |
| F10 | Provide a means of classifying an arbitrary user-supplied image | Yes — command line tool and graphical interface |
| F11 | Communicate the model's operating limits to a non-expert user | Yes — input checks and a standing scope notice |
| F12 | Be installable and runnable by a third party without the datasets | Yes — one-command setup, weights published separately |

Requirements F10 to F12 were not in the approved proposal, which specified only
"a runnable script or notebook". They were added after an intermediate review,
on the reasoning that a set of experiment scripts demonstrates that a model was
*measured* but not that it can be *used*, and that the distinction matters for
an artefact assessed on functionality.

F11 deserves particular comment because it emerged from the findings rather
than preceding them. Once the evaluation established that the detector fails
badly outside its training distribution, and fails toward "real" rather than
toward uncertainty, a tool that reported only a verdict and a confidence figure
would have been actively misleading. The requirement to surface the operating
limits is therefore a consequence of Chapter 6, not an interface preference.

### Non-functional requirements

| ID | Requirement | Met |
|---|---|---|
| N1 | Run end to end on CPU-only hardware with no GPU dependency | Yes |
| N2 | Any single training run to complete within one working day | Yes — longest 3.2 hours |
| N3 | Leave the machine usable during training | Yes — threads capped at 8 of 12 |
| N4 | Resume from the last completed epoch after interruption | Yes — per-epoch checkpointing |
| N5 | All results regenerable from a clean clone | Yes — seeded, manifests committed |
| N6 | All hyperparameters in one place, not scattered through code | Yes — `config.yaml` |
| N7 | No credentials or dataset content committed to version control | Yes — enforced by `.gitignore` |
| N8 | A third party should reach a working detector in one command | Yes — `setup.ps1` / `setup.sh` |
| N9 | The interface must not require the datasets or a Kaggle account | Yes — weights distributed via GitHub Releases |

N3 deserves comment. It sounds like a convenience requirement but it had a real
effect: an early benchmark run while a 3.75GB download was in progress produced
throughput 60% lower than a clean run. Since the machine was in use for other
work throughout, capping thread count kept both the training and the
measurements stable.

## 3.5 Ethical, legal and professional considerations

**Ethics approval.** The project was registered on the Arden Ethics Portal and
approval was granted before any work began, in accordance with the Arden
University Ethics Policy (Arden University, 2023).

**Applicability of participant procedures.** The Ethics Policy sets out
requirements for participant information sheets, informed consent, debriefing
and company sign-off. None of these apply here. The project uses only
pre-existing, publicly released datasets; there are no participants to inform,
consent, or debrief, and no host organisation to obtain sign-off from. The
corresponding forms were therefore not completed, and this is a considered
determination rather than an omission.

**Data protection.** No personal data as defined by the UK GDPR was collected.
The datasets do contain images of identifiable individuals, which is
unavoidable in facial forensics research. Three mitigations apply: the datasets
are distributed by their creators for research use under their own licence
terms; no images are redistributed by this project, with the repository
containing download scripts rather than image data; and no attempt was made to
identify, contact or link any depicted individual.

**Licensing.** Dataset A's real images derive from NVIDIA's Flickr-Faces-HQ,
licensed CC BY-NC-SA 4.0, restricting use to non-commercial purposes. This
project is academic and non-commercial, so the terms are satisfied. Dataset B
derives from OpenForensics, released for research purposes. Both licences are
recorded in the repository, and users must download the data themselves under
their own accounts.

**Professional practice.** The BCS Code of Conduct (BCS, 2022) requires
competence, due regard for the public interest, and professional integrity. Two aspects are
relevant. First, the project reports its own failures and corrections rather
than presenting only favourable results, which Chapter 5 and Chapter 7 document
in detail. Second, the dual-use character of the robustness findings was
considered explicitly before publication; that reasoning is set out in Chapter
7 rather than here, since it is a matter of critical judgement rather than
compliance.

## 3.6 Risk register

| Risk | Likelihood | Impact | Mitigation | Outcome |
|---|---|---|---|---|
| Dataset unavailable or withdrawn | Medium | High | Verify availability before committing; identify fallbacks | Both remained available |
| CPU training too slow to complete | Medium | High | Benchmark before committing to subset size | Benchmark run; overestimated cost by 7× |
| Datasets share a real-image source, invalidating the cross-dataset design | Low | Critical | Verify by perceptual hashing and alignment analysis before training | Verified independent |
| Training interrupted by machine use or shutdown | High | Medium | Checkpoint every epoch, support `--resume` | Not needed but available |
| Cross-dataset accuracy collapses to chance | High | Low | Treat as a finding and report honestly | Occurred; became the central result |
| Insufficient memory for feature matrices | Medium | Medium | Calculate requirements in advance; subsample training set | 489MB peak, within limits |
| Word count overrun | Medium | Low | Track per chapter during drafting | Tracked |

The third row is worth highlighting. The risk that both datasets drew their
real images from the same source would have invalidated the entire
cross-dataset argument, and it was cheap to test. Verification was performed
before any CNN training, on the reasoning that it made no sense to spend hours
of compute on a foundation that might not hold.

The fifth row records a risk that was accepted rather than mitigated. A
collapse to chance was considered a likely outcome from the outset, and the
plan was always to report it rather than tune until it disappeared.

---

*Draft word count: approximately 1,600.*
