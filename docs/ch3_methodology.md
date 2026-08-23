# Chapter 3: Requirements and Methodology

> **DRAFT v1, 11 August 2026.** Target 1,600 words.

---

## 3.1 Research methodology

In the terms set out by Saunders, Lewis and Thornhill (2023), this project is
positioned within a positivist philosophy and follows a deductive approach. The
literature supplies a mechanism, that detection relies on generator-specific
high-frequency artefacts, from which testable predictions are derived and then
measured. The strategy is experimental, the method quantitative and
mono-method, and the time horizon cross-sectional. Secondary data is used
throughout. No primary data was collected and no human participants were
involved, which is what removes the project from the scope of most of the
ethical review process discussed in section 3.5.

The literature review followed a structured search of IEEE Xplore, the ACM
Digital Library, ScienceDirect and Google Scholar, supplemented by arXiv for
recent preprints. Searches were restricted mainly to 2019 onwards because
generator technology has changed substantially and older detection results
often describe systems that no longer represent the threat.

The experimental component follows a comparative design. Rather than proposing
a novel architecture, it evaluates four models of increasing sophistication
under one fixed protocol, so performance differences can be attributed to the
models rather than to how they were tested. Every model faces two conditions:
an in-domain test set from the same dataset as training, and a cross-dataset
set built from an entirely different manipulation method. This follows from the
research questions being comparative, since neither RQ3 nor RQ5 can be answered
by training one model and reporting its accuracy.

## 3.2 Development methodology

An Agile, iterative approach was adopted rather than a Waterfall sequence.
Dawson (2015) characterises the choice as depending on how well requirements
can be specified at the outset, and that framing is what settles it here.

The project's later stages depended on findings from earlier ones in ways that
could not have been specified in advance. The resolution ablation came from
noticing a converged model had plateaued below expectation, and the robustness
suite's central prediction came from that ablation's result. A Waterfall plan
fixed at proposal stage would have specified neither. Work therefore proceeded
in short cycles. Implement a component, run it, examine the output, let the
result determine the next task. Each cycle produced something runnable, so
errors surfaced quickly.

The approach borrows from Scrum rather than implementing it. Schwaber and
Sutherland (2020) define the framework around a team, and most of its machinery
presupposes more than one person. What did transfer was the Definition of Done,
which the 2020 revision makes a mandatory commitment attached to the increment.
An explicit standard for "finished", meaning runnable, seeded, output written
to `results/` and numbers reconciled against whatever they were meant to
reproduce, stopped partially-working components accumulating.

The trade-off is less predictable scheduling, and this did cost time: three
training runs where one was planned. A more rigid plan would have been easier
to track but would have produced a weaker result, because the questions worth
asking were not visible at the outset.

## 3.3 Deviation from the approved proposal

The approved proposal specified FaceForensics++ and the Kaggle Deepfake
Detection Challenge dataset. Neither was used. This section documents why, as
the change materially affects how the results should be interpreted.

**FaceForensics++** requires a signed End User Licence Agreement submitted to
the hosting institution and manually approved before download, with turnaround
typically one to two weeks and no guarantee of falling inside the project
window. It is also distributed as raw video, requiring frame extraction and
face cropping, which is an expensive preprocessing stage without GPU
acceleration. **The DFDC corpus** is approximately 470GB of video, exceeding
available storage and bandwidth, and decoding that volume on a CPU-only machine
was not achievable in the time.

**The binding constraint** was hardware: twelve CPU cores, 16GB of RAM and
integrated graphics with no CUDA support. This is stated plainly rather than
minimised, because it drove several subsequent decisions.

Replacement datasets were selected against four criteria: pre-extracted face
crops so no video decoding is required; open access with no licence delay; two
genuinely distinct generation methods so the cross-dataset comparison is
meaningful; and sufficient volume to allow subsampling. The consequences are
acknowledged. FF++ is the field's standard benchmark, so results here cannot be
placed directly alongside the bulk of published work, and the substitutes skew
toward fully synthetic faces and in-the-wild face swaps rather than the range of
manipulation types FF++ covers. Chapter 7 revisits both. The supervisor was
notified before the change was implemented.

## 3.4 Requirements

### Functional requirements

| ID | Requirement | Met |
|---|---|---|
| F1 | Download and prepare datasets reproducibly from a documented source | Yes |
| F2 | Produce class-balanced, disjoint train/validation/test splits from a fixed seed | Yes |
| F3 | Implement at least one classical baseline for comparison | Yes, two implemented |
| F4 | Implement a CNN classifier using transfer learning | Yes |
| F5 | Report accuracy, precision, recall, F1 and ROC-AUC | Yes |
| F6 | Evaluate every model on an unseen dataset without retraining | Yes |
| F7 | Apply a systematic input-degradation suite to the trained model | Yes, 22 settings |
| F8 | Implement an adversarial attack | Yes, FGSM in two label regimes |
| F9 | Save trained weights for reuse without retraining | Yes |
| F10 | Provide a means of classifying an arbitrary user-supplied image | Yes, graphical interface plus a command line tool for batches |
| F11 | Communicate the model's operating limits to a non-expert user | Yes, input checks and a standing scope notice |
| F12 | Be installable and runnable by a third party without the datasets | Yes, graphical installer with weights published separately |
| F13 | Be removable without leaving the machine altered | Yes, graphical uninstaller and scripted equivalents |

Requirements F10 to F13 were not in the approved proposal, which specified only
"a runnable script or notebook". They were added after an intermediate review,
on the reasoning that a set of experiment scripts demonstrates that a model was
*measured* but not that it can be *used*, and that the distinction matters for
an artefact assessed on functionality. F13 was added last and is the least
obvious of the four. Asking somebody to evaluate software is also asking them
to let it onto their machine, and being able to reverse that cleanly is part of
what makes the request reasonable.

F11 deserves comment because it emerged from the findings rather than preceding
them. Once the evaluation established that the detector fails badly outside its
training distribution, and fails toward "real" rather than toward uncertainty,
a tool reporting only a verdict and a confidence figure would have been
actively misleading. Surfacing the operating limits is therefore a consequence
of Chapter 6, not an interface preference.

### Non-functional requirements

| ID | Requirement | Met |
|---|---|---|
| N1 | Run end to end on CPU-only hardware with no GPU dependency | Yes |
| N2 | Any single training run to complete within one working day | Yes, longest 3.2 hours |
| N3 | Leave the machine usable during training | Yes, threads capped at 8 of 12 |
| N4 | Resume from the last completed epoch after interruption | Yes, per-epoch checkpointing |
| N5 | All results regenerable from a clean clone | Yes, seeded with manifests committed |
| N6 | All hyperparameters in one place, not scattered through code | Yes, `config.yaml` |
| N7 | No credentials or dataset content committed to version control | Yes, enforced by `.gitignore` |
| N8 | A third party should reach a working detector without using a terminal | Yes, graphical installer, with `setup.ps1` and `setup.sh` for those who prefer a shell |
| N9 | The interface must not require the datasets or a Kaggle account | Yes, weights distributed via GitHub Releases |

N3 sounds like a convenience requirement but had a real effect. An early
benchmark run while a 3.75GB download was in progress produced throughput 60%
lower than a clean run. Since the machine was in use for other work throughout,
capping thread count kept both the training and the measurements stable.

## 3.5 Ethical, legal and professional considerations

**Ethics approval.** The project was registered on the Arden Ethics Portal and
approval was granted before any work began, in accordance with the Arden
University Ethics Policy (Arden University, 2023).

**Applicability of participant procedures.** The Ethics Policy sets out
requirements for participant information sheets, informed consent, debriefing
and company sign-off. None apply here. The project uses only pre-existing,
publicly released datasets, so there are no participants to inform, consent or
debrief and no host organisation to obtain sign-off from. The corresponding
forms were therefore not completed, which was a considered determination rather
than an omission.

**Data protection.** No personal data as defined by the UK GDPR was collected.
The datasets do contain images of identifiable individuals, which is
unavoidable in facial forensics research, and three mitigations apply. They are
distributed by their creators for research use under their own licence terms.
No images are redistributed here, the repository containing download scripts
rather than image data. And no attempt was made to identify, contact or link
any depicted individual.

**Licensing.** Dataset A's real images derive from NVIDIA's Flickr-Faces-HQ,
licensed CC BY-NC-SA 4.0 and restricted to non-commercial use, which this
academic project satisfies. Dataset B derives from OpenForensics, released for
research purposes. Both licences are recorded in the repository and users must
download the data themselves under their own accounts.

**Professional practice.** The BCS Code of Conduct (BCS, 2022) requires
competence, due regard for the public interest and professional integrity. Two
aspects are relevant. The project reports its own failures and corrections
rather than only favourable results, documented in Chapters 5 and 7. And the
dual-use character of the robustness findings was considered explicitly before
publication, with that reasoning set out in Chapter 7 since it is a matter of
critical judgement rather than compliance.

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

Two rows are worth highlighting. The third, both datasets drawing real images
from the same source, would have invalidated the entire cross-dataset argument,
and it was cheap to test, so verification was performed before any CNN training
rather than spending hours of compute on a foundation that might not hold. The
fifth records a risk accepted rather than mitigated. A collapse to chance was
considered likely from the outset, and the plan was always to report it rather
than tune until it disappeared.

---

*Draft word count: approximately 1,600.*
