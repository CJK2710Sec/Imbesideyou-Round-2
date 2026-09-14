# Day 3 Report — Process Identity, Segmentation & Dataset B Analysis

## 1. Objective

Day 3 focused on moving from boundary detection to process identity, segment construction, interleaving analysis, and application of the completed pipeline to Dataset B.

The main goals were:

- reconstruct logical process executions from the Dataset A ground truth;
- understand process interleaving and repeated-process behavior;
- build behavioral features for process identity;
- evaluate sequence/context information;
- construct process segments using the learned boundary detector;
- assign process identities to Dataset A segments;
- evaluate local and process-consistency refinements;
- apply the boundary pipeline to Dataset B, which has no ground truth;
- discover behavioral process families in Dataset B;
- prioritize a candidate for automation.

## 2. Dataset A Process Structure

The Dataset A ground-truth manifest contains:

- 2,009 process executions;
- 15 process families;
- 576 repeated-process cases;
- 183 A-B-A interleaving patterns;
- 63 cross-chunk continuations.

This confirmed that a session is not equivalent to a single business process. Processes can be interrupted, resumed, repeated, and interleaved within the same session.

## 3. Process Identity Experiments

### 3.1 Aggregate behavioral features

An initial identity classifier used behavioral features such as duration, application entropy, unique applications, event counts, keyboard activity, clipboard activity, clicks, screenshots, and timing density.

Results showed that behavioral information contains useful process-identity signal, but aggregate statistics alone were insufficient.

### 3.2 Sequence/context features

Sequence features were added using event, application, and layer n-grams together with contextual behavioral features.

The sequence/context Random Forest achieved approximately:

- Accuracy: 0.3420
- Macro-F1: 0.3404

This improved over the aggregate behavioral baseline and showed that the order and local structure of operations provide additional identity information.

### 3.3 Transition-aware experiment

A transition-aware hybrid model was tested, but the corrected experiment showed that adding transition weighting reduced performance.

Therefore the transition hybrid was rejected and was not used in the final Dataset B analysis.

## 4. Dataset A Segment Construction

The learned boundary probability from the completed Day 2 boundary-confirmation pipeline was used with a threshold of 0.70.

For Dataset A:

- GT process-switch boundaries: 1,590
- True positives: 1,163
- False positives: 292
- False negatives: 427
- Precision: 0.7993
- Recall: 0.7314
- F1: 0.7639
- Mean localization error: 0.1286 s
- Median localization error: 0.0888 s
- P90 localization error: 0.2896 s
- 97.59% of matched boundaries were within 0.5 s
- 100% were within 1 s
- Constructed segments: 1,518

Overlap against complete ground-truth execution windows was:

- >=50% overlap: 1,342 / 1,518 = 88.41%
- >=80% overlap: 1,281 / 1,518 = 84.39%

These results support using the learned boundary detector as the segmentation component.

## 5. Dataset A Identity Assignment

The sequence/context identity model was applied to the constructed segments.

The final comparison tested:

1. RF-only
2. RF + local-neighbor evidence
3. RF + process-consistency
4. Combined RF + local + process evidence

The winning method was **RF + process-consistency**, selected by Macro-F1 first and accuracy second.

| Method | Accuracy | Macro-F1 | Coverage |
|---|---:|---:|---:|
| RF-only | 0.7049 | 0.7031 | 0.8846 |
| RF + local-neighbor | 0.6885 | 0.6852 | 0.8846 |
| RF + process-consistency | **0.7049** | **0.7034** | 0.8846 |
| Combined | 0.6841 | 0.6804 | 0.8846 |

The Dataset A identity result should be described as **GT-overlap validation on predicted segments**, not as unbiased out-of-fold performance, because the process-consistency prototypes use Dataset A ground-truth training information.

## 6. Dataset B Application

Dataset B contains:

- 15 sessions;
- 20,477 raw events;
- no ground-truth process labels.

The Dataset A boundary pipeline was applied to Dataset B using the learned boundary probability threshold of 0.70.

Results:

- 1,276 boundary candidates;
- 204 confirmed boundaries;
- 219 constructed segments.

Because Dataset B has different processes/applications and no GT, Dataset A process labels were not transferred to Dataset B. Instead, process families were discovered directly from Dataset B behavioral and sequence information.

## 7. Dataset B Process Discovery

Unsupervised behavioral clustering produced:

- 2 discovered process families;
- silhouette score: 0.3193.

The discovered families were labelled:

- BPROC_01
- BPROC_02

These are **behavioral cluster identifiers**, not claims about the real business-process names.

### BPROC_01

- 199 detected segments
- 15 sessions
- 4 worker proxies
- 10,502.812 seconds total observed duration
- 175.05 minutes total
- 52.78 seconds mean duration
- 31.8 seconds median duration
- 127.5 seconds P90 duration
- duration CV: 1.056
- approximately 102.5 events per segment
- approximately 2.40 unique applications per segment
- priority score: 0.8973
- priority rank: 1

### BPROC_02

- 20 detected segments
- 11 sessions
- 4 worker proxies
- 72.019 seconds total observed duration
- 1.20 minutes total
- 3.60 seconds mean duration
- priority score: 0.0980
- priority rank: 2

BPROC_01 accounts for approximately 99.3% of the observed duration across the discovered families and is therefore the clear first automation candidate.

## 8. Interleaving and Segmentation Caveat

The transition analysis contains:

- BPROC_01 → BPROC_01: 167
- BPROC_01 → BPROC_02: 19
- BPROC_02 → BPROC_01: 17
- BPROC_02 → BPROC_02: 1

Thus 81.9% of detected transitions are BPROC_01 → BPROC_01.

This is important: detected segment counts must **not** be interpreted as business execution counts. The high same-family transition rate indicates that the boundary detector is fragmenting some continuing behavioral activity into multiple segments.

At the session level, 9 of 15 Dataset B sessions were classified as interleaved and 6 were not. A-B-A patterns were observed in several sessions.

Therefore, the current result is best treated as a behavioral/process-discovery result rather than a definitive business-case count.

## 9. Automation Candidate

### Priority candidate: BPROC_01

BPROC_01 is the strongest candidate because it:

- appears across all 15 Dataset B sessions;
- dominates observed process time;
- has a high automation-priority score;
- contains substantial interaction activity;
- frequently involves multiple applications and interaction layers.

However, its duration CV of approximately 1.06 indicates substantial variability. The current evidence therefore does not justify claiming that the entire workflow is deterministic.

### Recommended automation scope

The recommended next step is to identify the **stable, repetitive core** of BPROC_01 and automate that portion first.

Variable branches, ambiguous cases, and exception handling should remain human-controlled until the actual business actions are identified and validated.

## 10. Current Limitations

1. Dataset B has no ground truth, so boundary precision/recall and process-identity accuracy cannot be measured there.
2. The Dataset A-trained boundary detector may behave differently on Dataset B because the datasets contain different processes and applications.
3. BPROC_01/BPROC_02 are unsupervised behavioral labels and do not reveal business meaning by themselves.
4. The silhouette score of 0.3193 indicates only moderate separation between the discovered behavioral families.
5. Same-family transition frequency indicates segmentation fragmentation, so raw segment counts should not be treated as true execution counts.
6. The detailed segment summary contains activity counts but does not by itself expose enough business semantics to name the workflow confidently.

## 11. Day 4 Handoff

Day 4 should focus on **business interpretation of BPROC_01** rather than further tuning the segmentation model.

Planned work:

1. inspect raw Dataset B events belonging to BPROC_01;
2. recover actual applications, window titles, browser actions, form inputs, clipboard operations, and repeated action sequences;
3. reconstruct the stable workflow core;
4. identify variable branches and exception cases;
5. define automation scope;
6. compare feasible RPA/automation approaches;
7. estimate remaining manual work and expected impact;
8. document risks and mitigations;
9. prepare the final automation proposal.
