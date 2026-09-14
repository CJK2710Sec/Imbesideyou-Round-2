# Day 3 Work Log — Process Identity, Segmentation & Dataset B Analysis

## Day 3 Goal

Move beyond boundary detection and determine:

- what constitutes a logical process execution;
- how processes interleave and repeat;
- whether behavioral/sequence features can identify processes;
- how to construct usable segments;
- how to apply the pipeline to Dataset B without ground truth;
- which discovered process should be prioritized for automation.

---

## Phase 6.1 — Initial Process Identity Reconstruction

### Approach

Started with a direct reconstruction based on process-switch information.

### Problem discovered

The initial reconstruction produced suspicious results, including no A-B-A patterns.

The approach treated `process_switched_out` too directly as an execution-ending event and keyed active state only by process.

### Decision

Do not treat a session as a linear sequence of independent process executions.

Use the exact Dataset A ground-truth manifest for process identity analysis and explicitly account for repeated/interleaved processes.

---

## Phase 6.2 — Manifest-Based Interleaving Analysis

### Approach

Used `gt_manifest.json` as the authoritative structure for Dataset A executions.

### Result

- 2,009 GT executions
- 15 process families
- 576 repeated-process cases
- 183 A-B-A patterns
- 63 cross-chunk continuations

### Key insight

Processes can be interrupted, resumed, repeated, and interleaved. Dataset chunks are not equivalent to business-process boundaries.

### Decision

Keep process identity separate from raw session/chunk boundaries.

---

## Phase 6.3 — Identity Feature Analysis

### Approach

Built behavioral features from timed executions, including:

- duration;
- application entropy;
- unique applications;
- event count;
- keystrokes;
- clicks;
- clipboard changes;
- screenshots;
- event/activity density;
- application and timing characteristics.

### Result

- 2,009 manifest executions
- 257 missing end timestamps
- 1,752 complete timed executions

### Decision

Do not fabricate missing end timestamps. Use complete timed executions for temporal features while retaining the full manifest population for structural analysis.

---

## Phase 6.4 — Behavioral Identity Classification

### Models tested

1. Behavioral Logistic Regression
2. Behavioral Random Forest
3. App-only Logistic Regression

### Result

Behavioral features were better than application identity alone, but aggregate behavioral features were not sufficient for reliable process classification.

### Decision

Add sequence/context information.

---

## Phase 6.5 — Sequence + Context Identity

### Approach

Added:

- event n-grams;
- application n-grams;
- layer n-grams;
- contextual behavioral features.

Used a Random Forest with:

- 500 trees;
- balanced class weighting;
- `min_samples_leaf=2`;
- session-grouped cross-validation.

### Result

Sequence/context RF:

- Accuracy: 0.3420
- Macro-F1: 0.3404

### Key insight

Local operation order provides additional identity information beyond aggregate statistics.

### Decision

Use sequence/context features as the main identity representation.

---

## Phase 6.6 / 6.7 — Transition-Aware Experiment

### Experiment

Tested adding transition-aware information to the identity model.

### Result

The hybrid did not improve the base model.

Corrected comparison:

- Base RF: Accuracy 0.3415, Macro-F1 0.3367
- Hybrid: Accuracy 0.3261, Macro-F1 0.3179

### Decision

Reject the transition hybrid.

Do not use the deprecated transition weighting in the final 6.11 pipeline.

---

## Phase 6.8 — Dataset A Segment Construction

### Approach

Used the learned Day 2 boundary probability with threshold 0.70.

### Result

- GT boundaries: 1,590
- TP: 1,163
- FP: 292
- FN: 427
- Precision: 0.7993
- Recall: 0.7314
- F1: 0.7639
- Mean localization error: 0.1286 s
- Median: 0.0888 s
- P90: 0.2896 s
- Within 0.5 s: 97.59%
- Within 1 s: 100%
- Constructed segments: 1,518

Overlap:

- >=50%: 88.41%
- >=80%: 84.39%

### Decision

The boundary pipeline is sufficiently useful to proceed to identity assignment and Dataset B analysis.

---

## Phase 6.9 — Identity Assignment

### Approach

Applied the sequence/context Random Forest to constructed Dataset A segments.

### Result

- 1,518 segments loaded
- 1,517 usable segments
- 84 segments with confidence >=0.8
- Mean confidence: 0.4218
- Median confidence: 0.4205
- Mean margin: 0.2614

GT-overlap identity accuracy:

- 0.6572

### Important caveat

This was an in-domain GT-overlap validation, not an unbiased out-of-fold estimate.

---

## Phase 6.10 — Identity Refinement Comparison

### Methods

1. RF-only
2. RF + local-neighbor evidence
3. RF + process-consistency
4. Combined

### Result

| Method | Accuracy | Macro-F1 | Coverage |
|---|---:|---:|---:|
| RF-only | 0.7049 | 0.7031 | 0.8846 |
| RF + local-neighbor | 0.6885 | 0.6852 | 0.8846 |
| RF + process-consistency | 0.7049 | 0.7034 | 0.8846 |
| Combined | 0.6841 | 0.6804 | 0.8846 |

### Decision

Select **RF + process-consistency** as the winning method.

The improvement over RF-only is very small, so the process-consistency component should not be presented as a major performance breakthrough.

---

## Phase 6.11 — Dataset B Analysis

### Dataset

- 15 sessions
- 20,477 raw events
- no GT

### Boundary pipeline

- 1,276 candidates
- 204 confirmed boundaries
- 219 segments

### Important methodological decision

Dataset B has different processes/applications and no ground truth.

Therefore, do **not** transfer Dataset A process identities to Dataset B.

Instead, perform B-only behavioral process discovery.

---

## Dataset B Process Discovery

### Result

Two behavioral families were discovered:

- BPROC_01
- BPROC_02

Silhouette:

- 0.3193

### BPROC_01

- 199 detected segments
- 15 sessions
- 4 worker proxies
- 175.05 minutes total
- 52.78 seconds mean
- 31.8 seconds median
- 127.5 seconds P90
- duration CV ≈ 1.06
- priority score ≈ 0.8973

### BPROC_02

- 20 detected segments
- 11 sessions
- 4 worker proxies
- 1.20 minutes total
- 3.60 seconds mean
- priority score ≈ 0.0980

### Decision

BPROC_01 is the clear first automation candidate.

---

## Dataset B Transition Analysis

Observed:

- BPROC_01 → BPROC_01 = 167
- BPROC_01 → BPROC_02 = 19
- BPROC_02 → BPROC_01 = 17
- BPROC_02 → BPROC_02 = 1

### Key interpretation

81.9% of detected transitions are BPROC_01 → BPROC_01.

This means many detected boundaries are likely fragmenting continuing BPROC_01 activity.

### Decision

Do not equate detected segment count with business execution count.

---

## Dataset B Interleaving

Session-level analysis found:

- 9/15 sessions interleaved;
- 6/15 sessions non-interleaved.

A-B-A patterns were present in multiple sessions.

### Insight

Dataset B workflows are not simply one process per session. Secondary activity can occur around or within the dominant behavioral workflow.

---

## Final Day 3 Decisions

1. Treat sessions/chunks as observation containers, not business-process boundaries.
2. Keep process identity separate from boundary detection.
3. Use sequence/context features for identity.
4. Reject the transition-weighted hybrid.
5. Use the Day 2 boundary probability threshold of 0.70.
6. Do not transfer Dataset A process labels to Dataset B.
7. Use B-only unsupervised process discovery for Dataset B.
8. Treat BPROC_01/BPROC_02 as behavioral labels, not business names.
9. Treat BPROC_01 as the highest-priority automation candidate.
10. Do not define the exact business automation scope until raw BPROC_01 actions are inspected.

---

## Problems / Failures Encountered

### Failure 1 — Initial identity reconstruction

The first process reconstruction produced no A-B-A behavior.

**Cause:** process switches and suspensions were interpreted too simplistically.

**Fix:** use the exact GT manifest and explicitly analyze repeated/interleaved executions.

### Failure 2 — Transition hybrid

The transition-aware identity hybrid initially appeared promising but was not reproducible after correcting the evaluation.

**Fix:** rerun with the exact Phase 6.5 feature/API setup and compare directly against the base RF.

**Outcome:** hybrid performed worse and was removed from the final pipeline.

### Failure 3 — Dataset B confirmation feature mismatch

The Dataset B confirmation feature calculation initially lacked the fields:

- `persistence_points`
- `persistence_sec`

**Fix:** align the persistence output with the Day 2 confirmation schema.

**Outcome:** Phase 6.11 completed successfully.

---

## Day 3 Final Status

**TECHNICAL PIPELINE: COMPLETE**

**BUSINESS INTERPRETATION: intentionally deferred to Day 4**

Day 4 starts with BPROC_01 raw-event inspection and business-action reconstruction.
