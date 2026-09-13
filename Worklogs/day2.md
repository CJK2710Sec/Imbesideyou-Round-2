# Day 2 Work Log — Boundary Detection

## Objective

Continue Step 1 by determining whether raw operation logs contain reliable signals for identifying business-process boundaries, and develop a validated boundary-detection approach using Dataset A ground truth.

## What I investigated

### 1. Ground-truth boundary behaviour

I first analyzed all `gt.jsonl` files across Dataset A.

- 63 sessions
- 1,590 `process_switched_out` events used as candidate process boundaries
- 1,819 `process_started`
- 190 `process_resumed`
- 99 `process_suspended`
- 2,009 `task_started`

The implemented consistency checks found no issues in the checked GT pairing/order conditions.

### 2. Boundary-local signal analysis

Raw `events.jsonl` files were mapped around the 1,590 GT boundaries using multiple temporal windows.

The dataset contained:

- 117 raw event files
- 162,768 raw events
- all 1,590 GT boundaries matched to raw event streams

The strongest short-range signals were changes in:

- event-type structure
- event layer
- event-type/layer Jaccard similarity
- pre-boundary activity rate
- event-type entropy

Application and window-title changes were useful secondary signals, while browser-domain features were effectively non-discriminative in this dataset.

A key observation was that genuine boundaries often showed a quieter pre-boundary region followed by a short post-boundary activity burst.

### 3. Boundary vs normal activity

I created a balanced descriptive dataset using:

- 1,590 GT boundary examples
- 1,590 normal-operation examples sufficiently separated from GT boundaries

A session-grouped logistic-regression baseline showed very strong discrimination for short windows:

| Window | ROC-AUC |
|---|---:|
| ±0.5s | 0.9946 |
| ±1s | 0.9904 |
| ±2s | 0.9902 |
| ±3s | 0.9757 |
| ±5s | 0.9682 |
| ±10s | 0.9220 |

This established that boundary evidence is localized rather than broadly distributed over long windows.

### 4. Hard-negative testing

The initial boundary-vs-normal negatives were too easy, so I created hard negatives from activity near, but not equal to, known boundaries.

This exposed a major weakness: ordinary bursts such as typing, shortcuts, mouse activity, screenshots and clipboard events can look like boundaries.

The analysis showed that the first detector was learning:

> sudden change in short-term event structure

rather than:

> sustained change representing a business-process transition.

This was an important failure mode and prevented premature acceptance of the detector.

### 5. Temporal persistence and multi-scale analysis

I then tested whether genuine boundaries maintain transition evidence longer than hard negatives.

Across windows from ±0.5s to ±3s, GT boundaries consistently showed higher fractions of persistent high transition scores than hard negatives.

I also tested multi-scale score combinations. The simple average across all four scales performed best among the tested equal-weight combinations:

- ROC-AUC: 0.9786
- PR-AUC: 0.9424

The result supported using multiple temporal scales rather than relying on one narrow window.

### 6. Candidate detection and boundary confirmation

A continuous transition score was applied across the sessions to generate boundary candidates.

The first candidate detector had high recall but too many false positives:

- GT boundaries: 1,590
- candidates: 8,456
- TP: 1,456
- FP: 7,000
- FN: 134
- Precision: 0.172
- Recall: 0.916
- F1: 0.290

A second confirmation stage combined:

- transition-score evidence
- temporal persistence
- event/layer structure change
- application/window-title changes
- post-boundary stability

The hand-designed confirmation score produced a very small high-confidence population, showing that confidence could be made precise but that fixed thresholds were not sufficient for the whole candidate population.

### 7. Learned boundary confirmation

I trained a second logistic-regression confirmation model using one-to-one GT matching rather than assigning every candidate within a tolerance window as positive.

Corrected training labels:

- positive candidates: 1,456
- negative candidates: 7,000
- GT boundaries: 1,590
- unmatched GT boundaries: 134

Session-grouped cross-validation produced ROC-AUC values around 0.934–0.951 across folds, with overall:

- ROC-AUC: 0.9428
- PR-AUC: 0.7677

At the selected threshold of 0.70, strict one-to-one boundary evaluation produced:

- predictions: 1,455
- TP: 1,166
- FP: 289
- FN: 424
- Precision: 0.8014
- Recall: 0.7333
- F1: 0.7658

Localization was strong:

- mean error: 0.132s
- median error: 0.089s
- P90 error: 0.291s
- 97.34% of matched boundaries were within 0.5s
- 99.74% were within 1s

Session-level validation remained reasonably stable overall, although several sessions performed substantially worse. These sessions were retained as evidence that the approach is not uniformly reliable.

## What did not work

1. Treating every sudden event burst as a process boundary.
2. Relying on application switches alone.
3. Using large temporal windows as the primary signal.
4. Using simple candidate scores without a confirmation stage.
5. Assigning every candidate within ±2s of a GT boundary as positive; this inflated positives and produced invalid evaluation.
6. Assuming a high ROC-AUC on easy negatives represented final segmentation quality.
7. Using a single fixed hand-designed confirmation formula as the final decision rule.

## Key conclusion

Day 2 established a workable boundary-detection foundation, but it also showed that boundary detection alone is not the complete segmentation problem.

The strongest approach identified so far is:

**candidate generation → temporal/multi-scale evidence → learned boundary confirmation → strict one-to-one evaluation**

The current detector is good enough to proceed to process identity and segmentation design, but should not yet be described as production-ready process segmentation.

## Next step

Day 3 should focus on **process identity and segment construction**:

1. determine which signals consistently identify the logical business process;
2. distinguish process identity from application identity;
3. handle interruptions, suspension/resumption and interleaving;
4. build coherent segments from confirmed boundaries;
5. validate both boundary quality and same-process labeling consistency.
