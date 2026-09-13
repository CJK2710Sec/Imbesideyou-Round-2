# Day 2 Report — Boundary Detection Analysis

## 1. Objective

The objective of Day 2 was to determine whether raw operation logs contain enough temporal and structural information to identify transitions between business-process executions.

The analysis used Dataset A because it provides ground-truth process events and therefore allows quantitative validation.

---

## 2. Dataset and Ground Truth

Dataset A contains 63 sessions.

The ground-truth analysis identified:

- 2,009 task/process cases
- 1,590 `process_switched_out` events used as candidate process boundaries
- 1,819 `process_started`
- 190 `process_resumed`
- 99 `process_suspended`

The checked GT consistency conditions produced no detected issues.

Raw-event analysis covered:

- 117 `events.jsonl` files
- 162,768 raw events
- 100% matching of the 1,590 candidate GT boundaries to raw event streams

---

## 3. Boundary Signal Discovery

Multiple temporal windows around GT boundaries were evaluated.

The most useful signals were changes in:

- event type
- event layer
- event-type Jaccard similarity
- layer Jaccard similarity
- event-type entropy
- pre-boundary activity rate

Application name and window title provided secondary information.

Browser-domain features were effectively constant/non-discriminative for this dataset.

An important temporal pattern emerged:

**real boundaries tend to have lower activity immediately before the transition and a stronger activity burst immediately after it.**

The evidence was strongest at short temporal scales.

---

## 4. Boundary vs Normal Activity

A balanced boundary-vs-normal dataset was created with 1,590 positive and 1,590 negative examples.

Session-grouped logistic-regression baselines showed:

| Temporal window | ROC-AUC |
|---|---:|
| ±0.5s | 0.9946 |
| ±1s | 0.9904 |
| ±2s | 0.9902 |
| ±3s | 0.9757 |
| ±5s | 0.9682 |
| ±10s | 0.9220 |

The conclusion was that boundary evidence is highly localized.

However, this evaluation used relatively easy negatives and therefore could not be treated as final detector performance.

---

## 5. Hard-Negative Stress Test

Hard negatives were then sampled from ordinary activity near GT boundaries but not at the actual boundary.

This changed the conclusion substantially.

High-scoring false positives frequently contained:

- keystrokes
- shortcuts
- mouse clicks
- clipboard activity
- screenshots
- short application activity bursts

Therefore, the detector was initially learning **short-term activity change**, not necessarily a sustained business-process transition.

This was the most important failure mode discovered on Day 2.

---

## 6. Persistence and Multi-Scale Evidence

Temporal persistence was tested by evaluating transition scores around the candidate center.

GT boundaries showed consistently higher persistence of elevated transition evidence than hard negatives.

A multi-scale analysis was also performed using ±0.5s, ±1s, ±2s and ±3s scores.

The best simple equal-weight combination was the average across all four scales:

- ROC-AUC: 0.9786
- PR-AUC: 0.9424

This supports a multi-scale representation, while also showing that averaging scores does not completely solve the hard-negative problem.

---

## 7. Continuous Candidate Detection

A continuous transition score was applied to the event stream.

The resulting candidate detector achieved:

| Metric | Result |
|---|---:|
| GT boundaries | 1,590 |
| Candidates | 8,456 |
| TP | 1,456 |
| FP | 7,000 |
| FN | 134 |
| Precision | 0.172 |
| Recall | 0.916 |
| F1 | 0.290 |

Localization was reasonably good, but the false-positive rate was far too high for direct segmentation.

This confirmed that **candidate generation and final boundary confirmation must be separate stages**.

---

## 8. Learned Boundary Confirmation

A confirmation model was trained using corrected one-to-one matching between candidate boundaries and GT boundaries.

The corrected dataset contained:

- 1,456 positive candidates
- 7,000 negative candidates
- 134 GT boundaries without a matched candidate

Session-grouped cross-validation gave ROC-AUC values between 0.934 and 0.951.

Overall:

- ROC-AUC: 0.9428
- PR-AUC: 0.7677

At threshold 0.70:

| Metric | Result |
|---|---:|
| Predictions | 1,455 |
| TP | 1,166 |
| FP | 289 |
| FN | 424 |
| Precision | 0.8014 |
| Recall | 0.7333 |
| F1 | 0.7658 |

### Localization

For matched boundaries:

- Mean error: **0.132s**
- Median error: **0.089s**
- P90 error: **0.291s**
- Within 0.25s: approximately **87%**
- Within 0.50s: **97.34%**
- Within 1s: **99.74%**

The detector therefore produces reasonably precise boundary locations when it identifies a boundary.

---

## 9. Threshold Trade-off

The final validation compared thresholds 0.60, 0.70 and 0.80.

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.60 | 0.7541 | 0.7541 | 0.7541 |
| **0.70** | **0.8014** | **0.7333** | **0.7658** |
| 0.80 | 0.8361 | 0.7025 | 0.7635 |

0.70 was selected as the working operating point because it gave the best F1 among the tested thresholds while keeping precision above 80%.

This is an engineering operating point, not a claim that 0.70 is universally optimal.

---

## 10. Generalization Check

The final validation covered all 63 sessions.

Mean session-level results at threshold 0.70 were approximately:

- Precision: 0.804
- Recall: 0.736
- F1: 0.762

Performance was not uniform. Several sessions had substantially weaker recall, including one session where no GT boundaries were detected at the selected threshold.

This matters because a global aggregate score can hide session-specific failure modes.

---

## 11. What We Learned

### Strong signals

1. Event-type structure
2. Event-layer structure
3. Short-term temporal activity
4. Temporal persistence
5. Window-title/application information as secondary evidence

### Weak or redundant signals

- Browser domain
- Process name as a direct proxy for business process
- Duplicate count/rate features
- Large temporal windows as the primary representation

### Critical design insight

A process boundary cannot safely be defined as:

> “the moment an application changes”  
> or  
> “the moment activity suddenly increases.”

A better formulation is:

> **a boundary is a sustained change in the structure of observed operations, confirmed across temporal context.**

---

## 12. Current Architecture Direction

Day 2 supports the following pipeline:

```text
Raw events
    ↓
Temporal feature extraction
    ↓
Transition candidate generation
    ↓
Multi-scale / persistence evidence
    ↓
Learned boundary confirmation
    ↓
Confirmed boundaries
    ↓
Process identity + segment construction
```

The boundary detector should therefore be treated as a component of the final segmentation system rather than the entire system.

---

## 13. Limitations

The current results should not be interpreted as final end-to-end segmentation accuracy.

Important limitations:

1. Hard negatives are still constructed from Dataset A and are not equivalent to the unknown production distribution in Dataset B.
2. GT labels contain process-switch/interruption behaviour that requires additional interpretation for coherent business-process segments.
3. Candidate/GT matching uses a strict one-to-one tolerance-based procedure and is not a proof of globally optimal matching.
4. The learned model is currently a logistic-regression baseline; more important than model complexity is getting the temporal and process-state representation correct.
5. Boundary quality does not guarantee correct process identity or correct handling of interleaved work.

---

## 14. Day 2 Decision

**Status: READY FOR PHASE 6**

The boundary-detection foundation is sufficiently validated to move forward.

We will not spend additional time tuning the boundary classifier in isolation. The next priority is to understand **process identity and coherent segment construction**, because the final deliverable requires not only boundaries but also consistent labels for the same logical process.

---

## 15. Next Step

Day 3 / Phase 6 will investigate:

- process identity features
- logical process vs application identity
- repeated executions of the same process
- process variants
- interruptions and suspension/resumption
- interleaving
- segment construction
- consistent process labeling

The goal is to turn the validated boundary evidence into actual business-process segments suitable for Dataset B analysis.
