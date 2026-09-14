# Phase 6.6 — Transition-Aware Identity Resolution

## Objective

Test whether learned process-transition compatibility improves identity resolution beyond the Phase 6.5 sequence/context model.

## Evaluation Design

- GT executions: 2009
- Usable timed executions: 1734
- Process families: 15
- Session groups: 63
- Session-grouped GroupKFold
- Base model: sequence/context Random Forest
- Transition probabilities learned only from training-fold GT
- Test-fold neighboring GT labels are never used as prediction features
- Transition-aware decoding uses previous predicted identity, not previous GT identity

## Results

- Base accuracy: 0.3535
- Base Macro F1: 0.3508
- Transition-aware accuracy: 0.3489
- Transition-aware Macro F1: 0.3442
- Accuracy delta: -0.0046
- Macro F1 delta: -0.0066

## Verdict

NOT SUPPORTED: transition-aware resolution does not improve the base model on this evaluation.

## Interpretation

Phase 6.6 tests whether process identity should be resolved using both local behavioral evidence and process-transition structure.

A positive result supports a hybrid identity architecture in which sequence/context evidence is combined with transition compatibility.

A weak or negative result means transition information should not be forced into the identity model and the final approach should remain based primarily on local evidence and confidence.

The experiment does not establish that the result will generalize perfectly to Dataset B; Dataset B has no ground-truth identity labels.