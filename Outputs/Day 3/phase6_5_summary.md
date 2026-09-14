# Phase 6.5 — Sequence + Context Identity Analysis

## Objective

Test whether local event ordering and short behavioral patterns provide additional logical process identity information beyond aggregate execution-level features.

## Evaluation

- Manifest executions: 2009
- Complete timed executions: 1734
- Process families: 15
- Session groups: 63
- Session-grouped cross-validation
- GT process labels used only as targets
- No previous/next GT process labels used as features

## Results

### sequence_only_logistic
- Accuracy: 0.2970
- Macro F1: 0.3029
- Weighted F1: 0.2977

### sequence_context_random_forest
- Accuracy: 0.3420
- Macro F1: 0.3404
- Weighted F1: 0.3400

## Best model
- Model: sequence_context_random_forest
- Accuracy: 0.3420
- Macro F1: 0.3404

## Interpretation

Phase 6.5 evaluates whether local event ordering, transition structure, short n-grams, and interaction motifs add process-identity information beyond whole-execution aggregates.

The result should be compared with the Phase 6.4 behavioral Random Forest baseline (29.28% accuracy, 0.2828 Macro F1).

A meaningful improvement supports incorporating local sequence/context information into final identity and segmentation. A weak improvement supports a conservative identity mechanism based on process history, transition structure, variants, and confidence rather than standalone classification.