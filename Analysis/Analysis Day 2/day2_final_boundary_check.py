from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

BASE_OUTPUT = Path(
    r"Outputs\Day 2"
)

V2_ROOT = (
    BASE_OUTPUT /
    "day2_learned_boundary_confirmation_v2"
)

GT_FILE = (
    BASE_OUTPUT /
    "day2_gt_boundary_analysis" /
    "gt_switch_boundaries.csv"
)

SCORES_FILE = (
    V2_ROOT /
    "learned_boundary_scores_v2.csv"
)

THRESHOLD_FILE = (
    V2_ROOT /
    "actual_boundary_detection_thresholds.csv"
)

OUTPUT_ROOT = (
    BASE_OUTPUT /
    "day2_final_boundary_validation"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

MATCH_TOLERANCE_SEC = 2.0

# Main operating point from 5H-v2
MAIN_THRESHOLD = 0.70

# Additional operating points for comparison
COMPARISON_THRESHOLDS = [
    0.60,
    0.70,
    0.80,
]


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("PHASE 5I — FINAL DAY-2 BOUNDARY VALIDATION")
print("=" * 70)

print("\nLoading 5H-v2 scores...")

if not SCORES_FILE.exists():
    raise FileNotFoundError(
        f"Missing:\n{SCORES_FILE}"
    )

if not GT_FILE.exists():
    raise FileNotFoundError(
        f"Missing:\n{GT_FILE}"
    )

df = pd.read_csv(
    SCORES_FILE
)

gt = pd.read_csv(
    GT_FILE
)

print(
    f"Candidate rows : {len(df):,}"
)

print(
    f"GT boundaries  : {len(gt):,}"
)


# ============================================================
# CLEAN
# ============================================================

df["session_id"] = (
    df["session_id"]
    .astype(str)
)

gt["session_id"] = (
    gt["session_id"]
    .astype(str)
)

df["candidate_timestamp_ms"] = pd.to_numeric(
    df["candidate_timestamp_ms"],
    errors="coerce"
)

gt["timestamp_ms"] = pd.to_numeric(
    gt["timestamp_ms"],
    errors="coerce"
)

df["boundary_probability"] = pd.to_numeric(
    df["boundary_probability"],
    errors="coerce"
).fillna(0.0)

df = df.dropna(
    subset=[
        "session_id",
        "candidate_timestamp_ms"
    ]
).copy()

gt = gt.dropna(
    subset=[
        "session_id",
        "timestamp_ms"
    ]
).copy()

df = df.reset_index(drop=True)
gt = gt.reset_index(drop=True)


# ============================================================
# ONE-TO-ONE MATCHING
# ============================================================

def match_predictions_to_gt(
    predictions,
    gt_df,
    tolerance_sec
):
    """
    One-to-one temporal matching.

    Highest-probability predictions are considered first.

    This prevents multiple predictions from receiving credit
    for the same GT boundary.
    """

    tolerance_ms = (
        tolerance_sec * 1000.0
    )

    predictions = predictions.copy()

    predictions = predictions.sort_values(
        "boundary_probability",
        ascending=False
    )

    used_gt = set()

    matches = []

    for pred_idx, pred in predictions.iterrows():

        session_id = pred[
            "session_id"
        ]

        pred_time = pred[
            "candidate_timestamp_ms"
        ]

        session_gt = gt_df[
            gt_df["session_id"]
            == session_id
        ]

        if len(session_gt) == 0:
            continue

        distances = np.abs(
            session_gt[
                "timestamp_ms"
            ].to_numpy()
            - pred_time
        )

        valid_positions = np.where(
            distances <= tolerance_ms
        )[0]

        if len(valid_positions) == 0:
            continue

        # Find closest unused GT
        valid_pairs = []

        gt_indices = (
            session_gt.index.to_numpy()
        )

        for pos in valid_positions:

            gt_idx = int(
                gt_indices[pos]
            )

            if gt_idx in used_gt:
                continue

            valid_pairs.append(
                (
                    float(distances[pos]),
                    gt_idx
                )
            )

        if not valid_pairs:
            continue

        valid_pairs.sort(
            key=lambda x: x[0]
        )

        distance, gt_idx = (
            valid_pairs[0]
        )

        used_gt.add(
            gt_idx
        )

        matches.append({
            "candidate_index": int(
                pred_idx
            ),
            "gt_index": int(
                gt_idx
            ),
            "session_id": session_id,
            "candidate_timestamp_ms": float(
                pred_time
            ),
            "gt_timestamp_ms": float(
                gt.loc[
                    gt_idx,
                    "timestamp_ms"
                ]
            ),
            "error_ms": float(
                distance
            ),
            "probability": float(
                pred["boundary_probability"]
            ),
        })

    return pd.DataFrame(
        matches
    )


# ============================================================
# BASIC PERFORMANCE AT MAIN THRESHOLD
# ============================================================

print("\n" + "=" * 70)
print("MAIN OPERATING POINT")
print("=" * 70)

main_predictions = df[
    df["boundary_probability"]
    >= MAIN_THRESHOLD
].copy()

main_matches = match_predictions_to_gt(
    main_predictions,
    gt,
    MATCH_TOLERANCE_SEC
)

tp = len(main_matches)

fp = (
    len(main_predictions)
    - tp
)

fn = (
    len(gt)
    - tp
)

precision = (
    tp / (tp + fp)
    if tp + fp > 0
    else 0.0
)

recall = (
    tp / (tp + fn)
    if tp + fn > 0
    else 0.0
)

f1 = (
    2 * precision * recall
    / (precision + recall)
    if precision + recall > 0
    else 0.0
)

print(
    f"Threshold       : {MAIN_THRESHOLD:.2f}"
)

print(
    f"Predictions     : {len(main_predictions):,}"
)

print(
    f"TP              : {tp:,}"
)

print(
    f"FP              : {fp:,}"
)

print(
    f"FN              : {fn:,}"
)

print(
    f"Precision       : {precision:.4f}"
)

print(
    f"Recall          : {recall:.4f}"
)

print(
    f"F1              : {f1:.4f}"
)


# ============================================================
# LOCALIZATION DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("LOCALIZATION ANALYSIS")
print("=" * 70)

if len(main_matches) > 0:

    errors_sec = (
        main_matches["error_ms"]
        / 1000.0
    )

    localization = {
        "mean_sec": float(
            errors_sec.mean()
        ),
        "median_sec": float(
            errors_sec.median()
        ),
        "p75_sec": float(
            errors_sec.quantile(0.75)
        ),
        "p90_sec": float(
            errors_sec.quantile(0.90)
        ),
        "p95_sec": float(
            errors_sec.quantile(0.95)
        ),
        "max_sec": float(
            errors_sec.max()
        ),
        "within_0.10_sec": float(
            (errors_sec <= 0.10).mean()
        ),
        "within_0.25_sec": float(
            (errors_sec <= 0.25).mean()
        ),
        "within_0.50_sec": float(
            (errors_sec <= 0.50).mean()
        ),
        "within_1.00_sec": float(
            (errors_sec <= 1.00).mean()
        ),
        "within_2.00_sec": float(
            (errors_sec <= 2.00).mean()
        ),
    }

    for key, value in localization.items():

        if key.startswith("within"):

            print(
                f"{key:<22}: "
                f"{value * 100:.2f}%"
            )

        else:

            print(
                f"{key:<22}: "
                f"{value:.4f}s"
            )

else:

    localization = {}


# ============================================================
# SAVE MATCHES
# ============================================================

main_matches.to_csv(
    OUTPUT_ROOT /
    "main_threshold_matches.csv",
    index=False
)


# ============================================================
# THRESHOLD COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("OPERATING POINT COMPARISON")
print("=" * 70)

threshold_rows = []

for threshold in COMPARISON_THRESHOLDS:

    predictions = df[
        df["boundary_probability"]
        >= threshold
    ].copy()

    matches = match_predictions_to_gt(
        predictions,
        gt,
        MATCH_TOLERANCE_SEC
    )

    tp_n = len(matches)

    fp_n = (
        len(predictions)
        - tp_n
    )

    fn_n = (
        len(gt)
        - tp_n
    )

    p = (
        tp_n / (tp_n + fp_n)
        if tp_n + fp_n > 0
        else 0.0
    )

    r = (
        tp_n / (tp_n + fn_n)
        if tp_n + fn_n > 0
        else 0.0
    )

    f = (
        2 * p * r / (p + r)
        if p + r > 0
        else 0.0
    )

    threshold_rows.append({
        "threshold": threshold,
        "predictions": len(predictions),
        "tp": tp_n,
        "fp": fp_n,
        "fn": fn_n,
        "precision": p,
        "recall": r,
        "f1": f,
    })

    print(
        f"\nThreshold {threshold:.2f}"
    )

    print(
        f"  Predictions : {len(predictions):,}"
    )

    print(
        f"  TP          : {tp_n:,}"
    )

    print(
        f"  FP          : {fp_n:,}"
    )

    print(
        f"  FN          : {fn_n:,}"
    )

    print(
        f"  Precision   : {p:.4f}"
    )

    print(
        f"  Recall      : {r:.4f}"
    )

    print(
        f"  F1          : {f:.4f}"
    )


threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_df.to_csv(
    OUTPUT_ROOT /
    "operating_point_comparison.csv",
    index=False
)


# ============================================================
# PER-SESSION ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PER-SESSION ROBUSTNESS")
print("=" * 70)

session_rows = []

for session_id in sorted(
    gt["session_id"].unique()
):

    session_gt = gt[
        gt["session_id"]
        == session_id
    ]

    session_predictions = main_predictions[
        main_predictions["session_id"]
        == session_id
    ]

    session_matches = match_predictions_to_gt(
        session_predictions,
        session_gt,
        MATCH_TOLERANCE_SEC
    )

    session_tp = len(
        session_matches
    )

    session_fp = (
        len(session_predictions)
        - session_tp
    )

    session_fn = (
        len(session_gt)
        - session_tp
    )

    session_precision = (
        session_tp
        / (
            session_tp
            + session_fp
        )
        if session_tp + session_fp > 0
        else 0.0
    )

    session_recall = (
        session_tp
        / (
            session_tp
            + session_fn
        )
        if session_tp + session_fn > 0
        else 0.0
    )

    session_f1 = (
        2
        * session_precision
        * session_recall
        / (
            session_precision
            + session_recall
        )
        if (
            session_precision
            + session_recall
        ) > 0
        else 0.0
    )

    session_rows.append({
        "session_id": session_id,
        "gt_boundaries": len(
            session_gt
        ),
        "predictions": len(
            session_predictions
        ),
        "tp": session_tp,
        "fp": session_fp,
        "fn": session_fn,
        "precision": session_precision,
        "recall": session_recall,
        "f1": session_f1,
    })


session_df = pd.DataFrame(
    session_rows
)

session_df.to_csv(
    OUTPUT_ROOT /
    "session_metrics.csv",
    index=False
)


# ============================================================
# SESSION DISTRIBUTION SUMMARY
# ============================================================

print(
    f"\nSessions evaluated: "
    f"{len(session_df)}"
)

print(
    f"Mean session precision: "
    f"{session_df['precision'].mean():.4f}"
)

print(
    f"Median session precision: "
    f"{session_df['precision'].median():.4f}"
)

print(
    f"Mean session recall: "
    f"{session_df['recall'].mean():.4f}"
)

print(
    f"Median session recall: "
    f"{session_df['recall'].median():.4f}"
)

print(
    f"Mean session F1: "
    f"{session_df['f1'].mean():.4f}"
)

print(
    f"Median session F1: "
    f"{session_df['f1'].median():.4f}"
)


# ============================================================
# WORST SESSIONS
# ============================================================

worst_sessions = (
    session_df
    .sort_values(
        ["f1", "recall"],
        ascending=True
    )
    .head(10)
)

worst_sessions.to_csv(
    OUTPUT_ROOT /
    "worst_sessions.csv",
    index=False
)

print(
    "\nLowest-F1 sessions:"
)

print(
    worst_sessions[
        [
            "session_id",
            "gt_boundaries",
            "predictions",
            "precision",
            "recall",
            "f1",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# FALSE POSITIVE FORENSICS
# ============================================================

print("\n" + "=" * 70)
print("FALSE-POSITIVE FORENSICS")
print("=" * 70)

matched_candidate_indices = set()

if len(main_matches) > 0:

    matched_candidate_indices = set(
        main_matches[
            "candidate_index"
        ].astype(int)
    )

main_predictions_with_index = (
    main_predictions.copy()
)

main_predictions_with_index[
    "candidate_index"
] = main_predictions_with_index.index

false_positives = (
    main_predictions_with_index[
        ~main_predictions_with_index[
            "candidate_index"
        ].isin(
            matched_candidate_indices
        )
    ]
    .copy()
)

print(
    f"False positives: "
    f"{len(false_positives):,}"
)


fp_feature_columns = [
    "session_id",
    "candidate_timestamp_ms",
    "boundary_probability",
    "boundary_score",
    "persistence_sec",
    "persistence_points",
    "pre_event_count",
    "post_event_count",
    "event_jaccard",
    "layer_jaccard",
    "app_jaccard",
    "process_jaccard",
    "title_jaccard",
    "event_change",
    "layer_change",
    "app_change",
    "process_change",
    "title_change",
    "structure_score",
    "post_stability_score",
    "score_at_candidate",
    "score_pre_mean",
    "score_post_mean",
    "score_pre_max",
    "score_post_max",
]

fp_feature_columns = [
    c
    for c in fp_feature_columns
    if c in false_positives.columns
]

false_positives[
    fp_feature_columns
].sort_values(
    "boundary_probability",
    ascending=False
).to_csv(
    OUTPUT_ROOT /
    "false_positive_forensics.csv",
    index=False
)


# ============================================================
# FALSE NEGATIVE FORENSICS
# ============================================================

print("\n" + "=" * 70)
print("FALSE-NEGATIVE FORENSICS")
print("=" * 70)

matched_gt_indices = set()

if len(main_matches) > 0:

    matched_gt_indices = set(
        main_matches[
            "gt_index"
        ].astype(int)
    )

false_negative_gt = gt[
    ~gt.index.isin(
        matched_gt_indices
    )
].copy()

print(
    f"False-negative GT boundaries: "
    f"{len(false_negative_gt):,}"
)

false_negative_gt.to_csv(
    OUTPUT_ROOT /
    "false_negative_gt_boundaries.csv",
    index=False
)


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SCORE DISTRIBUTION")
print("=" * 70)

if len(main_matches) > 0:

    tp_scores = main_matches[
        "probability"
    ]

else:

    tp_scores = pd.Series(
        dtype=float
    )

fp_scores = false_positives[
    "boundary_probability"
]

matched_gt_candidate_indices = (
    list(matched_candidate_indices)
)

all_candidate_indices = set(
    main_predictions.index
)

print(
    f"TP score mean: "
    f"{tp_scores.mean():.4f}"
)

print(
    f"FP score mean: "
    f"{fp_scores.mean():.4f}"
)

print(
    f"FP score median: "
    f"{fp_scores.median():.4f}"
)

print(
    f"FP score P90: "
    f"{fp_scores.quantile(0.90):.4f}"
)

print(
    f"FP score P95: "
    f"{fp_scores.quantile(0.95):.4f}"
)


# ============================================================
# DAY-2 DECISION
# ============================================================

if f1 >= 0.75 and precision >= 0.75:

    decision = (
        "READY_FOR_PHASE_6"
    )

elif f1 >= 0.65:

    decision = (
        "PROVISIONALLY_READY_FOR_PHASE_6"
    )

else:

    decision = (
        "BOUNDARY_DETECTOR_NEEDS_MORE_WORK"
    )


print("\n" + "=" * 70)
print("DAY-2 BOUNDARY DECISION")
print("=" * 70)

print(
    f"Decision: {decision}"
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "phase": "5I",

    "gt_boundaries": int(
        len(gt)
    ),

    "candidate_rows": int(
        len(df)
    ),

    "main_threshold": MAIN_THRESHOLD,

    "match_tolerance_sec":
        MATCH_TOLERANCE_SEC,

    "predictions": int(
        len(main_predictions)
    ),

    "tp": int(tp),

    "fp": int(fp),

    "fn": int(fn),

    "precision": float(
        precision
    ),

    "recall": float(
        recall
    ),

    "f1": float(
        f1
    ),

    "localization": localization,

    "mean_session_precision": float(
        session_df["precision"].mean()
    ),

    "median_session_precision": float(
        session_df["precision"].median()
    ),

    "mean_session_recall": float(
        session_df["recall"].mean()
    ),

    "median_session_recall": float(
        session_df["recall"].median()
    ),

    "mean_session_f1": float(
        session_df["f1"].mean()
    ),

    "median_session_f1": float(
        session_df["f1"].median()
    ),

    "false_positive_count": int(
        len(false_positives)
    ),

    "false_negative_count": int(
        len(false_negative_gt)
    ),

    "decision": decision,
}


with open(
    OUTPUT_ROOT /
    "phase5i_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5I COMPLETE")
print("=" * 70)

print(
    f"Boundary F1      : {f1:.4f}"
)

print(
    f"Precision        : {precision:.4f}"
)

print(
    f"Recall           : {recall:.4f}"
)

if localization:

    print(
        f"Median error     : "
        f"{localization['median_sec']:.4f}s"
    )

    print(
        f"P90 error        : "
        f"{localization['p90_sec']:.4f}s"
    )

print(
    f"False positives  : "
    f"{len(false_positives):,}"
)

print(
    f"False negatives  : "
    f"{len(false_negative_gt):,}"
)

print(
    f"\nDecision         : "
    f"{decision}"
)

print(
    f"\nOutputs saved to:\n"
    f"{OUTPUT_ROOT}"
)