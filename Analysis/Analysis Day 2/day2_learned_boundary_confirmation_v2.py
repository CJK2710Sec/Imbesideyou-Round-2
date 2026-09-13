from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

FEATURE_FILE = Path(
    r"Outputs\Day 2\day2_boundary_confirmation"
) / "candidate_confirmation_features.csv"

GT_FILE = Path(
    r"Outputs\Day 2\day2_gt_boundary_analysis"
) / "gt_switch_boundaries.csv"

OUTPUT_ROOT = Path(
    r"Outputs\Day 2\day2_learned_boundary_confirmation_v2"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

MATCH_TOLERANCE_SEC = 2.0

N_SPLITS = 5

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 5H-v2 — LEARNED BOUNDARY CONFIRMATION")
print("ONE-TO-ONE GT MATCHING")
print("=" * 70)

print("\nLoading 5G candidate features...")

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Feature file not found:\n{FEATURE_FILE}"
    )

if not GT_FILE.exists():
    raise FileNotFoundError(
        f"GT file not found:\n{GT_FILE}"
    )

df = pd.read_csv(FEATURE_FILE)

gt = pd.read_csv(GT_FILE)

print(f"Candidate rows : {len(df):,}")
print(f"GT boundaries  : {len(gt):,}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_candidate_columns = [
    "session_id",
    "candidate_timestamp_ms",
]

required_gt_columns = [
    "session_id",
    "timestamp_ms",
]

for col in required_candidate_columns:

    if col not in df.columns:
        raise ValueError(
            f"Missing candidate column: {col}"
        )

for col in required_gt_columns:

    if col not in gt.columns:
        raise ValueError(
            f"Missing GT column: {col}"
        )


# ============================================================
# CLEAN TIMESTAMPS
# ============================================================

df["candidate_timestamp_ms"] = pd.to_numeric(
    df["candidate_timestamp_ms"],
    errors="coerce"
)

gt["timestamp_ms"] = pd.to_numeric(
    gt["timestamp_ms"],
    errors="coerce"
)

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

df["session_id"] = df["session_id"].astype(str)
gt["session_id"] = gt["session_id"].astype(str)

df = df.reset_index(drop=True)

gt = gt.reset_index(drop=True)


# ============================================================
# ONE-TO-ONE MATCHING
# ============================================================

print("\n" + "=" * 70)
print("BUILDING ONE-TO-ONE GT ↔ CANDIDATE MATCHES")
print("=" * 70)

tolerance_ms = MATCH_TOLERANCE_SEC * 1000.0

df["label"] = 0
df["matched_gt_index"] = -1
df["gt_distance_ms"] = np.nan


def one_to_one_match(
    candidate_times,
    candidate_indices,
    gt_times,
    gt_indices,
    tolerance
):
    """
    Greedy one-to-one matching.

    Candidate and GT pairs are sorted by temporal distance.
    Each candidate and GT can be used only once.

    This guarantees:

        one GT -> at most one candidate
        one candidate -> at most one GT
    """

    possible_pairs = []

    for local_c, global_c in enumerate(candidate_indices):

        ctime = candidate_times[local_c]

        distances = np.abs(
            gt_times - ctime
        )

        valid_positions = np.where(
            distances <= tolerance
        )[0]

        for pos in valid_positions:

            possible_pairs.append(
                (
                    float(distances[pos]),
                    int(global_c),
                    int(gt_indices[pos])
                )
            )

    possible_pairs.sort(
        key=lambda x: x[0]
    )

    used_candidates = set()
    used_gt = set()

    matches = []

    for distance, candidate_idx, gt_idx in possible_pairs:

        if candidate_idx in used_candidates:
            continue

        if gt_idx in used_gt:
            continue

        used_candidates.add(candidate_idx)
        used_gt.add(gt_idx)

        matches.append(
            (
                candidate_idx,
                gt_idx,
                distance
            )
        )

    return matches


all_matches = []

session_ids = sorted(
    set(df["session_id"]).intersection(
        set(gt["session_id"])
    )
)

print(
    f"Sessions with candidates + GT: "
    f"{len(session_ids)}"
)

for session_id in session_ids:

    candidate_mask = (
        df["session_id"] == session_id
    )

    gt_mask = (
        gt["session_id"] == session_id
    )

    candidate_indices = (
        df.index[candidate_mask]
        .to_numpy()
    )

    gt_indices = (
        gt.index[gt_mask]
        .to_numpy()
    )

    if len(candidate_indices) == 0:
        continue

    if len(gt_indices) == 0:
        continue

    candidate_times = df.loc[
        candidate_indices,
        "candidate_timestamp_ms"
    ].to_numpy()

    gt_times = gt.loc[
        gt_indices,
        "timestamp_ms"
    ].to_numpy()

    matches = one_to_one_match(
        candidate_times,
        candidate_indices,
        gt_times,
        gt_indices,
        tolerance_ms
    )

    all_matches.extend(matches)


# ============================================================
# APPLY MATCH LABELS
# ============================================================

for candidate_idx, gt_idx, distance in all_matches:

    df.loc[
        candidate_idx,
        "label"
    ] = 1

    df.loc[
        candidate_idx,
        "matched_gt_index"
    ] = gt_idx

    df.loc[
        candidate_idx,
        "gt_distance_ms"
    ] = distance


# ============================================================
# MATCHING SUMMARY
# ============================================================

matched_count = int(
    (df["label"] == 1).sum()
)

gt_count = len(gt)

unmatched_gt = gt_count - matched_count

print("\nGT boundaries:")
print(
    f"  Total GT boundaries     : {gt_count:,}"
)

print(
    f"  Matched positive        : {matched_count:,}"
)

print(
    f"  Unmatched GT boundaries : {unmatched_gt:,}"
)

print(
    f"  Candidate negatives     : "
    f"{(df['label'] == 0).sum():,}"
)

print(
    f"  Max possible positives  : "
    f"{gt_count:,}"
)


# ============================================================
# SAVE MATCHED DATA
# ============================================================

df.to_csv(
    OUTPUT_ROOT /
    "one_to_one_labeled_candidates.csv",
    index=False
)


# ============================================================
# CHECK FOR DUPLICATE GT ASSIGNMENTS
# ============================================================

matched = df[
    df["label"] == 1
].copy()

duplicate_gt_matches = (
    matched[
        matched["matched_gt_index"] >= 0
    ]
    .groupby("matched_gt_index")
    .size()
)

duplicate_gt_count = int(
    (duplicate_gt_matches > 1).sum()
)

print("\nMatching sanity check:")
print(
    f"  Duplicate GT assignments : "
    f"{duplicate_gt_count}"
)

if duplicate_gt_count != 0:
    raise RuntimeError(
        "One-to-one matching failed: "
        "duplicate GT assignments detected."
    )


# ============================================================
# FEATURE SELECTION
# ============================================================

print("\n" + "=" * 70)
print("SELECTING LEARNED FEATURES")
print("=" * 70)


excluded_columns = {
    "label",
    "matched_gt_index",
    "gt_distance_ms",

    # identifiers
    "session_id",

    # timestamps / temporal bookkeeping
    "candidate_timestamp_ms",
    "candidate_timestamp_sec",
    "region_start_ms",
    "region_end_ms",

    # DO NOT feed the hand-designed 5G final score
    "confirmation_score",
}


numeric_columns = df.select_dtypes(
    include=[np.number]
).columns.tolist()


feature_cols = [
    c for c in numeric_columns
    if c not in excluded_columns
]


if not feature_cols:

    raise ValueError(
        "No usable numeric features found."
    )


print(
    f"Number of learned features: "
    f"{len(feature_cols)}"
)

for feature in feature_cols:
    print(f"  {feature}")


# ============================================================
# CLEAN FEATURES
# ============================================================

for col in feature_cols:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df[feature_cols] = (
    df[feature_cols]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0.0)
)


X = df[feature_cols]

y = df["label"].astype(int)

groups = df["session_id"].astype(str)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("CORRECTED CLASS DISTRIBUTION")
print("=" * 70)

positive_count = int(
    (y == 1).sum()
)

negative_count = int(
    (y == 0).sum()
)

print(
    f"Positive candidates : {positive_count:,}"
)

print(
    f"Negative candidates : {negative_count:,}"
)

print(
    f"Total candidates    : {len(df):,}"
)


# ============================================================
# SESSION-GROUPED CROSS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("SESSION-GROUPED CROSS VALIDATION")
print("=" * 70)

gkf = GroupKFold(
    n_splits=N_SPLITS
)

oof_probability = np.zeros(
    len(df),
    dtype=float
)

fold_results = []


for fold, (
    train_idx,
    test_idx
) in enumerate(
    gkf.split(X, y, groups),
    start=1
):

    print(
        f"\nFold {fold}/{N_SPLITS}"
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        ),
    ])

    model.fit(
        X_train,
        y_train
    )

    probability = model.predict_proba(
        X_test
    )[:, 1]

    oof_probability[
        test_idx
    ] = probability

    auc = roc_auc_score(
        y_test,
        probability
    )

    pr_auc = average_precision_score(
        y_test,
        probability
    )

    print(
        f"  ROC-AUC : {auc:.4f}"
    )

    print(
        f"  PR-AUC  : {pr_auc:.4f}"
    )

    fold_results.append({
        "fold": fold,
        "roc_auc": float(auc),
        "pr_auc": float(pr_auc),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_sessions": int(
            groups.iloc[train_idx].nunique()
        ),
        "test_sessions": int(
            groups.iloc[test_idx].nunique()
        ),
    })


df["boundary_probability"] = (
    oof_probability
)


# ============================================================
# OVERALL CLASSIFICATION METRICS
# ============================================================

overall_auc = roc_auc_score(
    y,
    oof_probability
)

overall_pr_auc = average_precision_score(
    y,
    oof_probability
)

print("\n" + "=" * 70)
print("OVERALL LEARNED MODEL")
print("=" * 70)

print(
    f"ROC-AUC : {overall_auc:.4f}"
)

print(
    f"PR-AUC  : {overall_pr_auc:.4f}"
)


# ============================================================
# CLASSIFICATION THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE CLASSIFICATION THRESHOLDS")
print("=" * 70)

thresholds = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

classification_rows = []


for threshold in thresholds:

    predictions = (
        oof_probability >= threshold
    ).astype(int)

    tp = int(
        ((predictions == 1) & (y == 1))
        .sum()
    )

    fp = int(
        ((predictions == 1) & (y == 0))
        .sum()
    )

    fn = int(
        ((predictions == 0) & (y == 1))
        .sum()
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"  Candidates : "
        f"{predictions.sum():,}"
    )

    print(
        f"  TP         : {tp:,}"
    )

    print(
        f"  FP         : {fp:,}"
    )

    print(
        f"  FN         : {fn:,}"
    )

    print(
        f"  Precision  : {precision:.4f}"
    )

    print(
        f"  Recall     : {recall:.4f}"
    )

    print(
        f"  F1         : {f1:.4f}"
    )

    classification_rows.append({
        "threshold": threshold,
        "candidates": int(predictions.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    })


classification_df = pd.DataFrame(
    classification_rows
)

classification_df.to_csv(
    OUTPUT_ROOT /
    "candidate_classification_thresholds.csv",
    index=False
)


# ============================================================
# ACTUAL BOUNDARY DETECTION EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("ACTUAL ONE-TO-ONE BOUNDARY DETECTION")
print("=" * 70)


def evaluate_boundary_detection(
    candidate_df,
    gt_df,
    probability_threshold,
    tolerance_sec
):
    """
    Evaluate predicted boundary candidates against GT.

    A predicted candidate can match only one GT.
    A GT can match only one prediction.
    """

    selected = candidate_df[
        candidate_df["boundary_probability"]
        >= probability_threshold
    ].copy()

    tolerance = tolerance_sec * 1000.0

    matched_candidates = set()
    matched_gt = set()

    pairs = []

    # Build all candidate-GT pairs
    possible_pairs = []

    for session_id in sorted(
        set(selected["session_id"])
        .intersection(
            set(gt_df["session_id"])
        )
    ):

        candidates = selected[
            selected["session_id"]
            == session_id
        ]

        gt_session = gt_df[
            gt_df["session_id"]
            == session_id
        ]

        for candidate_idx, candidate_row in (
            candidates.iterrows()
        ):

            candidate_time = (
                candidate_row[
                    "candidate_timestamp_ms"
                ]
            )

            distances = np.abs(
                gt_session[
                    "timestamp_ms"
                ].to_numpy()
                - candidate_time
            )

            valid = np.where(
                distances <= tolerance
            )[0]

            gt_indices = (
                gt_session.index.to_numpy()
            )

            for pos in valid:

                possible_pairs.append(
                    (
                        float(distances[pos]),
                        candidate_idx,
                        int(gt_indices[pos])
                    )
                )

    # Closest pairs first
    possible_pairs.sort(
        key=lambda x: x[0]
    )

    for distance, candidate_idx, gt_idx in (
        possible_pairs
    ):

        if candidate_idx in matched_candidates:
            continue

        if gt_idx in matched_gt:
            continue

        matched_candidates.add(
            candidate_idx
        )

        matched_gt.add(
            gt_idx
        )

        pairs.append(
            (
                candidate_idx,
                gt_idx,
                distance
            )
        )

    tp = len(pairs)

    fp = len(selected) - tp

    fn = len(gt_df) - tp

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    errors_sec = [
        distance / 1000.0
        for _, _, distance in pairs
    ]

    if errors_sec:

        mean_error = float(
            np.mean(errors_sec)
        )

        median_error = float(
            np.median(errors_sec)
        )

        p90_error = float(
            np.percentile(
                errors_sec,
                90
            )
        )

    else:

        mean_error = 0.0
        median_error = 0.0
        p90_error = 0.0

    return {
        "threshold": probability_threshold,
        "predicted_candidates": int(
            len(selected)
        ),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_error_sec": mean_error,
        "median_error_sec": median_error,
        "p90_error_sec": p90_error,
    }


boundary_rows = []


for threshold in thresholds:

    result = evaluate_boundary_detection(
        df,
        gt,
        threshold,
        MATCH_TOLERANCE_SEC
    )

    boundary_rows.append(
        result
    )

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"  Candidates : "
        f"{result['predicted_candidates']:,}"
    )

    print(
        f"  TP         : {result['tp']:,}"
    )

    print(
        f"  FP         : {result['fp']:,}"
    )

    print(
        f"  FN         : {result['fn']:,}"
    )

    print(
        f"  Precision  : "
        f"{result['precision']:.4f}"
    )

    print(
        f"  Recall     : "
        f"{result['recall']:.4f}"
    )

    print(
        f"  F1         : "
        f"{result['f1']:.4f}"
    )

    print(
        f"  Mean error : "
        f"{result['mean_error_sec']:.3f}s"
    )

    print(
        f"  Median     : "
        f"{result['median_error_sec']:.3f}s"
    )

    print(
        f"  P90        : "
        f"{result['p90_error_sec']:.3f}s"
    )


boundary_df = pd.DataFrame(
    boundary_rows
)

boundary_df.to_csv(
    OUTPUT_ROOT /
    "actual_boundary_detection_thresholds.csv",
    index=False
)


# ============================================================
# FIND BEST F1
# ============================================================

best_row = boundary_df.loc[
    boundary_df["f1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST ACTUAL BOUNDARY DETECTOR")
print("=" * 70)

print(
    f"Threshold       : "
    f"{best_row['threshold']:.2f}"
)

print(
    f"Candidates      : "
    f"{int(best_row['predicted_candidates']):,}"
)

print(
    f"TP              : "
    f"{int(best_row['tp']):,}"
)

print(
    f"FP              : "
    f"{int(best_row['fp']):,}"
)

print(
    f"FN              : "
    f"{int(best_row['fn']):,}"
)

print(
    f"Precision       : "
    f"{best_row['precision']:.4f}"
)

print(
    f"Recall          : "
    f"{best_row['recall']:.4f}"
)

print(
    f"F1              : "
    f"{best_row['f1']:.4f}"
)

print(
    f"Mean error      : "
    f"{best_row['mean_error_sec']:.3f}s"
)

print(
    f"Median error    : "
    f"{best_row['median_error_sec']:.3f}s"
)

print(
    f"P90 error       : "
    f"{best_row['p90_error_sec']:.3f}s"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("LEARNED FEATURE IMPORTANCE")
print("=" * 70)

final_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    ),
])

final_model.fit(
    X,
    y
)

classifier = final_model.named_steps[
    "classifier"
]

coefficients = classifier.coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": coefficients,
    "abs_coefficient": np.abs(
        coefficients
    ),
})

coef_df = coef_df.sort_values(
    "abs_coefficient",
    ascending=False
)

coef_df.to_csv(
    OUTPUT_ROOT /
    "learned_feature_coefficients.csv",
    index=False
)

print("\nTop learned features:")

for _, row in coef_df.head(20).iterrows():

    direction = (
        "increases"
        if row["coefficient"] > 0
        else "decreases"
    )

    print(
        f"  {row['feature']:<30} "
        f"{row['coefficient']:>8.4f} "
        f"({direction})"
    )


# ============================================================
# SAVE OOF SCORES
# ============================================================

df.to_csv(
    OUTPUT_ROOT /
    "learned_boundary_scores_v2.csv",
    index=False
)


# ============================================================
# SAVE CV RESULTS
# ============================================================

pd.DataFrame(
    fold_results
).to_csv(
    OUTPUT_ROOT /
    "cross_validation_results.csv",
    index=False
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "phase": "5H-v2",

    "candidate_count": int(
        len(df)
    ),

    "gt_boundary_count": int(
        len(gt)
    ),

    "matched_positive_candidates": int(
        positive_count
    ),

    "negative_candidates": int(
        negative_count
    ),

    "unmatched_gt_boundaries": int(
        unmatched_gt
    ),

    "match_tolerance_sec": (
        MATCH_TOLERANCE_SEC
    ),

    "n_sessions": int(
        groups.nunique()
    ),

    "n_features": int(
        len(feature_cols)
    ),

    "features": feature_cols,

    "roc_auc": float(
        overall_auc
    ),

    "pr_auc": float(
        overall_pr_auc
    ),

    "best_boundary_threshold": float(
        best_row["threshold"]
    ),

    "best_boundary_precision": float(
        best_row["precision"]
    ),

    "best_boundary_recall": float(
        best_row["recall"]
    ),

    "best_boundary_f1": float(
        best_row["f1"]
    ),

    "best_boundary_mean_error_sec": float(
        best_row["mean_error_sec"]
    ),

    "best_boundary_median_error_sec": float(
        best_row["median_error_sec"]
    ),

    "best_boundary_p90_error_sec": float(
        best_row["p90_error_sec"]
    ),
}


with open(
    OUTPUT_ROOT /
    "phase5h_v2_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL SANITY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("SANITY CHECKS")
print("=" * 70)

assert positive_count <= gt_count

assert duplicate_gt_count == 0

assert len(df) == (
    positive_count +
    negative_count
)

print(
    "  Positive <= GT count : PASS"
)

print(
    "  One-to-one GT match  : PASS"
)

print(
    "  Label accounting     : PASS"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 5H-v2 COMPLETE")
print("=" * 70)

print(
    f"ROC-AUC : {overall_auc:.4f}"
)

print(
    f"PR-AUC  : {overall_pr_auc:.4f}"
)

print(
    f"\nBest actual boundary F1 : "
    f"{best_row['f1']:.4f}"
)

print(
    f"Best threshold          : "
    f"{best_row['threshold']:.2f}"
)

print(
    f"Precision               : "
    f"{best_row['precision']:.4f}"
)

print(
    f"Recall                  : "
    f"{best_row['recall']:.4f}"
)

print(
    f"Localization error      : "
    f"{best_row['median_error_sec']:.3f}s median"
)

print(
    f"\nOutputs saved to:\n"
    f"{OUTPUT_ROOT}"
)