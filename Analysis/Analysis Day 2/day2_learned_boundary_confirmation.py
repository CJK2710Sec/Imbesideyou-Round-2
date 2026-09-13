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
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(r"Dataset A\dataset_a")

OUTPUT_ROOT = Path(
    r"Outputs\Day 2\day2_learned_boundary_confirmation"
)

FEATURE_FILE = Path(
    r"Outputs\Day 2\day2_boundary_confirmation"
) / "candidate_confirmation_features.csv"

GT_FILE = Path(
    r"Outputs\Day 2\day2_gt_boundary_analysis"
) / "gt_switch_boundaries.csv"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

MATCH_TOLERANCE_SEC = 2.0


# ============================================================
# HELPERS
# ============================================================

def safe_numeric(df, columns):
    """
    Convert selected columns to numeric and replace invalid
    values with zero.
    """
    existing = [c for c in columns if c in df.columns]

    for c in existing:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[existing] = df[existing].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0.0)

    return df


def choose_feature_columns(df):
    """
    Select numeric confirmation features.

    Exclude:
      - labels
      - timestamps
      - IDs
      - GT information
      - candidate bookkeeping
    """

    excluded_keywords = [
        "timestamp",
        "session_id",
        "event_id",
        "candidate_id",
        "gt_index",
        "label",
        "target",
        "matched",
        "distance",
        "from_process",
        "to_process",
        "current_process",
        "confirmation_score",
    ]

    numeric_cols = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    features = []

    for c in numeric_cols:

        c_lower = c.lower()

        if any(k in c_lower for k in excluded_keywords):
            continue

        features.append(c)

    return features


# ============================================================
# LOAD 5G FEATURES
# ============================================================

print("=" * 70)
print("PHASE 5H — LEARNED BOUNDARY CONFIRMATION")
print("=" * 70)

print("\nLoading 5G confirmation features...")

if not FEATURE_FILE.exists():
    raise FileNotFoundError(
        f"Missing feature file:\n{FEATURE_FILE}"
    )

df = pd.read_csv(FEATURE_FILE)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# IDENTIFY LABEL
# ============================================================

possible_labels = [
    "label",
    "target",
    "is_boundary",
    "boundary_label",
]

label_col = None

for c in possible_labels:
    if c in df.columns:
        label_col = c
        break


# ============================================================
# IF 5G FILE DOES NOT CONTAIN LABELS
# BUILD LABELS USING GT
# ============================================================

if label_col is None:

    print("\nNo label column found.")
    print("Building labels from GT boundaries...")

    if "session_id" not in df.columns:
        raise ValueError(
            "session_id is required to construct labels."
        )

    timestamp_col = None

    for c in [
        "center_timestamp_ms",
        "timestamp_ms",
        "candidate_timestamp_ms",
        "center_time_ms",
    ]:
        if c in df.columns:
            timestamp_col = c
            break

    if timestamp_col is None:

        for c in [
            "center_timestamp",
            "timestamp",
            "candidate_timestamp",
        ]:
            if c in df.columns:
                timestamp_col = c
                break

    if timestamp_col is None:
        raise ValueError(
            "Could not identify candidate timestamp column."
        )

    gt = pd.read_csv(GT_FILE)

    gt["timestamp_ms"] = pd.to_numeric(
        gt["timestamp_ms"],
        errors="coerce"
    )

    df[timestamp_col] = pd.to_numeric(
        df[timestamp_col],
        errors="coerce"
    )

    df["label"] = 0

    tolerance_ms = MATCH_TOLERANCE_SEC * 1000

    gt_by_session = {
        sid: grp["timestamp_ms"].dropna().values
        for sid, grp in gt.groupby("session_id")
    }

    for idx, row in df.iterrows():

        sid = row["session_id"]
        ts = row[timestamp_col]

        if pd.isna(ts):
            continue

        gt_times = gt_by_session.get(sid)

        if gt_times is None or len(gt_times) == 0:
            continue

        distance = np.min(
            np.abs(gt_times - ts)
        )

        if distance <= tolerance_ms:
            df.at[idx, "label"] = 1

    label_col = "label"


# ============================================================
# BASIC DATA CHECK
# ============================================================

print("\nClass distribution:")

print(
    df[label_col]
    .value_counts()
    .sort_index()
)

print(
    f"Positive: {(df[label_col] == 1).sum():,}"
)

print(
    f"Negative: {(df[label_col] == 0).sum():,}"
)


# ============================================================
# FEATURE SELECTION
# ============================================================

feature_cols = choose_feature_columns(df)

if not feature_cols:
    raise ValueError(
        "No usable numeric features found."
    )

print("\nSelected features:")
for c in feature_cols:
    print(f"  {c}")


# ============================================================
# CLEAN FEATURES
# ============================================================

df = safe_numeric(df, feature_cols)

X = df[feature_cols].copy()

y = df[label_col].astype(int)

groups = df["session_id"].astype(str)


# ============================================================
# SESSION-GROUPED CROSS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("SESSION-GROUPED CROSS VALIDATION")
print("=" * 70)

gkf = GroupKFold(
    n_splits=N_SPLITS
)

oof_probability = np.zeros(len(df))

fold_results = []


for fold, (train_idx, test_idx) in enumerate(
    gkf.split(X, y, groups),
    start=1
):

    print(f"\nFold {fold}/{N_SPLITS}")

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
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )
        ),
    ])

    model.fit(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    oof_probability[test_idx] = probabilities

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    ap = average_precision_score(
        y_test,
        probabilities
    )

    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  PR-AUC  : {ap:.4f}")

    fold_results.append({
        "fold": fold,
        "roc_auc": auc,
        "pr_auc": ap,
        "train_rows": len(train_idx),
        "test_rows": len(test_idx),
        "train_sessions": groups.iloc[train_idx].nunique(),
        "test_sessions": groups.iloc[test_idx].nunique(),
    })


# ============================================================
# SAVE OOF SCORES
# ============================================================

df["boundary_probability"] = oof_probability

score_file = (
    OUTPUT_ROOT /
    "learned_boundary_scores.csv"
)

df.to_csv(
    score_file,
    index=False
)


# ============================================================
# OVERALL ROC / PR
# ============================================================

overall_auc = roc_auc_score(
    y,
    oof_probability
)

overall_ap = average_precision_score(
    y,
    oof_probability
)

print("\n" + "=" * 70)
print("OVERALL MODEL PERFORMANCE")
print("=" * 70)

print(
    f"ROC-AUC : {overall_auc:.4f}"
)

print(
    f"PR-AUC  : {overall_ap:.4f}"
)


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS")
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

threshold_rows = []

for threshold in thresholds:

    predictions = (
        oof_probability >= threshold
    ).astype(int)

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

    tp = int(
        ((predictions == 1) & (y == 1)).sum()
    )

    fp = int(
        ((predictions == 1) & (y == 0)).sum()
    )

    fn = int(
        ((predictions == 0) & (y == 1)).sum()
    )

    candidates = int(
        predictions.sum()
    )

    print(f"\nThreshold: {threshold:.2f}")
    print(f"  Candidates : {candidates:,}")
    print(f"  TP         : {tp:,}")
    print(f"  FP         : {fp:,}")
    print(f"  FN         : {fn:,}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1         : {f1:.4f}")

    threshold_rows.append({
        "threshold": threshold,
        "candidates": candidates,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })


threshold_df = pd.DataFrame(
    threshold_rows
)

threshold_df.to_csv(
    OUTPUT_ROOT /
    "learned_boundary_thresholds.csv",
    index=False
)


# ============================================================
# SESSION-LEVEL PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("SESSION-LEVEL PERFORMANCE")
print("=" * 70)

SESSION_THRESHOLD = 0.50

session_rows = []

for sid, group in df.groupby("session_id"):

    yy = group[label_col].astype(int)

    pp = (
        group["boundary_probability"]
        >= SESSION_THRESHOLD
    ).astype(int)

    if yy.sum() == 0:
        continue

    session_rows.append({
        "session_id": sid,
        "n_rows": len(group),
        "gt_boundaries": int(yy.sum()),
        "predicted": int(pp.sum()),
        "precision": precision_score(
            yy,
            pp,
            zero_division=0
        ),
        "recall": recall_score(
            yy,
            pp,
            zero_division=0
        ),
        "f1": f1_score(
            yy,
            pp,
            zero_division=0
        ),
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
# FEATURE COEFFICIENTS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL INTERPRETABLE MODEL")
print("=" * 70)

final_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
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
    "abs_coefficient": np.abs(coefficients),
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

print("\nMost important learned features:")

for _, row in coef_df.head(20).iterrows():

    direction = (
        "increases boundary probability"
        if row["coefficient"] > 0
        else "decreases boundary probability"
    )

    print(
        f"  {row['feature']}: "
        f"{row['coefficient']:.4f} "
        f"({direction})"
    )


# ============================================================
# SAVE MODEL SUMMARY
# ============================================================

summary = {
    "phase": "5H",
    "rows": int(len(df)),
    "features": feature_cols,
    "positive_examples": int(y.sum()),
    "negative_examples": int((y == 0).sum()),
    "n_sessions": int(groups.nunique()),
    "cv_folds": N_SPLITS,
    "overall_roc_auc": float(overall_auc),
    "overall_pr_auc": float(overall_ap),
    "threshold_results": threshold_rows,
}

with open(
    OUTPUT_ROOT /
    "phase5h_summary.json",
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
print("PHASE 5H COMPLETE")
print("=" * 70)

print(
    f"ROC-AUC : {overall_auc:.4f}"
)

print(
    f"PR-AUC  : {overall_ap:.4f}"
)

print(
    f"\nOutputs saved to:\n{OUTPUT_ROOT}"
)