from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path(
    r"Outputs\Day 2\day2_boundary_vs_normal"
    r"\boundary_vs_normal_features.csv"
)

OUTPUT_DIR = Path(
    r"Outputs\Day 2\day2_boundary_detector"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

WINDOWS = [0.5, 1, 2, 3, 5, 10]


# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {

    "A_temporal": [
        "pre_event_rate",
        "post_event_rate",
        "pre_event_type_entropy",
        "post_event_type_entropy",
    ],

    "B_event_structure": [
        "pre_event_rate",
        "post_event_rate",
        "event_type_jaccard",
        "event_type_new_count",
        "event_type_disappeared_count",
        "pre_event_type_entropy",
        "post_event_type_entropy",
    ],

    "C_event_layer": [
        "pre_event_rate",
        "post_event_rate",
        "event_type_jaccard",
        "event_type_new_count",
        "event_type_disappeared_count",
        "pre_event_type_entropy",
        "post_event_type_entropy",
        "layer_jaccard",
        "layer_disappeared_count",
    ],

    "D_lean_full": [
        "pre_event_rate",
        "post_event_rate",
        "event_type_jaccard",
        "event_type_new_count",
        "event_type_disappeared_count",
        "pre_event_type_entropy",
        "post_event_type_entropy",
        "layer_jaccard",
        "layer_disappeared_count",
        "app_name_changed",
        "window_title_changed",
    ],
}


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("DAY 2 — PHASE 5A")
print("CORRECTED BOUNDARY DETECTOR")
print("=" * 70)

print("\nLoading:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print(f"\nRows: {len(df):,}")

required_columns = [
    "label",
    "session_id",
    "window_sec",
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Missing required column: {col}"
        )


# ============================================================
# BASIC SANITY CHECK
# ============================================================

print("\nWindow distribution:")

print(
    df["window_sec"]
    .value_counts()
    .sort_index()
)

print("\nLabel distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
)

print(
    f"\nSessions: "
    f"{df['session_id'].nunique()}"
)


# ============================================================
# RESULTS
# ============================================================

all_results = []
all_fold_results = []


# ============================================================
# LOOP OVER WINDOWS
# ============================================================

for window in WINDOWS:

    print("\n")
    print("=" * 70)
    print(f"WINDOW ±{window}s")
    print("=" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    # Filter rows using window_sec.
    # --------------------------------------------------------

    window_df = df[
        np.isclose(
            df["window_sec"],
            window
        )
    ].copy()

    print(
        f"Examples in window: "
        f"{len(window_df):,}"
    )

    if len(window_df) == 0:
        print("SKIPPING — no rows.")
        continue

    print(
        "Labels:",
        window_df["label"]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # Each feature set
    # --------------------------------------------------------

    for feature_set_name, features in FEATURE_SETS.items():

        missing = [
            f for f in features
            if f not in window_df.columns
        ]

        if missing:
            print(
                f"\nSkipping {feature_set_name}"
            )
            print(
                "Missing:",
                missing
            )
            continue

        print("\n" + "-" * 60)
        print(feature_set_name)
        print("-" * 60)

        X = window_df[features].copy()

        y = window_df["label"].astype(int)

        groups = window_df["session_id"]

        # ----------------------------------------------------
        # Clean numerical data
        # ----------------------------------------------------

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Median imputation using the current dataset.
        X = X.fillna(
            X.median()
        )

        X = X.fillna(0)

        # ----------------------------------------------------
        # Grouped CV
        # ----------------------------------------------------

        n_splits = min(
            5,
            groups.nunique()
        )

        cv = GroupKFold(
            n_splits=n_splits
        )

        fold_results = []

        for fold, (
            train_idx,
            test_idx
        ) in enumerate(
            cv.split(
                X,
                y,
                groups=groups
            ),
            start=1
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # ------------------------------------------------
            # Logistic regression baseline
            # ------------------------------------------------

            model = Pipeline([
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced"
                    )
                )
            ])

            model.fit(
                X_train,
                y_train
            )

            probability = model.predict_proba(
                X_test
            )[:, 1]

            prediction = (
                probability >= 0.5
            ).astype(int)

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            roc_auc = roc_auc_score(
                y_test,
                probability
            )

            pr_auc = average_precision_score(
                y_test,
                probability
            )

            precision = precision_score(
                y_test,
                prediction,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                prediction,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                prediction,
                zero_division=0
            )

            tn, fp, fn, tp = confusion_matrix(
                y_test,
                prediction,
                labels=[0, 1]
            ).ravel()

            specificity = (
                tn / (tn + fp)
                if (tn + fp) > 0
                else 0
            )

            result = {
                "window_sec": window,
                "feature_set": feature_set_name,
                "fold": fold,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "specificity": specificity,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }

            fold_results.append(result)
            all_fold_results.append(result)

            print(
                f"Fold {fold}: "
                f"ROC={roc_auc:.4f} | "
                f"PR={pr_auc:.4f} | "
                f"P={precision:.4f} | "
                f"R={recall:.4f} | "
                f"F1={f1:.4f}"
            )

        # ----------------------------------------------------
        # Aggregate folds
        # ----------------------------------------------------

        fold_df = pd.DataFrame(
            fold_results
        )

        summary = {
            "window_sec": window,
            "feature_set": feature_set_name,
            "n_features": len(features),

            "roc_auc_mean":
                fold_df["roc_auc"].mean(),

            "roc_auc_std":
                fold_df["roc_auc"].std(),

            "pr_auc_mean":
                fold_df["pr_auc"].mean(),

            "pr_auc_std":
                fold_df["pr_auc"].std(),

            "precision_mean":
                fold_df["precision"].mean(),

            "recall_mean":
                fold_df["recall"].mean(),

            "f1_mean":
                fold_df["f1"].mean(),

            "specificity_mean":
                fold_df["specificity"].mean(),
        }

        all_results.append(
            summary
        )

        print("\nAVERAGE")

        print(
            f"ROC-AUC    : "
            f"{summary['roc_auc_mean']:.4f}"
        )

        print(
            f"PR-AUC     : "
            f"{summary['pr_auc_mean']:.4f}"
        )

        print(
            f"Precision  : "
            f"{summary['precision_mean']:.4f}"
        )

        print(
            f"Recall     : "
            f"{summary['recall_mean']:.4f}"
        )

        print(
            f"F1         : "
            f"{summary['f1_mean']:.4f}"
        )

        print(
            f"Specificity: "
            f"{summary['specificity_mean']:.4f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    all_results
)

fold_df = pd.DataFrame(
    all_fold_results
)


results_file = (
    OUTPUT_DIR /
    "phase5_model_comparison.csv"
)

fold_file = (
    OUTPUT_DIR /
    "phase5_fold_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

fold_df.to_csv(
    fold_file,
    index=False
)


# ============================================================
# RANKING
# ============================================================

ranking = results_df.sort_values(
    [
        "f1_mean",
        "pr_auc_mean",
        "roc_auc_mean"
    ],
    ascending=False
)

ranking_file = (
    OUTPUT_DIR /
    "phase5_model_ranking.csv"
)

ranking.to_csv(
    ranking_file,
    index=False
)


# ============================================================
# BEST MODEL
# ============================================================

best = ranking.iloc[0].to_dict()


summary = {
    "phase": "Day 2 Phase 5A",
    "description": (
        "Session-grouped logistic regression "
        "boundary detector using correctly "
        "filtered temporal windows."
    ),
    "rows": len(df),
    "sessions": int(
        df["session_id"].nunique()
    ),
    "windows": WINDOWS,
    "best_configuration": best,
}


summary_file = (
    OUTPUT_DIR /
    "phase5_summary.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 5A — FINAL RANKING")
print("=" * 70)

print(
    ranking[
        [
            "window_sec",
            "feature_set",
            "roc_auc_mean",
            "pr_auc_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
        ]
    ].to_string(
        index=False
    )
)

print("\n")
print("=" * 70)
print("BEST CONFIGURATION")
print("=" * 70)

print(
    f"Window      : ±{best['window_sec']}s"
)

print(
    f"Features    : {best['feature_set']}"
)

print(
    f"ROC-AUC     : "
    f"{best['roc_auc_mean']:.4f}"
)

print(
    f"PR-AUC      : "
    f"{best['pr_auc_mean']:.4f}"
)

print(
    f"Precision   : "
    f"{best['precision_mean']:.4f}"
)

print(
    f"Recall      : "
    f"{best['recall_mean']:.4f}"
)

print(
    f"F1          : "
    f"{best['f1_mean']:.4f}"
)

print("\nSaved:")
print(results_file)
print(fold_file)
print(ranking_file)
print(summary_file)