import argparse
import csv
import json
import math
import importlib.util
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold


# ============================================================
# PHASE 6.7
# CORRECTED HYBRID IDENTITY CONFIDENCE ANALYSIS
#
# IMPORTANT:
# Uses the EXACT Phase 6.5 feature/session pipeline.
#
# Hybrid:
#   Sequence + Context Random Forest
#              +
#   Transition compatibility
#
# Selected transition weight:
#   1.15
# ============================================================

TRANSITION_WEIGHT = 1.15

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# LOAD THE PROVEN PHASE 6.5 IMPLEMENTATION
# ============================================================

def load_phase65_module():

    script_path = (
        Path(__file__).resolve().parent
        / "day3_sequence_context.py"
    )

    if not script_path.exists():

        raise FileNotFoundError(
            "\nCould not find the proven Phase 6.5 script:\n"
            f"{script_path}\n\n"
            "Keep day3_phase6_5_sequence_context_fixed.py "
            "in the same folder as this script."
        )

    spec = importlib.util.spec_from_file_location(
        "phase65",
        script_path
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# TRANSITION MATRIX
# TRAINING FOLD ONLY
# ============================================================

def learn_transition_matrix(
    train_rows,
    labels
):

    transition_counts = (
        defaultdict(Counter)
    )

    # --------------------------------------------------------
    # Group by session.
    # Sort by actual execution start timestamp.
    # --------------------------------------------------------

    train_df = pd.DataFrame(
        train_rows
    )

    if train_df.empty:
        return {}

    train_df["_start_sort"] = pd.to_datetime(
        train_df["start_ts"],
        utc=True,
        errors="coerce"
    )

    for session_id, group in (
        train_df.groupby(
            "session_id"
        )
    ):

        group = group.sort_values(
            "_start_sort"
        )

        sequence = list(
            group["label"]
        )

        for previous, current in zip(
            sequence[:-1],
            sequence[1:]
        ):

            transition_counts[
                previous
            ][
                current
            ] += 1

    # --------------------------------------------------------
    # Laplace smoothing
    # --------------------------------------------------------

    matrix = {}

    smoothing = 1.0

    n_classes = len(
        labels
    )

    for previous in labels:

        total = sum(
            transition_counts[
                previous
            ].values()
        )

        matrix[
            previous
        ] = {}

        for current in labels:

            count = (
                transition_counts[
                    previous
                ][
                    current
                ]
            )

            probability = (
                count + smoothing
            ) / (
                total
                +
                smoothing *
                n_classes
            )

            matrix[
                previous
            ][
                current
            ] = probability

    return matrix


# ============================================================
# HYBRID SCORING
# ============================================================

def hybrid_predictions(
    probabilities,
    classes,
    test_rows,
    transition_matrix,
    weight
):

    # --------------------------------------------------------
    # We need predictions in chronological order WITHIN each
    # session because transition context is sequential.
    #
    # Results are restored to original test-row order later.
    # --------------------------------------------------------

    indexed_rows = list(
        enumerate(test_rows)
    )

    indexed_rows.sort(
        key=lambda item: (
            item[1]["session_id"],
            pd.Timestamp(
                item[1]["start_ts"]
            )
        )
    )

    class_list = list(
        classes
    )

    results = {}

    previous_prediction = {}

    for original_index, row in (
        indexed_rows
    ):

        probability_vector = (
            probabilities[
                original_index
            ]
        )

        session_id = (
            row["session_id"]
        )

        previous = (
            previous_prediction.get(
                session_id
            )
        )

        scores = []

        # ----------------------------------------------------
        # Combine base identity probability with transition
        # probability.
        # ----------------------------------------------------

        for j, cls in enumerate(
            class_list
        ):

            base_probability = max(
                probability_vector[j],
                1e-12
            )

            score = math.log(
                base_probability
            )

            if previous is not None:

                transition_probability = max(
                    transition_matrix[
                        previous
                    ][
                        cls
                    ],
                    1e-12
                )

                score += (
                    weight
                    *
                    math.log(
                        transition_probability
                    )
                )

            scores.append(
                score
            )

        # ----------------------------------------------------
        # Convert hybrid scores into normalized probabilities.
        # ----------------------------------------------------

        score_array = np.asarray(
            scores,
            dtype=float
        )

        score_array -= np.max(
            score_array
        )

        exp_scores = np.exp(
            score_array
        )

        normalized = (
            exp_scores
            /
            np.sum(exp_scores)
        )

        ranking = np.argsort(
            normalized
        )[::-1]

        best_index = ranking[0]
        second_index = ranking[1]

        best_label = (
            class_list[
                best_index
            ]
        )

        second_label = (
            class_list[
                second_index
            ]
        )

        best_probability = float(
            normalized[
                best_index
            ]
        )

        second_probability = float(
            normalized[
                second_index
            ]
        )

        margin = (
            best_probability
            -
            second_probability
        )

        entropy = 0.0

        for p in normalized:

            if p > 0:

                entropy -= (
                    p
                    *
                    math.log2(p)
                )

        results[
            original_index
        ] = {

            "predicted_process":
                best_label,

            "second_best_process":
                second_label,

            "confidence":
                best_probability,

            "top2_probability":
                second_probability,

            "margin":
                margin,

            "entropy":
                entropy
        }

        # ----------------------------------------------------
        # IMPORTANT:
        # use PREVIOUS PREDICTION,
        # never previous GT label.
        # ----------------------------------------------------

        previous_prediction[
            session_id
        ] = best_label

    return results


# ============================================================
# CONFIDENCE BINS
# ============================================================

def make_confidence_bins(
    df
):

    bins = [
        0.0,
        0.20,
        0.40,
        0.60,
        0.70,
        0.80,
        0.90,
        1.01
    ]

    labels = [
        "0.00-0.19",
        "0.20-0.39",
        "0.40-0.59",
        "0.60-0.69",
        "0.70-0.79",
        "0.80-0.89",
        "0.90-1.00"
    ]

    temp = df.copy()

    temp[
        "confidence_bin"
    ] = pd.cut(
        temp["confidence"],
        bins=bins,
        labels=labels,
        right=False
    )

    rows = []

    for confidence_bin, group in (
        temp.groupby(
            "confidence_bin",
            observed=False
        )
    ):

        if len(group) == 0:
            continue

        rows.append({

            "confidence_bin":
                str(confidence_bin),

            "count":
                len(group),

            "accuracy":
                group[
                    "correct"
                ].mean(),

            "mean_confidence":
                group[
                    "confidence"
                ].mean(),

            "mean_margin":
                group[
                    "margin"
                ].mean()
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def make_threshold_table(
    df
):

    thresholds = [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90
    ]

    rows = []

    for threshold in thresholds:

        selected = df[
            df["confidence"]
            >= threshold
        ]

        if selected.empty:
            continue

        rows.append({

            "threshold":
                threshold,

            "assigned":
                len(selected),

            "total":
                len(df),

            "coverage":
                len(selected)
                /
                len(df),

            "accuracy":
                selected[
                    "correct"
                ].mean(),

            "errors":
                int(
                    (
                        ~selected[
                            "correct"
                        ]
                    ).sum()
                )
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# AMBIGUITY PAIRS
# ============================================================

def make_ambiguity_table(
    df
):

    counter = Counter()

    for _, row in (
        df.iterrows()
    ):

        true_label = row[
            "true_process"
        ]

        predicted_label = row[
            "predicted_process"
        ]

        if (
            true_label
            !=
            predicted_label
        ):

            counter[
                (
                    true_label,
                    predicted_label
                )
            ] += 1

    rows = []

    for (
        (true_label, predicted_label),
        count
    ) in counter.most_common():

        rows.append({

            "true_process":
                true_label,

            "predicted_process":
                predicted_label,

            "count":
                count
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-a",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("=" * 70)
    print(
        "PHASE 6.7 - "
        "CORRECTED HYBRID IDENTITY CONFIDENCE ANALYSIS"
    )
    print("=" * 70)

    print()
    print(
        f"Transition weight : "
        f"{TRANSITION_WEIGHT}"
    )

    # ========================================================
    # LOAD EXACT PHASE 6.5 PIPELINE
    # ========================================================

    phase65 = (
        load_phase65_module()
    )

    dataset = Path(
        args.dataset_a
    )

    # --------------------------------------------------------
    # EXACT same manifest loading
    # --------------------------------------------------------

    manifests = (
        phase65.load_manifest(
            dataset
        )
    )

    # --------------------------------------------------------
    # EXACT same raw event loading
    # --------------------------------------------------------

    events_by_session = (
        phase65.load_events(
            dataset
        )
    )

    manifest_total = sum(
        len(executions)
        for _, executions
        in manifests
    )

    print()
    print(
        "VALIDATED DATA PIPELINE"
    )

    print(
        f"Manifest executions found : "
        f"{manifest_total}"
    )

    print(
        f"Sessions with raw events  : "
        f"{len(events_by_session)}"
    )

    # --------------------------------------------------------
    # EXACT same vocabulary
    # --------------------------------------------------------

    vocab = (
        phase65.collect_vocab(
            events_by_session
        )
    )

    # --------------------------------------------------------
    # EXACT same feature construction
    # --------------------------------------------------------

    rows, diagnostics = (
        phase65.build_feature_dataset(
            manifests,
            events_by_session,
            vocab
        )
    )

    print()
    print(
        "FEATURE EXTRACTION DIAGNOSTICS"
    )

    for key, value in (
        diagnostics.items()
    ):

        print(
            f"{key:30s}: "
            f"{value}"
        )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    if len(rows) < 1600:

        print()
        print("=" * 70)
        print(
            "ERROR: TOO MANY EXECUTIONS LOST"
        )
        print("=" * 70)

        print(
            f"Expected approximately 1734 usable executions, "
            f"but only {len(rows)} were obtained."
        )

        print(
            "\nDO NOT use this result."
        )

        return

    # ========================================================
    # ADD TIMING INFORMATION
    #
    # Phase 6.5 intentionally keeps feature rows compact.
    # We reconstruct the timing lookup from the SAME manifests.
    # ========================================================

    timing_lookup = {}

    for manifest_path, executions in (
        manifests
    ):

        session_id = (
            manifest_path.parent.name
        )

        for execution_index, execution in (
            enumerate(executions)
        ):

            label = execution.get(
                "process_code"
            )

            key = (
                session_id,
                label,
                execution_index
            )

            timing_lookup[
                key
            ] = {

                "start_ts":
                    execution.get(
                        "start_ts"
                    ),

                "end_ts":
                    execution.get(
                        "end_ts"
                    ),

                "case_id":
                    execution.get(
                        "case_id"
                    )
            }

    enriched_rows = []

    for row in rows:

        key = (
            row["session_id"],
            row["label"],
            row["execution_index"]
        )

        timing = timing_lookup.get(
            key,
            {}
        )

        new_row = dict(
            row
        )

        new_row[
            "start_ts"
        ] = timing.get(
            "start_ts"
        )

        new_row[
            "end_ts"
        ] = timing.get(
            "end_ts"
        )

        new_row[
            "case_id"
        ] = timing.get(
            "case_id"
        )

        enriched_rows.append(
            new_row
        )

    rows = enriched_rows

    # ========================================================
    # FEATURES
    # EXACT SAME COMBINED FEATURES AS PHASE 6.5
    # ========================================================

    excluded = {
        "session_id",
        "execution_index",
        "label",
        "variant",
        "start_ts",
        "end_ts",
        "case_id"
    }

    feature_names = sorted({

        key

        for row in rows

        for key, value
        in row.items()

        if (
            key not in excluded
            and
            isinstance(
                value,
                (int, float)
            )
        )
    })

    sequence_features = [

        feature

        for feature in feature_names

        if (
            "ngram" in feature
            or
            "transition" in feature
            or
            "motif__" in feature
            or
            feature in {

                "median_event_gap",
                "p90_event_gap",
                "max_event_gap",
                "first_event_hash",
                "last_event_hash",
                "first_app_hash",
                "last_app_hash"
            }
        )
    ]

    context_features = [

        feature

        for feature in feature_names

        if (
            feature.startswith(
                "app_ngram_"
            )
            or
            feature.startswith(
                "layer_ngram_"
            )
            or
            "app_transition" in feature
            or
            "layer_transition" in feature
        )
    ]

    combined_features = sorted(
        set(
            sequence_features
            +
            context_features
        )
    )

    # ========================================================
    # DATASET SUMMARY
    # ========================================================

    labels = sorted({

        row["label"]

        for row in rows
    })

    sessions = sorted({

        row["session_id"]

        for row in rows
    })

    print()
    print("=" * 70)
    print(
        "PHASE 6.7 DATASET"
    )
    print("=" * 70)

    print(
        f"Complete timed executions : "
        f"{len(rows)}"
    )

    print(
        f"Process classes            : "
        f"{len(labels)}"
    )

    print(
        f"Session groups             : "
        f"{len(sessions)}"
    )

    print(
        f"Total numeric features     : "
        f"{len(feature_names)}"
    )

    print(
        f"Sequence/context features : "
        f"{len(combined_features)}"
    )

    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    X = np.asarray([

        [
            float(
                row.get(
                    feature,
                    0.0
                )
            )

            for feature
            in combined_features
        ]

        for row in rows

    ], dtype=float)

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    y = np.asarray([
        row["label"]
        for row in rows
    ])

    groups = np.asarray([
        row["session_id"]
        for row in rows
    ])

    n_splits = min(
        args.folds,
        len(
            set(groups)
        )
    )

    splitter = GroupKFold(
        n_splits=n_splits
    )

    all_results = []

    fold_metrics = []

    # ========================================================
    # FOLDS
    # ========================================================

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        splitter.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        print()
        print(
            f"Fold {fold}/{n_splits}"
        )

        X_train = X[
            train_idx
        ]

        X_test = X[
            test_idx
        ]

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        train_rows = [
            rows[i]
            for i in train_idx
        ]

        test_rows = [
            rows[i]
            for i in test_idx
        ]

        # ----------------------------------------------------
        # EXACT RF CONFIGURATION FROM PHASE 6.5
        # ----------------------------------------------------

        model = RandomForestClassifier(

            n_estimators=500,

            random_state=42,

            n_jobs=-1,

            class_weight=
                "balanced_subsample",

            min_samples_leaf=2
        )

        model.fit(
            X_train,
            y_train
        )

        probabilities = (
            model.predict_proba(
                X_test
            )
        )

        base_predictions = (
            model.predict(
                X_test
            )
        )

        model_classes = list(
            model.classes_
        )

        # ----------------------------------------------------
        # TRAINING-FOLD TRANSITIONS ONLY
        # ----------------------------------------------------

        transition_matrix = (
            learn_transition_matrix(
                train_rows,
                labels
            )
        )

        # ----------------------------------------------------
        # HYBRID
        # ----------------------------------------------------

        hybrid = (
            hybrid_predictions(
                probabilities,
                model_classes,
                test_rows,
                transition_matrix,
                TRANSITION_WEIGHT
            )
        )

        hybrid_predictions_list = [

            hybrid[i][
                "predicted_process"
            ]

            for i in range(
                len(test_rows)
            )
        ]

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        base_accuracy = (
            accuracy_score(
                y_test,
                base_predictions
            )
        )

        base_macro_f1 = (
            f1_score(
                y_test,
                base_predictions,
                average="macro",
                zero_division=0
            )
        )

        hybrid_accuracy = (
            accuracy_score(
                y_test,
                hybrid_predictions_list
            )
        )

        hybrid_macro_f1 = (
            f1_score(
                y_test,
                hybrid_predictions_list,
                average="macro",
                zero_division=0
            )
        )

        print(
            f"Base RF              "
            f"Acc={base_accuracy:.4f} "
            f"MacroF1={base_macro_f1:.4f}"
        )

        print(
            f"Hybrid RF + Trans.   "
            f"Acc={hybrid_accuracy:.4f} "
            f"MacroF1={hybrid_macro_f1:.4f}"
        )

        fold_metrics.append({

            "fold":
                fold,

            "base_accuracy":
                base_accuracy,

            "base_macro_f1":
                base_macro_f1,

            "hybrid_accuracy":
                hybrid_accuracy,

            "hybrid_macro_f1":
                hybrid_macro_f1
        })

        # ----------------------------------------------------
        # EXECUTION CONFIDENCE
        # ----------------------------------------------------

        for local_index, row in enumerate(
            test_rows
        ):

            prediction = hybrid[
                local_index
            ]

            true_label = row[
                "label"
            ]

            predicted_label = (
                prediction[
                    "predicted_process"
                ]
            )

            all_results.append({

                "fold":
                    fold,

                "session_id":
                    row[
                        "session_id"
                    ],

                "start_ts":
                    row[
                        "start_ts"
                    ],

                "end_ts":
                    row[
                        "end_ts"
                    ],

                "case_id":
                    row[
                        "case_id"
                    ],

                "true_process":
                    true_label,

                "predicted_process":
                    predicted_label,

                "second_best_process":
                    prediction[
                        "second_best_process"
                    ],

                "confidence":
                    prediction[
                        "confidence"
                    ],

                "top2_probability":
                    prediction[
                        "top2_probability"
                    ],

                "margin":
                    prediction[
                        "margin"
                    ],

                "entropy":
                    prediction[
                        "entropy"
                    ],

                "correct":
                    bool(
                        true_label
                        ==
                        predicted_label
                    )
            })

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        output_dir /
        "phase6_7_execution_confidence.csv",
        index=False
    )

    fold_df = pd.DataFrame(
        fold_metrics
    )

    fold_df.to_csv(
        output_dir /
        "phase6_7_fold_metrics.csv",
        index=False
    )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    base_accuracy = (
        fold_df[
            "base_accuracy"
        ].mean()
    )

    base_macro_f1 = (
        fold_df[
            "base_macro_f1"
        ].mean()
    )

    hybrid_accuracy = (
        fold_df[
            "hybrid_accuracy"
        ].mean()
    )

    hybrid_macro_f1 = (
        fold_df[
            "hybrid_macro_f1"
        ].mean()
    )

    print()
    print("=" * 70)
    print(
        "OVERALL HYBRID RESULTS"
    )
    print("=" * 70)

    print()
    print("BASE MODEL")

    print(
        f"Accuracy    : "
        f"{base_accuracy:.4f}"
    )

    print(
        f"Macro F1    : "
        f"{base_macro_f1:.4f}"
    )

    print()
    print("HYBRID MODEL")

    print(
        f"Accuracy    : "
        f"{hybrid_accuracy:.4f}"
    )

    print(
        f"Macro F1    : "
        f"{hybrid_macro_f1:.4f}"
    )

    print()
    print(
        f"Accuracy delta : "
        f"{hybrid_accuracy - base_accuracy:+.4f}"
    )

    print(
        f"Macro F1 delta : "
        f"{hybrid_macro_f1 - base_macro_f1:+.4f}"
    )

    # ========================================================
    # CONFIDENCE BINS
    # ========================================================

    bins_df = (
        make_confidence_bins(
            results_df
        )
    )

    bins_df.to_csv(
        output_dir /
        "phase6_7_confidence_bins.csv",
        index=False
    )

    print()
    print("=" * 70)
    print(
        "CONFIDENCE BIN ANALYSIS"
    )
    print("=" * 70)

    print(
        bins_df.to_string(
            index=False
        )
    )

    # ========================================================
    # THRESHOLDS
    # ========================================================

    threshold_df = (
        make_threshold_table(
            results_df
        )
    )

    threshold_df.to_csv(
        output_dir /
        "phase6_7_confidence_thresholds.csv",
        index=False
    )

    print()
    print("=" * 70)
    print(
        "CONFIDENCE THRESHOLD ANALYSIS"
    )
    print("=" * 70)

    print(
        threshold_df.to_string(
            index=False
        )
    )

    # ========================================================
    # AMBIGUITY
    # ========================================================

    ambiguity_df = (
        make_ambiguity_table(
            results_df
        )
    )

    ambiguity_df.to_csv(
        output_dir /
        "phase6_7_ambiguity_pairs.csv",
        index=False
    )

    print()
    print("=" * 70)
    print(
        "TOP AMBIGUITY PAIRS"
    )
    print("=" * 70)

    if ambiguity_df.empty:

        print(
            "No incorrect predictions."
        )

    else:

        print(
            ambiguity_df.head(
                20
            ).to_string(
                index=False
            )
        )

    # ========================================================
    # CONFIDENCE QUALITY
    # ========================================================

    correct_rows = results_df[
        results_df[
            "correct"
        ]
    ]

    incorrect_rows = results_df[
        ~results_df[
            "correct"
        ]
    ]

    correct_mean_confidence = (
        correct_rows[
            "confidence"
        ].mean()
    )

    incorrect_mean_confidence = (
        incorrect_rows[
            "confidence"
        ].mean()
    )

    correct_mean_margin = (
        correct_rows[
            "margin"
        ].mean()
    )

    incorrect_mean_margin = (
        incorrect_rows[
            "margin"
        ].mean()
    )

    print()
    print("=" * 70)
    print(
        "CONFIDENCE QUALITY"
    )
    print("=" * 70)

    print()
    print(
        "Correct predictions"
    )

    print(
        f"Mean confidence : "
        f"{correct_mean_confidence:.4f}"
    )

    print(
        f"Mean margin     : "
        f"{correct_mean_margin:.4f}"
    )

    print()
    print(
        "Incorrect predictions"
    )

    print(
        f"Mean confidence : "
        f"{incorrect_mean_confidence:.4f}"
    )

    print(
        f"Mean margin     : "
        f"{incorrect_mean_margin:.4f}"
    )

    # ========================================================
    # HIGH CONFIDENCE
    # ========================================================

    high_threshold = 0.80

    high_confidence = results_df[
        results_df[
            "confidence"
        ]
        >=
        high_threshold
    ]

    high_errors = (
        high_confidence[
            ~high_confidence[
                "correct"
            ]
        ]
    )

    print()
    print("=" * 70)
    print(
        "HIGH-CONFIDENCE ERROR CHECK"
    )
    print("=" * 70)

    print()
    print(
        f"Threshold : "
        f"{high_threshold}"
    )

    print(
        f"High-confidence executions : "
        f"{len(high_confidence)}"
    )

    print(
        f"High-confidence errors     : "
        f"{len(high_errors)}"
    )

    if len(high_confidence) > 0:

        print(
            f"High-confidence accuracy   : "
            f"{high_confidence['correct'].mean():.4f}"
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {

        "phase":
            "6.7",

        "transition_weight":
            TRANSITION_WEIGHT,

        "manifest_executions":
            int(manifest_total),

        "usable_executions":
            int(len(rows)),

        "process_classes":
            int(len(labels)),

        "session_groups":
            int(len(sessions)),

        "feature_count":
            int(len(feature_names)),

        "combined_feature_count":
            int(len(combined_features)),

        "base_accuracy":
            float(base_accuracy),

        "base_macro_f1":
            float(base_macro_f1),

        "hybrid_accuracy":
            float(hybrid_accuracy),

        "hybrid_macro_f1":
            float(hybrid_macro_f1),

        "accuracy_delta":
            float(
                hybrid_accuracy
                -
                base_accuracy
            ),

        "macro_f1_delta":
            float(
                hybrid_macro_f1
                -
                base_macro_f1
            ),

        "correct_mean_confidence":
            float(
                correct_mean_confidence
            ),

        "incorrect_mean_confidence":
            float(
                incorrect_mean_confidence
            ),

        "correct_mean_margin":
            float(
                correct_mean_margin
            ),

        "incorrect_mean_margin":
            float(
                incorrect_mean_margin
            ),

        "high_confidence_threshold":
            high_threshold,

        "high_confidence_count":
            int(
                len(
                    high_confidence
                )
            ),

        "high_confidence_errors":
            int(
                len(
                    high_errors
                )
            )
    }

    with open(
        output_dir /
        "phase6_7_confidence_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "PHASE 6.7 COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Outputs saved to:"
    )

    print(
        output_dir
    )

    print()
    print(
        "Key file:"
    )

    print(
        "phase6_7_execution_confidence.csv"
    )


if __name__ == "__main__":
    main()