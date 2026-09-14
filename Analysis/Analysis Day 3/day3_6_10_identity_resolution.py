import argparse
import importlib.util
import json
import math
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# PHASE 6.10
#
# IDENTITY RESOLUTION MODEL SELECTION
#
# Four approaches are evaluated against Dataset A GT:
#
#   1. RF-only
#   2. RF + local-neighbor evidence
#   3. RF + process-level consistency
#   4. Combined RF + local + process consistency
#
# The winning approach is selected ONLY from Dataset A GT
# validation.
#
# IMPORTANT:
#   - Phase 6.5 feature pipeline is reused exactly.
#   - Phase 6.8 boundaries are preserved.
#   - Phase 6.9 RF configuration is preserved.
#   - No transition hybrid is used.
#   - Neighbor evidence is soft evidence, never a hard rule.
#   - A-B-A patterns remain possible.
#   - Repeated process executions remain possible.
# ============================================================


RANDOM_STATE = 42

DEFAULT_LOCAL_WEIGHT = 0.20
DEFAULT_PROCESS_WEIGHT = 0.20

DEFAULT_MIN_GT_OVERLAP = 0.50

EPS = 1e-12


# ============================================================
# PHASE 6.5 LOADER
# ============================================================

def load_phase65_module():

    here = Path(__file__).resolve().parent

    candidates = [
        here / "day3_sequence_context.py",
        here / "day3_phase6_5_sequence_context_fixed.py",
        here / "day3_phase6_5_sequence_context.py",
    ]

    for script_path in candidates:

        if script_path.exists():

            spec = importlib.util.spec_from_file_location(
                "phase65",
                script_path,
            )

            if spec is None or spec.loader is None:
                continue

            module = importlib.util.module_from_spec(spec)

            spec.loader.exec_module(module)

            return module

    raise FileNotFoundError(
        "Could not find the proven Phase 6.5 sequence/context "
        "script in the same directory as this script."
    )


# ============================================================
# JSONL
# ============================================================

def load_jsonl(path):

    rows = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return rows


def save_jsonl(path, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            clean = {}

            for key, value in row.items():

                if hasattr(value, "isoformat"):

                    clean[key] = value.isoformat()

                elif isinstance(value, np.generic):

                    clean[key] = value.item()

                elif isinstance(value, np.ndarray):

                    clean[key] = value.tolist()

                else:

                    clean[key] = value

            f.write(
                json.dumps(
                    clean,
                    ensure_ascii=False,
                ) + "\n"
            )


# ============================================================
# TIMESTAMP
# ============================================================

def parse_ts_ms(value):

    if value is None:
        return None

    try:

        ts = pd.Timestamp(value)

        if ts.tzinfo is None:

            ts = ts.tz_localize("UTC")

        else:

            ts = ts.tz_convert("UTC")

        return float(
            ts.timestamp() * 1000.0
        )

    except Exception:

        return None


# ============================================================
# EXACT FEATURE SELECTION FROM PHASE 6.9
# ============================================================

def get_training_feature_names(rows):

    excluded = {
        "session_id",
        "execution_index",
        "label",
        "variant",
        "start_ts",
        "end_ts",
        "case_id",
    }

    feature_names = sorted({

        key

        for row in rows

        for key, value in row.items()

        if (
            key not in excluded
            and isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            )
        )
    })

    return feature_names


def get_combined_features(feature_names):

    sequence_features = [

        feature

        for feature in feature_names

        if (
            "ngram" in feature
            or "transition" in feature
            or "motif__" in feature
            or feature in {
                "median_event_gap",
                "p90_event_gap",
                "max_event_gap",
                "first_event_hash",
                "last_event_hash",
                "first_app_hash",
                "last_app_hash",
            }
        )
    ]

    context_features = [

        feature

        for feature in feature_names

        if (
            feature.startswith("app_ngram_")
            or feature.startswith("layer_ngram_")
            or "app_transition" in feature
            or "layer_transition" in feature
        )
    ]

    return sorted(
        set(
            sequence_features
            + context_features
        )
    )


# ============================================================
# BUILD TRAINING MATRIX
#
# EXACT DATA TYPE:
#   training_rows = list[dict]
#
# NOT a DataFrame.
# ============================================================

def build_training_matrix(
    training_rows,
    combined_features,
):

    X = np.asarray(

        [
            [
                float(
                    row.get(
                        feature,
                        0.0,
                    )
                )

                for feature in combined_features
            ]

            for row in training_rows
        ],

        dtype=float,
    )

    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    y = np.asarray(
        [
            row["label"]
            for row in training_rows
        ]
    )

    groups = np.asarray(
        [
            row["session_id"]
            for row in training_rows
        ]
    )

    return X, y, groups


# ============================================================
# BUILD SEGMENT FEATURES
#
# Uses EXACT Phase 6.5 make_features().
# This is the same mechanism used by the proven 6.9 script.
# ============================================================

def build_segment_features(
    phase65,
    segments,
    events_by_session,
    vocab,
):

    rows = []

    diagnostics = defaultdict(int)

    for index, segment in enumerate(segments):

        session_id = segment.get(
            "session_id"
        )

        start_ms = parse_ts_ms(
            segment.get("start")
        )

        end_ms = parse_ts_ms(
            segment.get("end")
        )

        if (
            session_id is None
            or start_ms is None
            or end_ms is None
        ):

            diagnostics[
                "invalid_segment"
            ] += 1

            continue

        if end_ms <= start_ms:

            diagnostics[
                "invalid_interval"
            ] += 1

            continue

        events = events_by_session.get(
            session_id,
            [],
        )

        if not events:

            diagnostics[
                "missing_session_events"
            ] += 1

            continue

        timestamp_array = np.asarray(
            [
                event["_ts_ms"]
                for event in events
            ],
            dtype=float,
        )

        features = phase65.make_features(
            events,
            timestamp_array,
            start_ms,
            end_ms,
            vocab,
        )

        if features is None:

            diagnostics[
                "no_events_inside_segment"
            ] += 1

            continue

        row = dict(features)

        row["segment_index"] = index
        row["session_id"] = session_id
        row["start"] = segment.get("start")
        row["end"] = segment.get("end")

        rows.append(row)

    diagnostics[
        "segments_total"
    ] = len(segments)

    diagnostics[
        "usable_segments"
    ] = len(rows)

    return rows, dict(diagnostics)


# ============================================================
# BUILD MATRIX FOR ARBITRARY SEGMENT ROWS
# ============================================================

def build_feature_matrix(
    rows,
    feature_names,
):

    X = np.asarray(

        [
            [
                float(
                    row.get(
                        feature,
                        0.0,
                    )
                )

                for feature in feature_names
            ]

            for row in rows
        ],

        dtype=float,
    )

    return np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


# ============================================================
# GT MANIFEST
# ============================================================

def load_gt_windows(dataset):

    windows = []

    for manifest_path in sorted(
        Path(dataset).rglob(
            "gt_manifest.json"
        )
    ):

        try:

            with open(
                manifest_path,
                "r",
                encoding="utf-8",
            ) as f:

                manifest = json.load(f)

        except Exception:

            continue

        session_id = (
            manifest
            .get("session", {})
            .get("session_id")
        )

        if not session_id:

            session_id = manifest_path.parent.name

        for process in manifest.get(
            "processes",
            [],
        ):

            label = process.get(
                "code"
            )

            family_name = process.get(
                "family_name"
            )

            for execution in process.get(
                "executions",
                [],
            ):

                start_ms = parse_ts_ms(
                    execution.get(
                        "start_ts"
                    )
                )

                end_ms = parse_ts_ms(
                    execution.get(
                        "end_ts"
                    )
                )

                if (
                    start_ms is None
                    or end_ms is None
                    or end_ms <= start_ms
                ):

                    continue

                windows.append({

                    "session_id": session_id,

                    "start_ms": start_ms,

                    "end_ms": end_ms,

                    "label": label,

                    "family_name": family_name,

                    "variant": execution.get(
                        "variant"
                    ),

                    "case_id": execution.get(
                        "case_id"
                    ),
                })

    return windows


# ============================================================
# MATCH EACH PREDICTED SEGMENT TO BEST GT EXECUTION
# ============================================================

def best_gt_for_segment(
    segment,
    gt_windows,
):

    start_ms = parse_ts_ms(
        segment.get("start")
    )

    end_ms = parse_ts_ms(
        segment.get("end")
    )

    if (
        start_ms is None
        or end_ms is None
        or end_ms <= start_ms
    ):

        return None, 0.0

    session_id = segment.get(
        "session_id"
    )

    best = None
    best_ratio = 0.0

    for gt in gt_windows:

        if gt["session_id"] != session_id:
            continue

        overlap_start = max(
            start_ms,
            gt["start_ms"],
        )

        overlap_end = min(
            end_ms,
            gt["end_ms"],
        )

        if overlap_end <= overlap_start:
            continue

        overlap = (
            overlap_end
            - overlap_start
        )

        gt_duration = (
            gt["end_ms"]
            - gt["start_ms"]
        )

        if gt_duration <= 0:
            continue

        ratio = (
            overlap
            / gt_duration
        )

        if ratio > best_ratio:

            best_ratio = ratio
            best = gt

    return best, best_ratio


# ============================================================
# ATTACH GT TO SEGMENTS
# ============================================================

def attach_gt(
    segment_rows,
    gt_windows,
):

    output = []

    for row in segment_rows:

        gt, ratio = best_gt_for_segment(
            row,
            gt_windows,
        )

        result = dict(row)

        if gt is None:

            result["gt_process"] = None
            result["gt_family"] = None
            result["gt_overlap"] = 0.0
            result["gt_case_id"] = None
            result["gt_variant"] = None

        else:

            result["gt_process"] = (
                gt["label"]
            )

            result["gt_family"] = (
                gt["family_name"]
            )

            result["gt_overlap"] = float(
                ratio
            )

            result["gt_case_id"] = (
                gt["case_id"]
            )

            result["gt_variant"] = (
                gt["variant"]
            )

        output.append(result)

    return output


# ============================================================
# PROCESS PROTOTYPES
#
# Learn process-level behavioral signatures from the exact
# Phase 6.5 training rows.
#
# Prototype = mean feature vector per process.
# ============================================================

def build_process_prototypes(
    X_train,
    y_train,
):

    labels = sorted(
        set(y_train)
    )

    prototypes = {}

    for label in labels:

        mask = (
            y_train == label
        )

        class_matrix = X_train[
            mask
        ]

        if len(class_matrix) == 0:
            continue

        prototypes[label] = np.mean(
            class_matrix,
            axis=0,
        )

    return prototypes


# ============================================================
# STANDARDIZATION FOR PROTOTYPE SIMILARITY
#
# Scaling is learned ONLY from Dataset A training rows.
# ============================================================

def fit_feature_scaler(X_train):

    mean = np.mean(
        X_train,
        axis=0,
    )

    std = np.std(
        X_train,
        axis=0,
    )

    std = np.where(
        std < EPS,
        1.0,
        std,
    )

    return mean, std


def standardize(
    X,
    mean,
    std,
):

    return (
        X - mean
    ) / std


# ============================================================
# PROCESS PROTOTYPE SCORES
#
# Cosine similarity -> [0,1]
#
# This is process-level evidence, not a hard label rule.
# ============================================================

def compute_process_scores(
    X,
    prototypes,
    scaler_mean,
    scaler_std,
    labels,
):

    X_scaled = standardize(
        X,
        scaler_mean,
        scaler_std,
    )

    scores = np.zeros(
        (
            len(X),
            len(labels),
        ),
        dtype=float,
    )

    for j, label in enumerate(
        labels
    ):

        prototype = prototypes.get(
            label
        )

        if prototype is None:
            continue

        prototype_scaled = standardize(
            prototype.reshape(1, -1),
            scaler_mean,
            scaler_std,
        )[0]

        prototype_norm = np.linalg.norm(
            prototype_scaled
        )

        if prototype_norm < EPS:
            continue

        x_norm = np.linalg.norm(
            X_scaled,
            axis=1,
        )

        denominator = (
            x_norm
            * prototype_norm
        )

        numerator = (
            X_scaled
            @ prototype_scaled
        )

        cosine = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(
                numerator,
                dtype=float,
            ),
            where=denominator > EPS,
        )

        # cosine is [-1,1]
        # convert to [0,1]
        scores[:, j] = (
            cosine + 1.0
        ) / 2.0

    # Normalize row-wise so the process evidence becomes
    # comparable with RF probabilities.
    row_sum = scores.sum(
        axis=1,
        keepdims=True,
    )

    scores = np.divide(
        scores,
        row_sum,
        out=np.full_like(
            scores,
            1.0 / max(
                len(labels),
                1,
            ),
        ),
        where=row_sum > EPS,
    )

    return scores


# ============================================================
# LOCAL NEIGHBOR EVIDENCE
#
# Uses soft probability distributions from adjacent segments.
#
# CRITICAL:
#   This does NOT force neighbors to share a label.
#
# Example:
#   A -> B -> A remains possible.
# ============================================================

def compute_local_scores(
    segment_rows,
    rf_probabilities,
    classes,
):

    n = len(segment_rows)
    k = len(classes)

    local_scores = np.zeros(
        (n, k),
        dtype=float,
    )

    grouped = defaultdict(list)

    for i, row in enumerate(
        segment_rows
    ):

        grouped[
            row["session_id"]
        ].append(i)

    class_to_index = {
        str(label): i
        for i, label in enumerate(
            classes
        )
    }

    for session_id, indices in grouped.items():

        indices.sort(
            key=lambda i: (
                parse_ts_ms(
                    segment_rows[i]["start"]
                ),
                parse_ts_ms(
                    segment_rows[i]["end"]
                ),
                segment_rows[i][
                    "segment_index"
                ],
            )
        )

        for position, current_index in enumerate(
            indices
        ):

            neighbor_distributions = []

            # Previous segment
            if position > 0:

                previous_index = (
                    indices[position - 1]
                )

                neighbor_distributions.append(
                    rf_probabilities[
                        previous_index
                    ]
                )

            # Next segment
            if position + 1 < len(indices):

                next_index = (
                    indices[position + 1]
                )

                neighbor_distributions.append(
                    rf_probabilities[
                        next_index
                    ]
                )

            if not neighbor_distributions:

                local_scores[
                    current_index
                ] = (
                    1.0 / max(k, 1)
                )

                continue

            # ------------------------------------------------
            # SOFT averaging.
            #
            # This means A-B-A is NOT collapsed into A-A-A.
            # ------------------------------------------------

            local_distribution = np.mean(
                np.asarray(
                    neighbor_distributions
                ),
                axis=0,
            )

            total = local_distribution.sum()

            if total > EPS:

                local_distribution = (
                    local_distribution
                    / total
                )

            else:

                local_distribution[:] = (
                    1.0 / max(k, 1)
                )

            local_scores[
                current_index
            ] = local_distribution

    return local_scores


# ============================================================
# PROBABILITY NORMALIZATION
# ============================================================

def normalize_rows(matrix):

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    matrix = np.nan_to_num(
        matrix,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    matrix = np.maximum(
        matrix,
        0.0,
    )

    row_sum = matrix.sum(
        axis=1,
        keepdims=True,
    )

    return np.divide(
        matrix,
        row_sum,
        out=np.full_like(
            matrix,
            1.0 / max(
                matrix.shape[1],
                1,
            ),
        ),
        where=row_sum > EPS,
    )


# ============================================================
# COMBINE EVIDENCE
# ============================================================

def combine_probabilities(
    rf_probabilities,
    local_scores,
    process_scores,
    local_weight,
    process_weight,
):

    rf = normalize_rows(
        rf_probabilities
    )

    local = normalize_rows(
        local_scores
    )

    process = normalize_rows(
        process_scores
    )

    # RF receives the remaining weight.
    rf_weight = max(
        0.0,
        1.0
        - local_weight
        - process_weight,
    )

    combined = (
        rf_weight * rf
        + local_weight * local
        + process_weight * process
    )

    return normalize_rows(
        combined
    )


# ============================================================
# PREDICTION + CONFIDENCE
# ============================================================

def prediction_from_probabilities(
    probabilities,
    classes,
):

    predictions = []
    confidence = []
    margins = []
    second_best = []

    for row in probabilities:

        order = np.argsort(
            row
        )[::-1]

        best = int(
            order[0]
        )

        second = (
            int(order[1])
            if len(order) > 1
            else best
        )

        best_probability = float(
            row[best]
        )

        second_probability = float(
            row[second]
        )

        predictions.append(
            str(classes[best])
        )

        confidence.append(
            best_probability
        )

        margins.append(
            best_probability
            - second_probability
        )

        second_best.append(
            str(classes[second])
        )

    return (
        predictions,
        np.asarray(
            confidence,
            dtype=float,
        ),
        np.asarray(
            margins,
            dtype=float,
        ),
        second_best,
    )


# ============================================================
# METHOD EVALUATION
# ============================================================

def evaluate_method(
    name,
    probabilities,
    segment_rows,
    min_gt_overlap,
):

    predictions, confidence, margins, second_best = (
        prediction_from_probabilities(
            probabilities,
            np.asarray(
                sorted(
                    set(
                        row["gt_process"]
                        for row in segment_rows
                        if row.get(
                            "gt_process"
                        ) is not None
                    )
                )
            ),
        )
    )

    # The above class list must not be reconstructed from GT
    # because model classes can differ. This function is
    # replaced below by evaluate_method_with_classes().
    raise RuntimeError(
        "Internal evaluation dispatch error."
    )


def evaluate_method_with_classes(
    name,
    probabilities,
    classes,
    segment_rows,
    min_gt_overlap,
):

    predictions, confidence, margins, second_best = (
        prediction_from_probabilities(
            probabilities,
            classes,
        )
    )

    total = len(segment_rows)

    gt_evaluable_mask = np.asarray(
        [
            (
                row.get("gt_process")
                is not None
                and float(
                    row.get(
                        "gt_overlap",
                        0.0,
                    )
                ) >= min_gt_overlap
            )

            for row in segment_rows
        ],
        dtype=bool,
    )

    evaluable_count = int(
        gt_evaluable_mask.sum()
    )

    coverage = (
        evaluable_count / total
        if total
        else 0.0
    )

    if evaluable_count > 0:

        y_true = np.asarray(
            [
                row["gt_process"]
                for i, row in enumerate(
                    segment_rows
                )
                if gt_evaluable_mask[i]
            ]
        )

        y_pred = np.asarray(
            [
                predictions[i]
                for i in range(total)
                if gt_evaluable_mask[i]
            ]
        )

        accuracy = float(
            accuracy_score(
                y_true,
                y_pred,
            )
        )

        macro_f1 = float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        )

    else:

        accuracy = 0.0
        macro_f1 = 0.0

    overlap_50 = int(
        sum(
            1
            for row in segment_rows
            if float(
                row.get(
                    "gt_overlap",
                    0.0,
                )
            ) >= 0.50
        )
    )

    overlap_80 = int(
        sum(
            1
            for row in segment_rows
            if float(
                row.get(
                    "gt_overlap",
                    0.0,
                )
            ) >= 0.80
        )
    )

    mean_overlap = float(
        np.mean(
            [
                float(
                    row.get(
                        "gt_overlap",
                        0.0,
                    )
                )

                for row in segment_rows
            ]
        )
    )

    return {

        "method": name,

        "predictions": predictions,

        "confidence": confidence,

        "margins": margins,

        "second_best": second_best,

        "accuracy": accuracy,

        "macro_f1": macro_f1,

        "coverage": coverage,

        "evaluable_count": evaluable_count,

        "total_segments": total,

        "mean_confidence": float(
            np.mean(confidence)
        ),

        "median_confidence": float(
            np.median(confidence)
        ),

        "mean_margin": float(
            np.mean(margins)
        ),

        "overlap_50_count": overlap_50,

        "overlap_50_rate": (
            overlap_50 / total
            if total
            else 0.0
        ),

        "overlap_80_count": overlap_80,

        "overlap_80_rate": (
            overlap_80 / total
            if total
            else 0.0
        ),

        "mean_gt_overlap": mean_overlap,
    }


# ============================================================
# WINNER SELECTION
#
# Primary criterion:
#   Macro-F1
#
# Secondary:
#   Accuracy
#
# Tertiary:
#   Coverage
#
# Coverage is identical for the four methods because they all
# operate on the same Phase 6.8 segments. It is reported but
# cannot artificially make one method win.
# ============================================================

def select_winner(results):

    ranked = sorted(
        results,
        key=lambda r: (
            r["macro_f1"],
            r["accuracy"],
            r["coverage"],
            r["mean_gt_overlap"],
        ),
        reverse=True,
    )

    return ranked[0], ranked


# ============================================================
# BUILD ASSIGNMENT OUTPUT
# ============================================================

def build_assignments(
    segment_rows,
    winning_result,
):

    predictions = winning_result[
        "predictions"
    ]

    confidence = winning_result[
        "confidence"
    ]

    margins = winning_result[
        "margins"
    ]

    second_best = winning_result[
        "second_best"
    ]

    assignments = []

    for i, row in enumerate(
        segment_rows
    ):

        assignments.append({

            "segment_index": int(
                row["segment_index"]
            ),

            "session_id": row[
                "session_id"
            ],

            "start": row[
                "start"
            ],

            "end": row[
                "end"
            ],

            "label": predictions[i],

            "identity_confidence": float(
                confidence[i]
            ),

            "identity_margin": float(
                margins[i]
            ),

            "second_best_process": (
                second_best[i]
            ),

            "identity_method": (
                winning_result[
                    "method"
                ]
            ),

            "gt_process": row.get(
                "gt_process"
            ),

            "gt_overlap": float(
                row.get(
                    "gt_overlap",
                    0.0,
                )
            ),

        })

    return assignments


# ============================================================
# CHECK REPEATED-PROCESS / A-B-A BEHAVIOR
#
# This is diagnostic only.
# We DO NOT alter labels based on this.
# ============================================================

def analyze_repeated_process_patterns(
    assignments,
):

    grouped = defaultdict(list)

    for row in assignments:

        grouped[
            row["session_id"]
        ].append(row)

    repeated_process_sessions = 0
    aba_count = 0

    for session_id, rows in grouped.items():

        rows.sort(
            key=lambda r: (
                parse_ts_ms(
                    r["start"]
                ),
                r["segment_index"],
            )
        )

        labels = [
            row["label"]
            for row in rows
        ]

        counts = Counter(
            labels
        )

        if any(
            count >= 2
            for count in counts.values()
        ):

            repeated_process_sessions += 1

        for i in range(
            1,
            len(labels) - 1,
        ):

            if (
                labels[i - 1]
                == labels[i + 1]
                and labels[i - 1]
                != labels[i]
            ):

                aba_count += 1

    return {
        "sessions_with_repeated_process": (
            repeated_process_sessions
        ),
        "aba_patterns": aba_count,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Phase 6.10 identity model selection "
            "using Dataset A GT."
        )
    )

    parser.add_argument(
        "--dataset-a",
        required=True,
        help="Dataset A root directory.",
    )

    parser.add_argument(
        "--segments",
        default=(
            "Outputs/Day 3/"
            "phase6_8_validated_segments.jsonl"
        ),
        help=(
            "Phase 6.8 validated segment file."
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "Outputs/Day 3/"
            "phase6_10"
        ),
        help=(
            "Phase 6.10 output directory."
        ),
    )

    parser.add_argument(
        "--local-weight",
        type=float,
        default=DEFAULT_LOCAL_WEIGHT,
        help=(
            "Weight of local-neighbor evidence."
        ),
    )

    parser.add_argument(
        "--process-weight",
        type=float,
        default=DEFAULT_PROCESS_WEIGHT,
        help=(
            "Weight of process-level evidence."
        ),
    )

    parser.add_argument(
        "--min-gt-overlap",
        type=float,
        default=DEFAULT_MIN_GT_OVERLAP,
        help=(
            "Minimum GT overlap required for identity "
            "evaluation."
        ),
    )

    args = parser.parse_args()

    dataset = Path(
        args.dataset_a
    )

    segment_path = Path(
        args.segments
    )

    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # WEIGHT VALIDATION
    # --------------------------------------------------------

    if (
        args.local_weight < 0
        or args.process_weight < 0
        or (
            args.local_weight
            + args.process_weight
            > 1.0
        )
    ):

        raise ValueError(
            "local-weight + process-weight must be "
            "between 0 and 1."
        )

    print()
    print("=" * 76)
    print(
        "PHASE 6.10 - IDENTITY MODEL SELECTION"
    )
    print("=" * 76)

    print()
    print("EXPERIMENT DESIGN")
    print("-" * 76)

    print(
        "1. Exact Phase 6.5 feature pipeline"
    )

    print(
        "2. Phase 6.8 predicted segments"
    )

    print(
        "3. Same baseline Random Forest"
    )

    print(
        "4. RF-only"
    )

    print(
        "5. RF + local-neighbor evidence"
    )

    print(
        "6. RF + process-level consistency"
    )

    print(
        "7. Combined RF + local + process"
    )

    print(
        "8. Dataset A GT validation"
    )

    print(
        "9. GT-based winner selection"
    )

    print(
        "10. No hard neighbor constraint"
    )

    print(
        "11. A-B-A / repeated processes preserved"
    )

    print()
    print(
        f"Local evidence weight   : "
        f"{args.local_weight}"
    )

    print(
        f"Process evidence weight : "
        f"{args.process_weight}"
    )

    print(
        f"Minimum GT overlap      : "
        f"{args.min_gt_overlap}"
    )

    # --------------------------------------------------------
    # CHECK INPUTS
    # --------------------------------------------------------

    if not dataset.exists():

        raise FileNotFoundError(
            f"Dataset A does not exist:\n{dataset}"
        )

    if not segment_path.exists():

        raise FileNotFoundError(
            "Could not find Phase 6.8 segment file:\n"
            f"{segment_path}"
        )

    # --------------------------------------------------------
    # LOAD PHASE 6.5
    # --------------------------------------------------------

    print()
    print("LOADING EXACT PHASE 6.5 PIPELINE")
    print("-" * 76)

    phase65 = load_phase65_module()

    # --------------------------------------------------------
    # DATASET A TRAINING DATA
    # --------------------------------------------------------

    print()
    print("BUILDING PHASE 6.5 TRAINING FEATURES")
    print("-" * 76)

    manifests = phase65.load_manifest(
        dataset
    )

    events_by_session = (
        phase65.load_events(
            dataset
        )
    )

    vocab = phase65.collect_vocab(
        events_by_session
    )

    training_rows, diagnostics = (
        phase65.build_feature_dataset(
            manifests,
            events_by_session,
            vocab,
        )
    )

    print(
        f"Training rows        : "
        f"{len(training_rows)}"
    )

    print(
        f"Training diagnostics : "
        f"{diagnostics}"
    )

    if len(training_rows) < 1600:

        raise RuntimeError(
            "Unexpectedly small Phase 6.5 training "
            f"population: {len(training_rows)}"
        )

    # --------------------------------------------------------
    # EXACT 6.9 FEATURE SELECTION
    # --------------------------------------------------------

    feature_names = (
        get_training_feature_names(
            training_rows
        )
    )

    combined_features = (
        get_combined_features(
            feature_names
        )
    )

    print()
    print(
        f"Numeric features     : "
        f"{len(feature_names)}"
    )

    print(
        f"Identity features    : "
        f"{len(combined_features)}"
    )

    if len(combined_features) != 417:

        print()
        print(
            "WARNING:"
        )

        print(
            "Expected approximately 417 identity "
            "features from Phase 6.5/6.9."
        )

        print(
            f"Observed: {len(combined_features)}"
        )

    # --------------------------------------------------------
    # TRAIN MATRIX
    # --------------------------------------------------------

    X_train, y_train, train_groups = (
        build_training_matrix(
            training_rows,
            combined_features,
        )
    )

    labels = np.asarray(
        sorted(
            set(y_train)
        )
    )

    print(
        f"Process classes      : "
        f"{len(labels)}"
    )

    print(
        f"Classes              : "
        f"{list(labels)}"
    )

    # --------------------------------------------------------
    # SAME RF AS 6.9
    # --------------------------------------------------------

    model = RandomForestClassifier(

        n_estimators=500,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        class_weight="balanced_subsample",

        min_samples_leaf=2,
    )

    print()
    print("TRAINING BASELINE RF")
    print("-" * 76)

    model.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # PROCESS PROTOTYPES
    # --------------------------------------------------------

    print()
    print("BUILDING PROCESS-LEVEL SIGNATURES")
    print("-" * 76)

    scaler_mean, scaler_std = (
        fit_feature_scaler(
            X_train
        )
    )

    prototypes = (
        build_process_prototypes(
            X_train,
            y_train,
        )
    )

    print(
        f"Process prototypes  : "
        f"{len(prototypes)}"
    )

    # --------------------------------------------------------
    # LOAD PHASE 6.8 SEGMENTS
    # --------------------------------------------------------

    print()
    print("LOADING PHASE 6.8 SEGMENTS")
    print("-" * 76)

    segments = load_jsonl(
        segment_path
    )

    print(
        f"Segments loaded      : "
        f"{len(segments)}"
    )

    if not segments:

        raise RuntimeError(
            "Phase 6.8 segment file is empty."
        )

    # --------------------------------------------------------
    # BUILD SEGMENT FEATURES
    # --------------------------------------------------------

    segment_rows, segment_diagnostics = (
        build_segment_features(
            phase65,
            segments,
            events_by_session,
            vocab,
        )
    )

    print()
    print("SEGMENT FEATURE DIAGNOSTICS")
    print("-" * 76)

    for key, value in sorted(
        segment_diagnostics.items()
    ):

        print(
            f"{key:35s}: {value}"
        )

    if not segment_rows:

        raise RuntimeError(
            "No usable Phase 6.8 segment feature rows."
        )

    # --------------------------------------------------------
    # ATTACH GT
    # --------------------------------------------------------

    print()
    print("LOADING DATASET A GT")
    print("-" * 76)

    gt_windows = load_gt_windows(
        dataset
    )

    print(
        f"Complete GT windows  : "
        f"{len(gt_windows)}"
    )

    segment_rows = attach_gt(
        segment_rows,
        gt_windows,
    )

    gt_evaluable = sum(
        1
        for row in segment_rows
        if (
            row["gt_process"] is not None
            and row["gt_overlap"]
            >= args.min_gt_overlap
        )
    )

    print(
        f"GT-evaluable segments: "
        f"{gt_evaluable}"
    )

    print(
        f"GT coverage           : "
        f"{gt_evaluable / len(segment_rows):.4f}"
    )

    # --------------------------------------------------------
    # SEGMENT MATRIX
    # --------------------------------------------------------

    X_segments = build_feature_matrix(
        segment_rows,
        combined_features,
    )

    # --------------------------------------------------------
    # BASE RF PROBABILITIES
    # --------------------------------------------------------

    print()
    print("CALCULATING RF EVIDENCE")
    print("-" * 76)

    rf_probabilities = (
        model.predict_proba(
            X_segments
        )
    )

    rf_probabilities = (
        normalize_rows(
            rf_probabilities
        )
    )

    # --------------------------------------------------------
    # PROCESS-LEVEL EVIDENCE
    # --------------------------------------------------------

    print(
        "Calculating process-level evidence..."
    )

    process_scores = (
        compute_process_scores(
            X_segments,
            prototypes,
            scaler_mean,
            scaler_std,
            labels,
        )
    )

    # --------------------------------------------------------
    # LOCAL EVIDENCE
    # --------------------------------------------------------

    print(
        "Calculating local-neighbor evidence..."
    )

    local_scores = (
        compute_local_scores(
            segment_rows,
            rf_probabilities,
            labels,
        )
    )

    # --------------------------------------------------------
    # FOUR EXPERIMENTS
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print("EVALUATING FOUR IDENTITY APPROACHES")
    print("=" * 76)

    # --------------------------------------------------------
    # 1. RF ONLY
    # --------------------------------------------------------

    rf_only_probabilities = (
        rf_probabilities.copy()
    )

    # --------------------------------------------------------
    # 2. RF + LOCAL
    #
    # RF receives remaining 80%.
    # Local receives 20%.
    # --------------------------------------------------------

    rf_local_probabilities = (
        combine_probabilities(
            rf_probabilities,
            local_scores,
            np.zeros_like(
                process_scores
            ),
            local_weight=args.local_weight,
            process_weight=0.0,
        )
    )

    # --------------------------------------------------------
    # 3. RF + PROCESS
    #
    # RF receives remaining 80%.
    # Process evidence receives 20%.
    # --------------------------------------------------------

    rf_process_probabilities = (
        combine_probabilities(
            rf_probabilities,
            np.zeros_like(
                local_scores
            ),
            process_scores,
            local_weight=0.0,
            process_weight=args.process_weight,
        )
    )

    # --------------------------------------------------------
    # 4. COMBINED
    #
    # Example with default weights:
    #
    # RF      = 60%
    # Local   = 20%
    # Process = 20%
    # --------------------------------------------------------

    combined_probabilities = (
        combine_probabilities(
            rf_probabilities,
            local_scores,
            process_scores,
            local_weight=args.local_weight,
            process_weight=args.process_weight,
        )
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    result_rf = (
        evaluate_method_with_classes(
            "RF-only",
            rf_only_probabilities,
            labels,
            segment_rows,
            args.min_gt_overlap,
        )
    )

    result_local = (
        evaluate_method_with_classes(
            "RF + local-neighbor",
            rf_local_probabilities,
            labels,
            segment_rows,
            args.min_gt_overlap,
        )
    )

    result_process = (
        evaluate_method_with_classes(
            "RF + process-consistency",
            rf_process_probabilities,
            labels,
            segment_rows,
            args.min_gt_overlap,
        )
    )

    result_combined = (
        evaluate_method_with_classes(
            "Combined",
            combined_probabilities,
            labels,
            segment_rows,
            args.min_gt_overlap,
        )
    )

    results = [
        result_rf,
        result_local,
        result_process,
        result_combined,
    ]

    # --------------------------------------------------------
    # PRINT COMPARISON
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print("DATASET A GT COMPARISON")
    print("=" * 76)

    print()

    print(
        f"{'Method':30s}"
        f"{'Accuracy':>12s}"
        f"{'Macro-F1':>12s}"
        f"{'Coverage':>12s}"
        f"{'Mean Conf':>12s}"
        f"{'Mean Ov':>12s}"
    )

    print("-" * 90)

    for result in results:

        print(
            f"{result['method']:30s}"
            f"{result['accuracy']:12.4f}"
            f"{result['macro_f1']:12.4f}"
            f"{result['coverage']:12.4f}"
            f"{result['mean_confidence']:12.4f}"
            f"{result['mean_gt_overlap']:12.4f}"
        )

    print()

    for result in results:

        print(
            f"{result['method']}"
        )

        print(
            f"  Accuracy       : "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"  Macro-F1       : "
            f"{result['macro_f1']:.4f}"
        )

        print(
            f"  Coverage       : "
            f"{result['coverage']:.4f}"
        )

        print(
            f"  Mean confidence: "
            f"{result['mean_confidence']:.4f}"
        )

        print(
            f"  Median conf.   : "
            f"{result['median_confidence']:.4f}"
        )

        print(
            f"  Mean margin    : "
            f"{result['mean_margin']:.4f}"
        )

        print(
            f"  >=50% overlap  : "
            f"{result['overlap_50_count']}"
            f"/{result['total_segments']}"
            f" ({result['overlap_50_rate']:.4f})"
        )

        print(
            f"  >=80% overlap  : "
            f"{result['overlap_80_count']}"
            f"/{result['total_segments']}"
            f" ({result['overlap_80_rate']:.4f})"
        )

        print()

    # --------------------------------------------------------
    # WINNER
    # --------------------------------------------------------

    winner, ranked = select_winner(
        results
    )

    print("=" * 76)
    print("GT-VALIDATED WINNER")
    print("=" * 76)

    print()
    print(
        f"WINNER : {winner['method']}"
    )

    print(
        f"Accuracy : "
        f"{winner['accuracy']:.4f}"
    )

    print(
        f"Macro-F1 : "
        f"{winner['macro_f1']:.4f}"
    )

    print(
        f"Coverage : "
        f"{winner['coverage']:.4f}"
    )

    # --------------------------------------------------------
    # WINNING ASSIGNMENTS
    # --------------------------------------------------------

    assignments = build_assignments(
        segment_rows,
        winner,
    )

    # --------------------------------------------------------
    # REPEATED PROCESS DIAGNOSTICS
    # --------------------------------------------------------

    pattern_diagnostics = (
        analyze_repeated_process_patterns(
            assignments
        )
    )

    print()
    print("PROCESS INTERLEAVING DIAGNOSTICS")
    print("-" * 76)

    print(
        "Sessions with repeated process:"
        f" {pattern_diagnostics['sessions_with_repeated_process']}"
    )

    print(
        "A-B-A patterns:"
        f" {pattern_diagnostics['aba_patterns']}"
    )

    print()
    print(
        "NOTE: A-B-A and repeated-process patterns "
        "are allowed; no hard neighbor constraint was applied."
    )

    # --------------------------------------------------------
    # FINAL REQUIRED segments.jsonl
    # --------------------------------------------------------

    final_segments = [

        {
            "session_id": row[
                "session_id"
            ],

            "start": row[
                "start"
            ],

            "end": row[
                "end"
            ],

            "label": row[
                "label"
            ],
        }

        for row in assignments
    ]

    final_segments_path = (
        output / "segments.jsonl"
    )

    save_jsonl(
        final_segments_path,
        final_segments,
    )

    # --------------------------------------------------------
    # FULL WINNING ASSIGNMENTS
    # --------------------------------------------------------

    save_jsonl(
        output
        / "phase6_10_winning_identity_assignments.jsonl",
        assignments,
    )

    # --------------------------------------------------------
    # METHOD RESULTS CSV
    # --------------------------------------------------------

    comparison_rows = []

    for rank, result in enumerate(
        ranked,
        start=1,
    ):

        comparison_rows.append({

            "rank": rank,

            "method": result[
                "method"
            ],

            "accuracy": result[
                "accuracy"
            ],

            "macro_f1": result[
                "macro_f1"
            ],

            "coverage": result[
                "coverage"
            ],

            "evaluable_count": result[
                "evaluable_count"
            ],

            "total_segments": result[
                "total_segments"
            ],

            "mean_confidence": result[
                "mean_confidence"
            ],

            "median_confidence": result[
                "median_confidence"
            ],

            "mean_margin": result[
                "mean_margin"
            ],

            "mean_gt_overlap": result[
                "mean_gt_overlap"
            ],

            "overlap_50_count": result[
                "overlap_50_count"
            ],

            "overlap_50_rate": result[
                "overlap_50_rate"
            ],

            "overlap_80_count": result[
                "overlap_80_count"
            ],

            "overlap_80_rate": result[
                "overlap_80_rate"
            ],

        })

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df.to_csv(
        output
        / "phase6_10_method_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SEGMENT-LEVEL COMPARISON
    # --------------------------------------------------------

    rf_pred, rf_conf, rf_margin, rf_second = (
        prediction_from_probabilities(
            rf_only_probabilities,
            labels,
        )
    )

    local_pred, local_conf, local_margin, local_second = (
        prediction_from_probabilities(
            rf_local_probabilities,
            labels,
        )
    )

    process_pred, process_conf, process_margin, process_second = (
        prediction_from_probabilities(
            rf_process_probabilities,
            labels,
        )
    )

    combined_pred, combined_conf, combined_margin, combined_second = (
        prediction_from_probabilities(
            combined_probabilities,
            labels,
        )
    )

    segment_comparison = []

    for i, row in enumerate(
        segment_rows
    ):

        segment_comparison.append({

            "segment_index": int(
                row["segment_index"]
            ),

            "session_id": row[
                "session_id"
            ],

            "start": row[
                "start"
            ],

            "end": row[
                "end"
            ],

            "gt_process": row.get(
                "gt_process"
            ),

            "gt_overlap": float(
                row.get(
                    "gt_overlap",
                    0.0,
                )
            ),

            "rf_prediction": rf_pred[i],

            "rf_confidence": float(
                rf_conf[i]
            ),

            "rf_margin": float(
                rf_margin[i]
            ),

            "local_prediction": local_pred[i],

            "local_confidence": float(
                local_conf[i]
            ),

            "local_margin": float(
                local_margin[i]
            ),

            "process_prediction": process_pred[i],

            "process_confidence": float(
                process_conf[i]
            ),

            "process_margin": float(
                process_margin[i]
            ),

            "combined_prediction": combined_pred[i],

            "combined_confidence": float(
                combined_conf[i]
            ),

            "combined_margin": float(
                combined_margin[i]
            ),

            "winning_prediction": assignments[i][
                "label"
            ],

            "winning_method": winner[
                "method"
            ],

        })

    pd.DataFrame(
        segment_comparison
    ).to_csv(
        output
        / "phase6_10_segment_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {

        "phase": "6.10",

        "purpose": (
            "GT-validated identity model selection"
        ),

        "feature_pipeline": (
            "exact_phase_6_5"
        ),

        "identity_model": (
            "RandomForestClassifier"
        ),

        "random_forest": {

            "n_estimators": 500,

            "random_state": RANDOM_STATE,

            "class_weight": (
                "balanced_subsample"
            ),

            "min_samples_leaf": 2,

        },

        "transition_hybrid": False,

        "neighbor_rule": (
            "soft evidence only"
        ),

        "hard_neighbor_constraint": False,

        "repeated_process_allowed": True,

        "aba_allowed": True,

        "training_rows": int(
            len(training_rows)
        ),

        "numeric_feature_count": int(
            len(feature_names)
        ),

        "identity_feature_count": int(
            len(combined_features)
        ),

        "segments_loaded": int(
            len(segments)
        ),

        "usable_segments": int(
            len(segment_rows)
        ),

        "gt_windows": int(
            len(gt_windows)
        ),

        "gt_evaluable_segments": int(
            gt_evaluable
        ),

        "minimum_gt_overlap": float(
            args.min_gt_overlap
        ),

        "local_weight": float(
            args.local_weight
        ),

        "process_weight": float(
            args.process_weight
        ),

        "winner": winner[
            "method"
        ],

        "winner_accuracy": winner[
            "accuracy"
        ],

        "winner_macro_f1": winner[
            "macro_f1"
        ],

        "winner_coverage": winner[
            "coverage"
        ],

        "winner_mean_confidence": winner[
            "mean_confidence"
        ],

        "winner_median_confidence": winner[
            "median_confidence"
        ],

        "winner_mean_gt_overlap": winner[
            "mean_gt_overlap"
        ],

        "method_ranking": [

            {
                "rank": i + 1,

                "method": result[
                    "method"
                ],

                "accuracy": result[
                    "accuracy"
                ],

                "macro_f1": result[
                    "macro_f1"
                ],

                "coverage": result[
                    "coverage"
                ],

            }

            for i, result in enumerate(
                ranked
            )
        ],

        "interleaving_diagnostics": (
            pattern_diagnostics
        ),

        "segment_diagnostics": (
            segment_diagnostics
        ),

    }

    with open(
        output
        / "phase6_10_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print("PHASE 6.10 COMPLETE")
    print("=" * 76)

    print()
    print(
        f"WINNING APPROACH : "
        f"{winner['method']}"
    )

    print(
        f"GT Accuracy      : "
        f"{winner['accuracy']:.4f}"
    )

    print(
        f"GT Macro-F1      : "
        f"{winner['macro_f1']:.4f}"
    )

    print(
        f"GT Coverage      : "
        f"{winner['coverage']:.4f}"
    )

    print()
    print("OUTPUTS")
    print("-" * 76)

    print(
        output
        / "segments.jsonl"
    )

    print(
        output
        / "phase6_10_winning_identity_assignments.jsonl"
    )

    print(
        output
        / "phase6_10_method_comparison.csv"
    )

    print(
        output
        / "phase6_10_segment_comparison.csv"
    )

    print(
        output
        / "phase6_10_summary.json"
    )

    print()
    print(
        "FINAL DELIVERABLE:"
    )

    print(
        output
        / "segments.jsonl"
    )

    print()
    print(
        "Phase 6.8 boundaries were preserved."
    )

    print(
        "No hard neighbor-label constraint was applied."
    )

    print(
        "Repeated processes and A-B-A behavior remain possible."
    )

    print(
        "The winning method was selected using Dataset A GT."
    )


if __name__ == "__main__":
    main()