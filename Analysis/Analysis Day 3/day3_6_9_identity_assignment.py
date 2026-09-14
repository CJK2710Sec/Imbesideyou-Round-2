import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


RANDOM_STATE = 42
DEFAULT_CONFIDENCE_THRESHOLD = 0.80


# ============================================================
# LOAD PROVEN PHASE 6.5 PIPELINE
# ============================================================

def load_phase65_module():
    script_path = Path(__file__).resolve().parent / "day3_sequence_context.py"

    if not script_path.exists():
        # Also support the common filename used in the project.
        alternatives = [
            Path(__file__).resolve().parent / "day3_phase6_5_sequence_context_fixed.py",
            Path(__file__).resolve().parent / "day3_phase6_5_sequence_context.py",
        ]
        for candidate in alternatives:
            if candidate.exists():
                script_path = candidate
                break

    if not script_path.exists():
        raise FileNotFoundError(
            "Could not find the Phase 6.5 sequence/context script. "
            "Keep the proven Phase 6.5 script in the same folder."
        )

    spec = importlib.util.spec_from_file_location("phase65", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            clean = {}
            for key, value in row.items():
                if hasattr(value, "isoformat"):
                    clean[key] = value.isoformat()
                elif isinstance(value, np.generic):
                    clean[key] = value.item()
                else:
                    clean[key] = value
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


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
        return ts.timestamp() * 1000.0
    except Exception:
        return None


# ============================================================
# SEGMENT FEATURES
#
# IMPORTANT:
# Uses the EXACT Phase 6.5 make_features() implementation.
# ============================================================

def build_segment_features(phase65, segments, events_by_session, vocab):
    rows = []
    diagnostics = defaultdict(int)

    for index, segment in enumerate(segments):
        session_id = segment.get("session_id")
        start_ms = parse_ts_ms(segment.get("start"))
        end_ms = parse_ts_ms(segment.get("end"))

        if session_id is None or start_ms is None or end_ms is None:
            diagnostics["invalid_segment"] += 1
            continue

        if end_ms <= start_ms:
            diagnostics["invalid_interval"] += 1
            continue

        events = events_by_session.get(session_id, [])
        if not events:
            diagnostics["missing_session_events"] += 1
            continue

        timestamp_array = np.asarray(
            [event["_ts_ms"] for event in events],
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
            diagnostics["no_events_inside_segment"] += 1
            continue

        row = dict(features)
        row["segment_index"] = index
        row["session_id"] = session_id
        row["start"] = segment.get("start")
        row["end"] = segment.get("end")
        rows.append(row)

    diagnostics["segments_total"] = len(segments)
    diagnostics["usable_segments"] = len(rows)

    return rows, dict(diagnostics)


# ============================================================
# TRAINING FEATURES
#
# This reproduces the feature selection used in Phase 6.7.
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

    return sorted({
        key
        for row in rows
        for key, value in row.items()
        if (
            key not in excluded
            and isinstance(value, (int, float, np.integer, np.floating))
        )
    })


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

    return sorted(set(sequence_features + context_features))


# ============================================================
# CONFIDENCE
# ============================================================

def confidence_info(probabilities, classes):
    order = np.argsort(probabilities)[::-1]

    best = int(order[0])
    second = int(order[1]) if len(order) > 1 else best

    best_probability = float(probabilities[best])
    second_probability = float(probabilities[second])

    return {
        "predicted_process": str(classes[best]),
        "second_best_process": str(classes[second]),
        "confidence": best_probability,
        "top2_probability": second_probability,
        "margin": best_probability - second_probability,
    }


# ============================================================
# OPTIONAL GT OVERLAP VALIDATION
# ============================================================

def load_gt_windows(dataset):
    windows = []

    for manifest_path in sorted(Path(dataset).rglob("gt_manifest.json")):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue

        session_id = manifest.get("session", {}).get("session_id")
        if not session_id:
            session_id = manifest_path.parent.name

        for process in manifest.get("processes", []):
            label = process.get("code")
            family = process.get("family_name")

            for execution in process.get("executions", []):
                start = parse_ts_ms(execution.get("start_ts"))
                end = parse_ts_ms(execution.get("end_ts"))

                if start is None or end is None or end <= start:
                    continue

                windows.append({
                    "session_id": session_id,
                    "start_ms": start,
                    "end_ms": end,
                    "label": label,
                    "family_name": family,
                    "variant": execution.get("variant"),
                    "case_id": execution.get("case_id"),
                })

    return windows


def best_gt_for_segment(segment, gt_windows):
    start = parse_ts_ms(segment.get("start"))
    end = parse_ts_ms(segment.get("end"))

    if start is None or end is None or end <= start:
        return None, 0.0

    best = None
    best_ratio = 0.0
    segment_session = segment.get("session_id")

    for gt in gt_windows:
        if gt["session_id"] != segment_session:
            continue

        overlap_start = max(start, gt["start_ms"])
        overlap_end = min(end, gt["end_ms"])

        if overlap_end <= overlap_start:
            continue

        overlap = overlap_end - overlap_start
        gt_duration = gt["end_ms"] - gt["start_ms"]

        if gt_duration <= 0:
            continue

        ratio = overlap / gt_duration

        if ratio > best_ratio:
            best_ratio = ratio
            best = gt

    return best, best_ratio


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-a",
        required=True,
        help="Path to Dataset A root",
    )

    parser.add_argument(
        "--segments",
        default="Outputs/Day 3/phase6_8_validated_segments.jsonl",
        help="Phase 6.8 validated segments JSONL",
    )

    parser.add_argument(
        "--output",
        default="Outputs/Day 3/phase6_9",
        help="Phase 6.9 output directory",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
    )

    args = parser.parse_args()

    dataset = Path(args.dataset_a)
    segment_file = Path(args.segments)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("PHASE 6.9 - PROCESS IDENTITY ASSIGNMENT")
    print("=" * 70)

    print()
    print("IMPORTANT DECISIONS")
    print("-" * 70)
    print("Identity model       : Sequence + Context Random Forest")
    print("Transition hybrid    : DISABLED")
    print("Confidence is        : model confidence, not calibrated probability")
    print(f"High-confidence gate : {args.confidence_threshold}")

    if not segment_file.exists():
        raise FileNotFoundError(
            f"Could not find Phase 6.8 segment file:\n{segment_file}"
        )

    phase65 = load_phase65_module()

    # --------------------------------------------------------
    # LOAD TRAINING DATA
    # --------------------------------------------------------

    print()
    print("LOADING PHASE 6.5 TRAINING PIPELINE")
    print("-" * 70)

    manifests = phase65.load_manifest(dataset)
    events_by_session = phase65.load_events(dataset)
    vocab = phase65.collect_vocab(events_by_session)

    training_rows, diagnostics = phase65.build_feature_dataset(
        manifests,
        events_by_session,
        vocab,
    )

    print(f"Manifest executions : {sum(len(x[1]) for x in manifests)}")
    print(f"Usable training rows: {len(training_rows)}")

    if len(training_rows) < 1600:
        raise RuntimeError(
            f"Too few training rows ({len(training_rows)}). "
            "Expected approximately 1734."
        )

    feature_names = get_training_feature_names(training_rows)
    combined_features = get_combined_features(feature_names)

    print(f"Numeric features    : {len(feature_names)}")
    print(f"Identity features   : {len(combined_features)}")

    # --------------------------------------------------------
    # TRAIN FINAL BASE RF
    #
    # No transition hybrid: Phase 6.7 showed the hybrid
    # reduced validation performance and is therefore dropped.
    # --------------------------------------------------------

    X_train = np.asarray([
        [
            float(row.get(feature, 0.0))
            for feature in combined_features
        ]
        for row in training_rows
    ], dtype=float)

    X_train = np.nan_to_num(
        X_train,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    y_train = np.asarray([
        row["label"]
        for row in training_rows
    ])

    labels = sorted(set(y_train))

    print(f"Process classes     : {len(labels)}")
    print(f"Classes              : {labels}")

    model = RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )

    print()
    print("TRAINING FINAL IDENTITY MODEL")
    print("-" * 70)

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # LOAD PHASE 6.8 SEGMENTS
    # --------------------------------------------------------

    segments = load_jsonl(segment_file)

    print()
    print("LOADING PHASE 6.8 SEGMENTS")
    print("-" * 70)
    print(f"Segments loaded      : {len(segments)}")

    segment_rows, segment_diagnostics = build_segment_features(
        phase65,
        segments,
        events_by_session,
        vocab,
    )

    print()
    print("SEGMENT FEATURE DIAGNOSTICS")
    print("-" * 70)

    for key, value in segment_diagnostics.items():
        print(f"{key:30s}: {value}")

    if not segment_rows:
        raise RuntimeError("No usable segment feature rows were produced.")

    # --------------------------------------------------------
    # APPLY MODEL
    # --------------------------------------------------------

    X_segments = np.asarray([
        [
            float(row.get(feature, 0.0))
            for feature in combined_features
        ]
        for row in segment_rows
    ], dtype=float)

    X_segments = np.nan_to_num(
        X_segments,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    probabilities = model.predict_proba(X_segments)

    results = []

    for i, row in enumerate(segment_rows):
        info = confidence_info(
            probabilities[i],
            model.classes_,
        )

        result = {
            "segment_index": int(row["segment_index"]),
            "session_id": row["session_id"],
            "start": row["start"],
            "end": row["end"],
            **info,
            "high_confidence": (
                info["confidence"] >= args.confidence_threshold
            ),
        }

        results.append(result)

    results_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # CONFIDENCE SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IDENTITY CONFIDENCE")
    print("=" * 70)

    print(
        f"Usable segments       : {len(results_df)}"
    )

    high = results_df[
        results_df["confidence"] >= args.confidence_threshold
    ]

    print(
        f"High-confidence      : {len(high)} "
        f"({len(high) / len(results_df):.4f})"
    )

    print(
        f"Below threshold      : "
        f"{len(results_df) - len(high)}"
    )

    print()
    print(
        "Mean confidence      : "
        f"{results_df['confidence'].mean():.4f}"
    )

    print(
        "Median confidence    : "
        f"{results_df['confidence'].median():.4f}"
    )

    print(
        "Mean margin          : "
        f"{results_df['margin'].mean():.4f}"
    )

    # --------------------------------------------------------
    # CONFIDENCE BINS
    # --------------------------------------------------------

    bins = [
        0.0, 0.20, 0.40, 0.60,
        0.70, 0.80, 0.90, 1.01,
    ]

    bin_labels = [
        "0.00-0.19",
        "0.20-0.39",
        "0.40-0.59",
        "0.60-0.69",
        "0.70-0.79",
        "0.80-0.89",
        "0.90-1.00",
    ]

    results_df["confidence_bin"] = pd.cut(
        results_df["confidence"],
        bins=bins,
        labels=bin_labels,
        right=False,
    )

    confidence_bins = (
        results_df
        .groupby("confidence_bin", observed=False)
        .size()
        .reset_index(name="count")
    )

    # --------------------------------------------------------
    # OPTIONAL GT VALIDATION
    # --------------------------------------------------------

    print()
    print("GT OVERLAP VALIDATION")
    print("-" * 70)

    gt_windows = load_gt_windows(dataset)

    gt_correct = 0
    gt_evaluable = 0
    overlap_50 = 0
    overlap_80 = 0

    gt_annotations = {}

    for result in results:
        gt, ratio = best_gt_for_segment(result, gt_windows)

        if gt is not None:
            gt_evaluable += 1

            if ratio >= 0.50:
                overlap_50 += 1

            if ratio >= 0.80:
                overlap_80 += 1

            result["gt_process"] = gt["label"]
            result["gt_family"] = gt["family_name"]
            result["gt_overlap"] = ratio

            if result["predicted_process"] == gt["label"]:
                gt_correct += 1
        else:
            result["gt_process"] = None
            result["gt_family"] = None
            result["gt_overlap"] = 0.0

    print(f"Complete GT windows  : {len(gt_windows)}")
    print(f"GT-evaluable segments: {gt_evaluable}")

    if gt_evaluable:
        print(
            f"Identity accuracy on best-overlap GT : "
            f"{gt_correct / gt_evaluable:.4f}"
        )

    print(
        f">=50% GT overlap    : "
        f"{overlap_50}/{len(results)} "
        f"({overlap_50 / len(results):.4f})"
    )

    print(
        f">=80% GT overlap    : "
        f"{overlap_80}/{len(results)} "
        f"({overlap_80 / len(results):.4f})"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    results_df.to_csv(
        output / "phase6_9_segment_identity.csv",
        index=False,
    )

    confidence_bins.to_csv(
        output / "phase6_9_confidence_bins.csv",
        index=False,
    )

    save_jsonl(
        output / "phase6_9_segment_identity.jsonl",
        results,
    )

    # --------------------------------------------------------
    # PROCESS COUNTS
    # --------------------------------------------------------

    process_counts = (
        results_df["predicted_process"]
        .value_counts()
        .rename_axis("process")
        .reset_index(name="segment_count")
    )

    process_counts.to_csv(
        output / "phase6_9_predicted_process_counts.csv",
        index=False,
    )

    summary = {
        "phase": "6.9",
        "identity_model": "sequence_context_random_forest",
        "transition_hybrid": False,
        "training_rows": int(len(training_rows)),
        "process_classes": int(len(labels)),
        "feature_count": int(len(combined_features)),
        "segments_loaded": int(len(segments)),
        "usable_segments": int(len(results)),
        "confidence_threshold": float(args.confidence_threshold),
        "high_confidence_count": int(len(high)),
        "high_confidence_coverage": float(
            len(high) / len(results)
        ),
        "mean_confidence": float(
            results_df["confidence"].mean()
        ),
        "median_confidence": float(
            results_df["confidence"].median()
        ),
        "mean_margin": float(
            results_df["margin"].mean()
        ),
        "gt_windows": int(len(gt_windows)),
        "gt_evaluable_segments": int(gt_evaluable),
        "gt_identity_accuracy": (
            float(gt_correct / gt_evaluable)
            if gt_evaluable
            else None
        ),
        "overlap_50": int(overlap_50),
        "overlap_80": int(overlap_80),
    }

    with open(
        output / "phase6_9_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print("PHASE 6.9 COMPLETE")
    print("=" * 70)

    print()
    print("Outputs saved to:")
    print(output)

    print()
    print("Key file:")
    print("phase6_9_segment_identity.csv")


if __name__ == "__main__":
    main()
