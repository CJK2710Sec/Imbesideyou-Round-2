from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import argparse
import csv
import json
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DEFAULT_THRESHOLD = 0.70
MATCH_TOLERANCE_SECONDS = 1.0


# ============================================================
# TIMESTAMP HELPERS
# ============================================================

def parse_ts(value):
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except Exception:
        return None


def ms_to_datetime(value):
    return datetime.fromtimestamp(
        float(value) / 1000.0,
        tz=timezone.utc
    )


def datetime_to_ms(value):
    return value.timestamp() * 1000.0


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for row in rows:

            clean = {}

            for key, value in row.items():

                if isinstance(
                    value,
                    datetime
                ):
                    clean[key] = value.isoformat()

                else:
                    clean[key] = value

            f.write(
                json.dumps(
                    clean,
                    ensure_ascii=False
                ) + "\n"
            )


# ============================================================
# LOAD RAW EVENTS
#
# IMPORTANT:
# All chunks belonging to a session are combined.
# Chunk boundaries are NOT treated as process boundaries.
# ============================================================

def load_raw_events(dataset):

    events_by_session = defaultdict(list)

    dataset = Path(dataset)

    for session_path in sorted(
        dataset.glob("ses_*")
    ):

        session_id = session_path.name

        for event_path in sorted(
            session_path.glob(
                "chunk_*/events.jsonl"
            )
        ):

            with open(
                event_path,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    try:
                        event = json.loads(line)

                    except json.JSONDecodeError:
                        continue

                    timestamp = event.get(
                        "timestamp_iso"
                    )

                    dt = parse_ts(
                        timestamp
                    )

                    if dt is None:
                        continue

                    event["_dt"] = dt
                    event["_ms"] = (
                        datetime_to_ms(dt)
                    )

                    events_by_session[
                        session_id
                    ].append(event)

    for session_id in events_by_session:

        events_by_session[
            session_id
        ].sort(
            key=lambda x: x["_ms"]
        )

    return dict(
        events_by_session
    )


# ============================================================
# LOAD DAY 2 LEARNED BOUNDARIES
#
# ACTUAL SCHEMA CONFIRMED FROM USER:
#
# session_id
# candidate_timestamp_ms
# confirmation_score
#
# Threshold = 0.70
# ============================================================

def load_day2_boundaries(csv_path, threshold=0.70):
    print("\nDAY 2 LEARNED OUTPUT")
    print(f"File : {csv_path}")

    df = pd.read_csv(csv_path)

    print(f"Columns : {list(df.columns)}")
    print(f"Total CSV rows : {len(df)}")

    # IMPORTANT:
    # Day 2's official threshold results were based on
    # boundary_probability, NOT confirmation_score.
    if "boundary_probability" not in df.columns:
        raise ValueError(
            "boundary_probability column not found in Day 2 output."
        )

    df = df.dropna(
        subset=["session_id", "candidate_timestamp_ms", "boundary_probability"]
    ).copy()

    print(f"Valid candidate rows : {len(df)}")

    # Use the actual Day 2 learned boundary probability
    df = df[df["boundary_probability"] >= threshold].copy()

    print(f"Rows >= threshold : {len(df)}")

    # Sort by session and timestamp
    df = df.sort_values(
        ["session_id", "candidate_timestamp_ms"]
    ).reset_index(drop=True)

    # Deduplicate candidates that are extremely close together.
    # Keep the highest-probability candidate in each local cluster.
    selected = []

    for session_id, group in df.groupby("session_id", sort=False):
        group = group.sort_values("candidate_timestamp_ms")

        cluster = []

        for _, row in group.iterrows():
            if not cluster:
                cluster = [row]
                continue

            prev_ts = cluster[-1]["candidate_timestamp_ms"]
            current_ts = row["candidate_timestamp_ms"]

            # Candidates within 500 ms belong to the same boundary cluster
            if current_ts - prev_ts <= 500:
                cluster.append(row)
            else:
                best = max(
                    cluster,
                    key=lambda x: x["boundary_probability"]
                )
                selected.append(best)
                cluster = [row]

        if cluster:
            best = max(
                cluster,
                key=lambda x: x["boundary_probability"]
            )
            selected.append(best)

    result = pd.DataFrame(selected)

    print(f"After deduplication : {len(result)}")

    return result

# ============================================================
# LOAD GT PROCESS-SWITCH BOUNDARIES
#
# ACTUAL GT SCHEMA CONFIRMED FROM USER:
#
# {
#   "ts_utc": "...",
#   "event": "process_switched_out",
#   ...
# }
#
# ============================================================

def load_gt_boundaries(dataset):
    """
    Load authoritative GT process-switch boundaries directly from
    every gt.jsonl under Dataset A.

    GT boundary definition:
        event == "process_switched_out"
        timestamp == ts_utc

    The session_id is taken from the nearest ancestor whose name
    starts with "ses_".
    """
    boundaries = defaultdict(list)

    dataset = Path(dataset)

    gt_files = sorted(dataset.rglob("gt.jsonl"))

    print(f"GT files found : {len(gt_files)}")

    if not gt_files:
        raise FileNotFoundError(
            f"No gt.jsonl files found recursively under: {dataset.resolve()}"
        )

    total_switch_rows = 0
    files_with_switches = 0

    for gt_path in gt_files:
        session_id = None

        for parent in gt_path.parents:
            if parent.name.startswith("ses_"):
                session_id = parent.name
                break

        if session_id is None:
            print(f"WARNING: could not determine session for {gt_path}")
            continue

        file_switches = 0

        try:
            with open(
                gt_path,
                "r",
                encoding="utf-8"
            ) as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if row.get("event") != "process_switched_out":
                        continue

                    timestamp = parse_ts(
                        row.get("ts_utc")
                    )

                    if timestamp is None:
                        continue

                    boundaries[session_id].append(timestamp)
                    total_switch_rows += 1
                    file_switches += 1

        except OSError as exc:
            print(f"WARNING: could not read {gt_path}: {exc}")
            continue

        if file_switches:
            files_with_switches += 1

    for session_id in boundaries:
        boundaries[session_id] = sorted(
            set(boundaries[session_id])
        )

    unique_total = sum(
        len(values)
        for values in boundaries.values()
    )

    print(f"GT files with process switches : {files_with_switches}")
    print(f"Raw GT switch rows              : {total_switch_rows}")
    print(f"Unique GT process-switch boundaries : {unique_total}")
    print(f"GT sessions with boundaries     : {len(boundaries)}")

    if boundaries:
        first_session = sorted(boundaries.keys())[0]
        print(
            f"Example GT session              : {first_session}"
        )
        print(
            f"Example GT boundaries           : "
            f"{len(boundaries[first_session])}"
        )

    return dict(boundaries)


# ============================================================
# ONE-TO-ONE GT ↔ PREDICTED MATCHING
# ============================================================

def match_boundaries(
    predicted,
    gt,
    tolerance_seconds=1.0
):
    """
    One-to-one nearest-neighbour matching.

    IMPORTANT:
    predicted and gt must already belong to the same session.
    Timestamp units:
      predicted: candidate_timestamp_ms (epoch milliseconds)
      gt: datetime (UTC)
    """
    tolerance_ms = tolerance_seconds * 1000.0

    candidate_pairs = []

    for pi, prediction in enumerate(predicted):
        prediction_ms = float(
            prediction["candidate_timestamp_ms"]
        )

        for gi, gt_time in enumerate(gt):
            gt_ms = datetime_to_ms(gt_time)

            error = abs(prediction_ms - gt_ms)

            if error <= tolerance_ms:
                candidate_pairs.append(
                    (error, pi, gi)
                )

    candidate_pairs.sort(key=lambda x: x[0])

    matched_predictions = set()
    matched_gt = set()
    errors = []

    for error, pi, gi in candidate_pairs:
        if pi in matched_predictions:
            continue

        if gi in matched_gt:
            continue

        matched_predictions.add(pi)
        matched_gt.add(gi)

        errors.append(error / 1000.0)

    tp = len(matched_predictions)
    fp = len(predicted) - tp
    fn = len(gt) - len(matched_gt)

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
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "errors": errors
    }


# ============================================================
# CONSTRUCT TEMPORAL SEGMENTS
# ============================================================

def construct_segments(
    events_by_session,
    predicted_boundaries
):

    boundaries_by_session = defaultdict(
        list
    )

    for boundary in predicted_boundaries:

        boundaries_by_session[
            boundary["session_id"]
        ].append(
            boundary
        )

    segments = []

    for session_id, events in (
        events_by_session.items()
    ):

        if not events:
            continue

        session_start = events[0][
            "_dt"
        ]

        session_end = events[-1][
            "_dt"
        ]

        session_boundaries = sorted(
            boundaries_by_session.get(
                session_id,
                []
            ),
            key=lambda x:
                x["candidate_timestamp_ms"]
        )

        points = [
            {
                "timestamp":
                    session_start,

                "score":
                    None
            }
        ]

        for boundary in (
            session_boundaries
        ):

            timestamp_ms = boundary[
                "candidate_timestamp_ms"
            ]
            timestamp = datetime.fromtimestamp(
                timestamp_ms / 1000,
                tz=timezone.utc
            )
            if timestamp <= session_start:
                continue

            # Ignore boundaries outside
            # the observed session interval.

            if (
                timestamp <= session_start
                or timestamp >= session_end
            ):
                continue

            points.append({
                "timestamp":
                    timestamp,

                "score":
                    boundary["boundary_probability"]
            })

        points.append({
            "timestamp":
                session_end,

            "score":
                None
        })

        for i in range(
            len(points) - 1
        ):

            start = points[i][
                "timestamp"
            ]

            end = points[i + 1][
                "timestamp"
            ]

            if end <= start:
                continue

            segments.append({
                "session_id":
                    session_id,

                "start":
                    start,

                "end":
                    end,

                "boundary_score":
                    points[i + 1].get(
                        "score"
                    )
            })

    return segments


# ============================================================
# LOAD GT EXECUTION WINDOWS
# ============================================================

def load_gt_execution_windows(
    dataset
):

    windows = []

    dataset = Path(dataset)

    manifest_files = sorted(
        dataset.rglob(
            "gt_manifest.json"
        )
    )

    for manifest_path in (
        manifest_files
    ):

        try:

            with open(
                manifest_path,
                "r",
                encoding="utf-8"
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

            session_id = (
                manifest_path
                .parent
                .name
            )

        for process in manifest.get(
            "processes",
            []
        ):

            process_code = process.get(
                "code"
            )

            family_name = process.get(
                "family_name"
            )

            for execution in process.get(
                "executions",
                []
            ):

                start = parse_ts(
                    execution.get(
                        "start_ts"
                    )
                )

                end = parse_ts(
                    execution.get(
                        "end_ts"
                    )
                )

                # Incomplete executions do not
                # define a closed temporal window.

                if (
                    start is None
                    or end is None
                    or end <= start
                ):
                    continue

                windows.append({
                    "session_id":
                        session_id,

                    "start":
                        start,

                    "end":
                        end,

                    "label":
                        process_code,

                    "family_name":
                        family_name,

                    "variant":
                        execution.get(
                            "variant"
                        ),

                    "case_id":
                        execution.get(
                            "case_id"
                        ),

                    "continues_from_prev":
                        execution.get(
                            "continues_from_prev",
                            False
                        ),

                    "continues_to_next":
                        execution.get(
                            "continues_to_next",
                            False
                        )
                })

    return windows


# ============================================================
# SEGMENT / GT OVERLAP
# ============================================================

def overlap_ratio(
    segment,
    gt
):

    start = max(
        segment["start"],
        gt["start"]
    )

    end = min(
        segment["end"],
        gt["end"]
    )

    if end <= start:
        return 0.0

    overlap_seconds = (
        end - start
    ).total_seconds()

    gt_duration = (
        gt["end"]
        - gt["start"]
    ).total_seconds()

    if gt_duration <= 0:
        return 0.0

    return (
        overlap_seconds
        / gt_duration
    )


def validate_segments(
    segments,
    gt_windows
):

    gt_by_session = defaultdict(
        list
    )

    for gt in gt_windows:

        gt_by_session[
            gt["session_id"]
        ].append(gt)

    validated = []

    for segment in segments:

        best_gt = None
        best_overlap = 0.0

        for gt in gt_by_session.get(
            segment["session_id"],
            []
        ):

            overlap = overlap_ratio(
                segment,
                gt
            )

            if overlap > best_overlap:

                best_overlap = overlap
                best_gt = gt

        result = dict(
            segment
        )

        result["gt_overlap"] = (
            best_overlap
        )

        if best_gt is not None:

            result["gt_process"] = (
                best_gt["label"]
            )

            result["gt_family"] = (
                best_gt["family_name"]
            )

            result["gt_variant"] = (
                best_gt["variant"]
            )

            result["gt_case_id"] = (
                best_gt["case_id"]
            )

        else:

            result["gt_process"] = None
            result["gt_family"] = None
            result["gt_variant"] = None
            result["gt_case_id"] = None

        validated.append(
            result
        )

    return validated


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-a",
        default="Dataset A/dataset_a"
    )

    parser.add_argument(
        "--output",
        default="Outputs/Day 3"
    )

    parser.add_argument(
        "--boundary-threshold",
        type=float,
        default=0.70
    )

    args = parser.parse_args()

    dataset = Path(
        args.dataset_a
    )

    output = Path(
        args.output
    )

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # EXACT VERIFIED FILES
    # ========================================================

    prediction_file = Path(
        "Outputs/Day 2/"
        "day2_learned_boundary_confirmation_v2/"
        "learned_boundary_scores_v2.csv"
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 70)
    print(
        "PHASE 6.8 - COHERENT SEGMENT CONSTRUCTION"
    )
    print("=" * 70)

    print()
    print(
        "Hybrid transition model : DISABLED"
    )

    print(
        f"Boundary threshold      : "
        f"{args.boundary_threshold}"
    )

    # ========================================================
    # RAW EVENTS
    # ========================================================

    print()
    print(
        "LOADING RAW EVENTS"
    )
    print("-" * 70)

    events_by_session = (
        load_raw_events(
            dataset
        )
    )

    print(
        f"Sessions with raw events : "
        f"{len(events_by_session)}"
    )

    total_events = sum(
        len(events)
        for events in
        events_by_session.values()
    )

    print(
        f"Raw events               : "
        f"{total_events}"
    )

    # ========================================================
    # DAY 2 PREDICTIONS
    # ========================================================

    print()
    print(
        "LOADING DAY 2 LEARNED BOUNDARIES"
    )
    print("-" * 70)

    if not prediction_file.exists():

        raise FileNotFoundError(
            f"\nMissing Day 2 learned output:\n"
            f"{prediction_file}"
        )

    predicted_boundaries_df = (
        load_day2_boundaries(
            prediction_file,
            args.boundary_threshold
        )
    )

    predicted_boundaries = predicted_boundaries_df.to_dict("records")

    # ========================================================
    # GT BOUNDARIES
    #
    # Load directly from gt.jsonl because this is the
    # authoritative GT event representation.
    # ========================================================

    print()
    print(
        "LOADING GT PROCESS-SWITCH BOUNDARIES"
    )
    print("-" * 70)

    gt_boundaries = (
        load_gt_boundaries(
            dataset
        )
    )

    total_gt = sum(
        len(values)
        for values in
        gt_boundaries.values()
    )

    print(
        f"GT process-switch boundaries : "
        f"{total_gt}"
    )

    # ========================================================
    # BOUNDARY MATCHING
    # ========================================================

    print()
    print(
        "BOUNDARY MATCHING"
    )
    print("-" * 70)

    session_ids = sorted(
        set(gt_boundaries.keys())
        |
        {
            x["session_id"]
            for x in predicted_boundaries
        }
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_errors = []

    for session_id in session_ids:
        session_predictions = [
            x
            for x in predicted_boundaries
            if x["session_id"] == session_id
        ]

        session_gt = gt_boundaries.get(
            session_id,
            []
        )

        session_metrics = match_boundaries(
            session_predictions,
            session_gt,
            MATCH_TOLERANCE_SECONDS
        )

        total_tp += session_metrics["tp"]
        total_fp += session_metrics["fp"]
        total_fn += session_metrics["fn"]
        all_errors.extend(
            session_metrics["errors"]
        )

    precision = (
        total_tp / (total_tp + total_fp)
        if (total_tp + total_fp) > 0
        else 0.0
    )

    recall = (
        total_tp / (total_tp + total_fn)
        if (total_tp + total_fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    metrics = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "errors": all_errors
    }

    print(
        f"TP        : {metrics['tp']}"
    )

    print(
        f"FP        : {metrics['fp']}"
    )

    print(
        f"FN        : {metrics['fn']}"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{metrics['f1']:.4f}"
    )

    # ========================================================
    # LOCALIZATION
    # ========================================================

    if metrics["errors"]:

        errors = sorted(
            metrics["errors"]
        )

        mean_error = (
            sum(errors)
            / len(errors)
        )

        median_error = errors[
            len(errors) // 2
        ]

        p90_index = min(
            len(errors) - 1,
            int(
                0.90
                * len(errors)
            )
        )

        p90_error = errors[
            p90_index
        ]

        within_05 = (
            sum(
                e <= 0.5
                for e in errors
            )
            / len(errors)
        )

        within_1 = (
            sum(
                e <= 1.0
                for e in errors
            )
            / len(errors)
        )

        print()
        print(
            "BOUNDARY LOCALIZATION"
        )
        print("-" * 70)

        print(
            f"Mean error       : "
            f"{mean_error:.4f}s"
        )

        print(
            f"Median error     : "
            f"{median_error:.4f}s"
        )

        print(
            f"P90 error        : "
            f"{p90_error:.4f}s"
        )

        print(
            f"Within 0.5s      : "
            f"{within_05:.4f}"
        )

        print(
            f"Within 1.0s      : "
            f"{within_1:.4f}"
        )

    # ========================================================
    # SEGMENT CONSTRUCTION
    # ========================================================

    print()
    print(
        "SEGMENT CONSTRUCTION"
    )
    print("-" * 70)

    segments = construct_segments(
        events_by_session,
        predicted_boundaries
    )

    print(
        f"Constructed segments : "
        f"{len(segments)}"
    )

    # ========================================================
    # GT EXECUTION WINDOWS
    # ========================================================

    gt_windows = (
        load_gt_execution_windows(
            dataset
        )
    )

    print(
        f"Complete GT execution windows : "
        f"{len(gt_windows)}"
    )

    # ========================================================
    # OVERLAP VALIDATION
    # ========================================================

    validated = validate_segments(
        segments,
        gt_windows
    )

    total_segments = len(
        validated
    )

    overlap_50 = sum(
        x["gt_overlap"] >= 0.50
        for x in validated
    )

    overlap_80 = sum(
        x["gt_overlap"] >= 0.80
        for x in validated
    )

    print()

    if total_segments:

        print(
            f">=50% GT overlap : "
            f"{overlap_50} / "
            f"{total_segments} "
            f"("
            f"{overlap_50 / total_segments:.4f}"
            f")"
        )

        print(
            f">=80% GT overlap : "
            f"{overlap_80} / "
            f"{total_segments} "
            f"("
            f"{overlap_80 / total_segments:.4f}"
            f")"
        )

    else:

        print(
            ">=50% GT overlap : 0 / 0"
        )

        print(
            ">=80% GT overlap : 0 / 0"
        )

    # ========================================================
    # SAVE PREDICTED BOUNDARIES
    # ========================================================

    save_jsonl(
        output /
        "phase6_8_predicted_boundaries.jsonl",
        predicted_boundaries
    )

    # ========================================================
    # SAVE VALIDATED SEGMENTS
    # ========================================================

    save_jsonl(
        output /
        "phase6_8_validated_segments.jsonl",
        validated
    )

    # ========================================================
    # TEMPORARY SEGMENTS.JSONL
    #
    # Identity assignment comes after this phase.
    # ========================================================

    required_segments = []

    for index, segment in enumerate(
        segments,
        start=1
    ):

        required_segments.append({
            "session_id":
                segment["session_id"],

            "start":
                segment["start"].isoformat(),

            "end":
                segment["end"].isoformat(),

            "label":
                f"SEG_{index:05d}"
        })

    save_jsonl(
        output /
        "segments.jsonl",
        required_segments
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "phase":
            "6.8",

        "hybrid_transition_model":
            False,

        "boundary_threshold":
            args.boundary_threshold,

        "prediction_file":
            str(prediction_file),

        "gt_source":
            "Dataset A gt.jsonl process_switched_out",

        "predicted_boundaries":
            len(predicted_boundaries),

        "gt_boundaries":
            total_gt,

        "tp":
            metrics["tp"],

        "fp":
            metrics["fp"],

        "fn":
            metrics["fn"],

        "precision":
            metrics["precision"],

        "recall":
            metrics["recall"],

        "f1":
            metrics["f1"],

        "constructed_segments":
            len(segments),

        "overlap_50":
            overlap_50,

        "overlap_80":
            overlap_80
    }

    with open(
        output /
        "phase6_8_summary.json",
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
        "PHASE 6.8 COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Outputs saved to: "
        f"{output}"
    )


if __name__ == "__main__":
    main()