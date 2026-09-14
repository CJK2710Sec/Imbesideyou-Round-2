import json
import argparse
import math
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

try:
    from sklearn.feature_selection import mutual_info_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================
# TIMESTAMP
# ============================================================

def parse_ts(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        # Handle milliseconds / seconds
        if value > 10_000_000_000:
            return value / 1000.0
        return float(value)

    try:
        return datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        ).timestamp()
    except Exception:
        return None


def event_timestamp(event):
    for key in [
        "timestamp_ms",
        "ts_utc",
        "timestamp",
        "time",
        "ts",
        "event_time"
    ]:
        if key in event:
            value = parse_ts(event[key])
            if value is not None:
                return value

    return None


# ============================================================
# GENERIC FIELD EXTRACTION
# ============================================================

def event_type(event):
    return (
        event.get("event_type")
        or event.get("type")
        or event.get("event")
        or "UNKNOWN"
    )


def event_layer(event):
    return (
        event.get("layer")
        or event.get("event_layer")
        or "UNKNOWN"
    )


def get_context(event):
    context = event.get("context")

    if isinstance(context, dict):
        return context

    return {}


def get_payload(event):
    payload = event.get("payload")

    if isinstance(payload, dict):
        return payload

    return {}


def get_app(event):
    context = get_context(event)

    for key in [
        "active_app",
        "app",
        "application"
    ]:
        if context.get(key):
            return str(context[key])

    payload = get_payload(event)

    for key in [
        "active_app",
        "app",
        "application",
        "new_app"
    ]:
        if payload.get(key):
            return str(payload[key])

    for key in [
        "app",
        "application",
        "active_app"
    ]:
        if event.get(key):
            return str(event[key])

    return "UNKNOWN"


# ============================================================
# MANIFEST EXECUTIONS
# ============================================================

def load_executions(dataset_a):

    executions = []
    diagnostics = {
        "total_manifest_executions": 0,
        "missing_start_ts": 0,
        "missing_end_ts": 0,
        "invalid_start_ts": 0,
        "invalid_end_ts": 0,
        "start_after_end": 0,
        "valid_executions": 0,
        "skipped": []
    }

    manifest_files = sorted(
        Path(dataset_a).rglob("gt_manifest.json")
    )

    for manifest_path in manifest_files:

        with open(
            manifest_path,
            "r",
            encoding="utf-8"
        ) as f:

            manifest = json.load(f)

        for process in manifest.get("processes", []):

            code = process.get("code")
            family = process.get("family_name")
            domain = process.get("domain")

            for execution in process.get(
                "executions",
                []
            ):

                diagnostics["total_manifest_executions"] += 1

                start_raw = execution.get("start_ts")
                end_raw = execution.get("end_ts")

                # -----------------------------
                # Check start timestamp
                # -----------------------------

                if not start_raw:

                    diagnostics["missing_start_ts"] += 1

                    diagnostics["skipped"].append({
                        "reason": "missing_start_ts",
                        "process_code": code,
                        "case_id": execution.get("case_id"),
                        "manifest": str(manifest_path)
                    })

                    continue

                start = parse_ts(start_raw)

                if start is None:

                    diagnostics["invalid_start_ts"] += 1

                    diagnostics["skipped"].append({
                        "reason": "invalid_start_ts",
                        "process_code": code,
                        "case_id": execution.get("case_id"),
                        "start_ts": start_raw,
                        "manifest": str(manifest_path)
                    })

                    continue

                # -----------------------------
                # Check end timestamp
                # -----------------------------

                if not end_raw:

                    diagnostics["missing_end_ts"] += 1

                    diagnostics["skipped"].append({
                        "reason": "missing_end_ts",
                        "process_code": code,
                        "case_id": execution.get("case_id"),
                        "manifest": str(manifest_path)
                    })

                    continue

                end = parse_ts(end_raw)

                if end is None:

                    diagnostics["invalid_end_ts"] += 1

                    diagnostics["skipped"].append({
                        "reason": "invalid_end_ts",
                        "process_code": code,
                        "case_id": execution.get("case_id"),
                        "end_ts": end_raw,
                        "manifest": str(manifest_path)
                    })

                    continue

                # -----------------------------
                # Check ordering
                # -----------------------------

                if start >= end:

                    diagnostics["start_after_end"] += 1

                    diagnostics["skipped"].append({
                        "reason": "start_after_end",
                        "process_code": code,
                        "case_id": execution.get("case_id"),
                        "start_ts": start_raw,
                        "end_ts": end_raw,
                        "manifest": str(manifest_path)
                    })

                    continue

                # -----------------------------
                # Valid execution
                # -----------------------------

                diagnostics["valid_executions"] += 1

                executions.append({
                    "process_code": code,
                    "family_name": family,
                    "domain": domain,
                    "variant": execution.get("variant"),
                    "case_id": execution.get("case_id"),
                    "start": start,
                    "end": end,
                    "start_ts": start_raw,
                    "end_ts": end_raw,
                    "manifest": str(manifest_path)
                })

    return executions, diagnostics

# ============================================================
# RAW EVENTS
# ============================================================

def load_events(dataset_a):

    events = []

    event_files = sorted(
        Path(dataset_a).rglob("events.jsonl")
    )

    print(
        f"events.jsonl files found : "
        f"{len(event_files)}"
    )

    for path in event_files:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:
                    event = json.loads(line)
                except Exception:
                    continue

                ts = event_timestamp(event)

                if ts is None:
                    continue

                event["_ts"] = ts
                event["_source"] = str(path)

                events.append(event)

    events.sort(
        key=lambda x: x["_ts"]
    )

    return events


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def safe_entropy(counter):

    total = sum(counter.values())

    if total == 0:
        return 0.0

    entropy = 0.0

    for count in counter.values():

        p = count / total

        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def extract_features(execution, events):

    start = execution["start"]
    end = execution["end"]

    window = [
        e for e in events
        if start <= e["_ts"] <= end
    ]

    features = {
        "process_code": execution["process_code"],
        "variant": execution["variant"] or "UNKNOWN",
        "case_id": execution["case_id"],

        "duration_seconds":
            max(0.0, end - start),

        "event_count": len(window),

        "event_type_entropy": 0.0,
        "layer_entropy": 0.0,
        "app_entropy": 0.0,

        "keystroke_count": 0,
        "mouse_click_count": 0,
        "clipboard_count": 0,
        "browser_count": 0,
        "screenshot_count": 0,
        "window_count": 0,

        "unique_event_types": 0,
        "unique_layers": 0,
        "unique_apps": 0
    }

    event_types = Counter()
    layers = Counter()
    apps = Counter()

    for event in window:

        etype = event_type(event)
        layer = event_layer(event)
        app = get_app(event)

        event_types[etype] += 1
        layers[layer] += 1
        apps[app] += 1

        etype_lower = str(etype).lower()

        if "keystroke" in etype_lower:
            features["keystroke_count"] += 1

        if "click" in etype_lower or "mouse" in etype_lower:
            features["mouse_click_count"] += 1

        if "clipboard" in etype_lower:
            features["clipboard_count"] += 1

        if "browser" in etype_lower:
            features["browser_count"] += 1

        if "screenshot" in etype_lower:
            features["screenshot_count"] += 1

        if (
            "window" in etype_lower
            or "app_switch" in etype_lower
        ):
            features["window_count"] += 1

    features["unique_event_types"] = len(
        event_types
    )

    features["unique_layers"] = len(
        layers
    )

    features["unique_apps"] = len(
        apps
    )

    features["event_type_entropy"] = safe_entropy(
        event_types
    )

    features["layer_entropy"] = safe_entropy(
        layers
    )

    features["app_entropy"] = safe_entropy(
        apps
    )

    # Normalize activity counts by duration
    duration = max(
        features["duration_seconds"],
        0.001
    )

    features["events_per_second"] = (
        features["event_count"] / duration
    )

    features["keystrokes_per_second"] = (
        features["keystroke_count"] / duration
    )

    features["clicks_per_second"] = (
        features["mouse_click_count"] / duration
    )

    features["clipboard_per_second"] = (
        features["clipboard_count"] / duration
    )

    return features


# ============================================================
# PROCESS PROFILES
# ============================================================

def build_process_profiles(feature_rows):

    profiles = defaultdict(list)

    for row in feature_rows:
        profiles[
            row["process_code"]
        ].append(row)

    output = {}

    numeric_features = [
        "duration_seconds",
        "event_count",
        "events_per_second",
        "keystroke_count",
        "keystrokes_per_second",
        "mouse_click_count",
        "clicks_per_second",
        "clipboard_count",
        "clipboard_per_second",
        "browser_count",
        "screenshot_count",
        "window_count",
        "unique_event_types",
        "unique_layers",
        "unique_apps",
        "event_type_entropy",
        "layer_entropy",
        "app_entropy"
    ]

    for process, rows in profiles.items():

        profile = {
            "execution_count": len(rows),
            "features": {}
        }

        for feature in numeric_features:

            values = [
                float(r[feature])
                for r in rows
            ]

            profile["features"][feature] = {
                "mean": float(
                    np.mean(values)
                ),
                "median": float(
                    np.median(values)
                ),
                "std": float(
                    np.std(values)
                )
            }

        output[process] = profile

    return output


# ============================================================
# SIMPLE DISCRIMINATIVE SCORE
# ============================================================

def discriminative_scores(feature_rows):

    processes = sorted(
        set(
            r["process_code"]
            for r in feature_rows
        )
    )

    numeric_features = [
        "duration_seconds",
        "event_count",
        "events_per_second",
        "keystroke_count",
        "keystrokes_per_second",
        "mouse_click_count",
        "clicks_per_second",
        "clipboard_count",
        "clipboard_per_second",
        "browser_count",
        "screenshot_count",
        "window_count",
        "unique_event_types",
        "unique_layers",
        "unique_apps",
        "event_type_entropy",
        "layer_entropy",
        "app_entropy"
    ]

    results = {}

    for feature in numeric_features:

        overall = np.array([
            float(r[feature])
            for r in feature_rows
        ])

        overall_std = np.std(overall)

        if overall_std == 0:
            results[feature] = 0.0
            continue

        # Between-process variance
        process_means = []

        for process in processes:

            values = [
                float(r[feature])
                for r in feature_rows
                if r["process_code"] == process
            ]

            if values:
                process_means.append(
                    np.mean(values)
                )

        between_std = np.std(
            process_means
        )

        score = (
            between_std /
            overall_std
        )

        results[feature] = float(score)

    return dict(
        sorted(
            results.items(),
            key=lambda x: -x[1]
        )
    )


# ============================================================
# MUTUAL INFORMATION
# ============================================================

def mutual_information_scores(feature_rows):

    if not SKLEARN_AVAILABLE:
        return {}

    numeric_features = [
        "duration_seconds",
        "event_count",
        "events_per_second",
        "keystroke_count",
        "keystrokes_per_second",
        "mouse_click_count",
        "clicks_per_second",
        "clipboard_count",
        "clipboard_per_second",
        "browser_count",
        "screenshot_count",
        "window_count",
        "unique_event_types",
        "unique_layers",
        "unique_apps",
        "event_type_entropy",
        "layer_entropy",
        "app_entropy"
    ]

    X = np.array([
        [
            float(row[f])
            for f in numeric_features
        ]
        for row in feature_rows
    ])

    labels = [
        row["process_code"]
        for row in feature_rows
    ]

    label_map = {
        label: i
        for i, label in enumerate(
            sorted(set(labels))
        )
    }

    y = np.array([
        label_map[label]
        for label in labels
    ])

    scores = mutual_info_classif(
        X,
        y,
        random_state=42
    )

    return dict(
        sorted(
            {
                feature: float(score)
                for feature, score in zip(
                    numeric_features,
                    scores
                )
            }.items(),
            key=lambda x: -x[1]
        )
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

    args = parser.parse_args()

    dataset_a = Path(
        args.dataset_a
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not dataset_a.exists():

        raise FileNotFoundError(
            f"Dataset A not found: {dataset_a}"
        )

    print("=" * 60)
    print("PHASE 6.3 - PROCESS IDENTITY FEATURES")
    print("=" * 60)

    # --------------------------------------------------------
    # Load GT executions
    # --------------------------------------------------------

    executions, diagnostics = load_executions(
    dataset_a
    )

    print(
    f"GT executions loaded : "
    f"{len(executions)}"
    )

    print()
    print("EXECUTION TIMESTAMP DIAGNOSTICS")
    print("-" * 40)

    print(
    f"Manifest executions : "
    f"{diagnostics['total_manifest_executions']}"
    )

    print(
    f"Valid executions    : "
    f"{diagnostics['valid_executions']}"
    )

    print(
    f"Missing start_ts    : "
    f"{diagnostics['missing_start_ts']}"
    )

    print(
    f"Missing end_ts      : "
    f"{diagnostics['missing_end_ts']}"
    )

    print(
    f"Invalid start_ts    : "
    f"{diagnostics['invalid_start_ts']}"
    )

    print(
    f"Invalid end_ts      : "
    f"{diagnostics['invalid_end_ts']}"
    )

    print(
    f"Start >= end        : "
    f"{diagnostics['start_after_end']}"
    )

    print(
    f"Total skipped       : "
    f"{len(diagnostics['skipped'])}"
    )

    with open(
        output_dir /
        "phase6_3_missing_execution_diagnostics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            diagnostics,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ============================================================
    # INCOMPLETE EXECUTION DIAGNOSTICS
    # ============================================================

    from collections import Counter

    missing_end_records = []

    manifest_files = sorted(
    Path(dataset_a).rglob("gt_manifest.json")
    )

    for manifest_path in manifest_files:

        with open(
            manifest_path,
            "r",
            encoding="utf-8"
        ) as f:
            manifest = json.load(f)

        for process in manifest.get("processes", []):

            for execution in process.get("executions", []):

                if execution.get("end_ts") is not None:
                    continue

                missing_end_records.append({
                    "process_code": process.get("code"),
                    "family_name": process.get("family_name"),
                    "domain": process.get("domain"),
                    "variant": execution.get("variant"),
                    "case_id": execution.get("case_id"),
                    "start_ts": execution.get("start_ts"),
                    "end_ts": execution.get("end_ts"),
                    "continues_from_prev": execution.get(
                        "continues_from_prev"
                    ),
                    "continues_to_next": execution.get(
                        "continues_to_next"
                    ),
                    "phase": execution.get("phase"),
                    "seq": execution.get("seq"),
                    "manifest": str(manifest_path)
                })


    print() 
    print("INCOMPLETE EXECUTION DIAGNOSTICS")
    print("=" * 60)

    print(
    f"Missing end_ts executions : "
    f"{len(missing_end_records)}"
    )


    # ------------------------------------------------------------
    # BY PROCESS
    # ------------------------------------------------------------

    by_process = Counter(
    x["process_code"]
    for x in missing_end_records
    )

    print()
    print("MISSING END_TS BY PROCESS")
    print("-" * 40)

    for process, count in by_process.most_common():

        print(
            f"{str(process):<25} : {count}"
        )


    # ------------------------------------------------------------
    # BY VARIANT
    # ------------------------------------------------------------

    by_variant = Counter(
    str(x["variant"])
    for x in missing_end_records
    )

    print()
    print("MISSING END_TS BY VARIANT")
    print("-" * 40)

    for variant, count in by_variant.most_common():

        print(
            f"{variant:<25} : {count}"
        )


    # ------------------------------------------------------------
    # CONTINUATION STATUS
    # ------------------------------------------------------------

    by_continuation = Counter(
        str(x["continues_to_next"])
        for x in missing_end_records
    )

    print()
    print("MISSING END_TS BY CONTINUES_TO_NEXT")
    print("-" * 40)

    for value, count in by_continuation.most_common():

        print(
            f"{value:<10} : {count}"
        )


    # ------------------------------------------------------------
    # PREVIOUS CONTINUATION STATUS
    # ------------------------------------------------------------

    by_previous = Counter(
        str(x["continues_from_prev"])
        for x in missing_end_records
    )

    print()
    print("MISSING END_TS BY CONTINUES_FROM_PREV")
    print("-" * 40)

    for value, count in by_previous.most_common():

        print(
            f"{value:<10} : {count}"
        )


    # ------------------------------------------------------------
    # PHASE
    # ------------------------------------------------------------

    by_phase = Counter(
        str(x["phase"])
        for x in missing_end_records
    )

    print()
    print("MISSING END_TS BY PHASE")
    print("-" * 40)

    for phase, count in by_phase.most_common():

        print(
            f"{phase:<20} : {count}"
        )


    # ------------------------------------------------------------
    # SAVE FULL DIAGNOSTIC DATA
    # ------------------------------------------------------------

    with open(
        output_dir /
        "phase6_3_incomplete_execution_details.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            missing_end_records,
            f,
            ensure_ascii=False,
            indent=2
        )
    
    # --------------------------------------------------------
    # Load raw events
    # --------------------------------------------------------

    events = load_events(
        dataset_a
    )

    print(
        f"Raw events loaded    : "
        f"{len(events)}"
    )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    feature_rows = []

    for i, execution in enumerate(
        executions,
        start=1
    ):

        features = extract_features(
            execution,
            events
        )

        feature_rows.append(
            features
        )

        if i % 250 == 0:
            print(
                f"Processed executions : "
                f"{i}/{len(executions)}"
            )

    # --------------------------------------------------------
    # Profiles
    # --------------------------------------------------------

    profiles = build_process_profiles(
        feature_rows
    )

    # --------------------------------------------------------
    # Discriminative scores
    # --------------------------------------------------------

    discrimination = discriminative_scores(
        feature_rows
    )

    mi_scores = mutual_information_scores(
        feature_rows
    )

    # --------------------------------------------------------
    # Save feature rows
    # --------------------------------------------------------

    with open(
        output_dir /
        "phase6_3_execution_features.jsonl",
        "w",
        encoding="utf-8"
    ) as f:

        for row in feature_rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    # --------------------------------------------------------
    # Save profiles
    # --------------------------------------------------------

    with open(
        output_dir /
        "phase6_3_process_profiles.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            profiles,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Save discriminative scores
    # --------------------------------------------------------

    with open(
        output_dir /
        "phase6_3_feature_scores.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "between_process_variance_ratio":
                    discrimination,
                "mutual_information":
                    mi_scores
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_path = (
        output_dir /
        "day3_phase6_3_report.md"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# Day 3 — Phase 6.3\n\n"
        )

        f.write(
            "## Process Identity Feature Analysis\n\n"
        )

        f.write(
            f"- GT executions: {len(executions)}\n"
        )

        f.write(
            f"- Raw events: {len(events)}\n"
        )

        f.write(
            f"- Process families: "
            f"{len(profiles)}\n\n"
        )

        f.write(
            "## Most Discriminative Features\n\n"
        )

        f.write(
            "| Rank | Feature | Between-process score |\n"
        )

        f.write(
            "|---:|---|---:|\n"
        )

        for rank, (
            feature,
            score
        ) in enumerate(
            discrimination.items(),
            start=1
        ):

            f.write(
                f"| {rank} | "
                f"{feature} | "
                f"{score:.4f} |\n"
            )

        if mi_scores:

            f.write(
                "\n## Mutual Information\n\n"
            )

            f.write(
                "| Rank | Feature | MI score |\n"
            )

            f.write(
                "|---:|---|---:|\n"
            )

            for rank, (
                feature,
                score
            ) in enumerate(
                mi_scores.items(),
                start=1
            ):

                f.write(
                    f"| {rank} | "
                    f"{feature} | "
                    f"{score:.4f} |\n"
                )

        f.write(
            "\n## Interpretation\n\n"
        )

        f.write(
            "The analysis measures whether behavioral "
            "features extracted from raw operation logs "
            "differ systematically between logical "
            "process families. Features with high "
            "between-process variance or mutual "
            "information are candidates for process "
            "identity modeling. These scores are "
            "descriptive and are not themselves a final "
            "classification accuracy measure.\n"
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PHASE 6.3 COMPLETE")
    print("=" * 60)

    print(
        f"GT executions analyzed : "
        f"{len(executions)}"
    )

    print(
        f"Raw events analyzed    : "
        f"{len(events)}"
    )

    print(
        f"Process families       : "
        f"{len(profiles)}"
    )

    print()
    print("Top identity features:")

    for rank, (
        feature,
        score
    ) in enumerate(
        list(discrimination.items())[:10],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"{feature:<30} "
            f"{score:.4f}"
        )

    print()
    print(
        f"Outputs written to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()