import json
import argparse
import bisect
import csv
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EVENT_TYPES = [
    "app_switch", "keystroke", "shortcut", "mouse_click",
    "mouse_double_click", "mouse_scroll", "mouse_drag_drop",
    "clipboard_change", "text_input_complete", "window_title_change",
    "window_state_change", "dialog_opened", "dialog_closed",
    "screenshot_smart", "browser_click", "browser_form_input",
    "browser_navigation", "browser_tab_event", "browser_alert",
    "browser_error",
]
LAYERS = ["SYSTEM", "L1", "L2", "L3"]
NGRAM_SIZES = (2, 3)
TOP_NGRAMS = 80


def parse_args():
    p = argparse.ArgumentParser(
        description="Day 3 Phase 6.5 - Sequence + Context Identity Analysis"
    )
    p.add_argument("--dataset-a", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--folds", type=int, default=5)
    return p.parse_args()


def parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def iso_to_ms(value):
    dt = parse_ts(value)
    return None if dt is None else int(dt.timestamp() * 1000)


def find_manifest_files(dataset_a):
    return sorted(Path(dataset_a).rglob("gt_manifest.json"))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# IMPORTANT:
# This is copied from the proven Phase 6.2 GT extraction logic.
# gt_manifest.json is structured as:
#   session -> processes -> executions
# Do NOT flatten the top-level "processes" as execution records.
def extract_executions(manifest, manifest_path):
    executions = []

    session_info = manifest.get("session", {})
    session_start = session_info.get("start_ts")
    session_end = session_info.get("end_ts")

    for process in manifest.get("processes", []):
        process_code = process.get("code")
        family_name = process.get("family_name")
        domain = process.get("domain")

        for execution in process.get("executions", []):
            executions.append({
                "process_code": process_code,
                "family_name": family_name,
                "domain": domain,
                "variant": execution.get("variant"),
                "case_id": execution.get("case_id"),
                "start_ts": execution.get("start_ts"),
                "end_ts": execution.get("end_ts"),
                "start": parse_ts(execution.get("start_ts")),
                "end": parse_ts(execution.get("end_ts")),
                "phase": execution.get("phase"),
                "seq": execution.get("seq"),
                "continues_from_prev": execution.get(
                    "continues_from_prev", False
                ),
                "continues_to_next": execution.get(
                    "continues_to_next", False
                ),
                "apps": execution.get("apps", []),
                "session_start": session_start,
                "session_end": session_end,
                "source_file": str(manifest_path),
            })

    return executions


def load_manifest(dataset):
    manifests = []
    paths = find_manifest_files(dataset)

    print()
    print("MANIFEST DISCOVERY")
    print("------------------")
    print("Manifest files found :", len(paths))

    for path in paths:
        try:
            manifests.append((path, extract_executions(load_json(path), path)))
        except Exception as exc:
            print(f"WARNING: failed to read {path}: {exc}")

    return manifests


def load_events(dataset):
    events_by_session = {}

    for session_path in sorted(Path(dataset).glob("ses_*")):
        session_id = session_path.name
        events = []

        for event_path in sorted(session_path.glob("chunk_*/events.jsonl")):
            try:
                with event_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            event = json.loads(line)
                        except Exception:
                            continue

                        ts = iso_to_ms(event.get("timestamp_iso"))
                        if ts is None:
                            continue

                        event["_ts_ms"] = ts
                        events.append(event)
            except Exception as exc:
                print(f"WARNING reading {event_path}: {exc}")

        events.sort(key=lambda x: x["_ts_ms"])
        events_by_session[session_id] = events

    return events_by_session


def get_active_app(event):
    context = event.get("context") or {}
    active = context.get("active_app") or {}

    if isinstance(active, dict):
        return (
            active.get("app_name")
            or active.get("process_name")
            or "UNKNOWN_APP"
        )

    return "UNKNOWN_APP"


def make_ngrams(values, n):
    if len(values) < n:
        return []
    return list(zip(*(values[i:] for i in range(n))))


def collect_vocab(events_by_session):
    counters = {
        kind: {n: Counter() for n in NGRAM_SIZES}
        for kind in ("event", "app", "layer")
    }

    for events in events_by_session.values():
        event_types = [
            e.get("event_type", "UNKNOWN") for e in events
        ]
        apps = [get_active_app(e) for e in events]
        layers = [e.get("layer", "UNKNOWN") for e in events]

        for n in NGRAM_SIZES:
            counters["event"][n].update(make_ngrams(event_types, n))
            counters["app"][n].update(make_ngrams(apps, n))
            counters["layer"][n].update(make_ngrams(layers, n))

    return {
        kind: {
            n: [gram for gram, _ in counters[kind][n].most_common(TOP_NGRAMS)]
            for n in NGRAM_SIZES
        }
        for kind in counters
    }


def entropy(counter):
    total = sum(counter.values())
    if total <= 1:
        return 0.0

    result = 0.0
    for count in counter.values():
        p = count / total
        if p > 0:
            result -= p * math.log2(p)
    return result


def make_features(events, timestamp_array, start_ms, end_ms, vocab):
    left = bisect.bisect_left(timestamp_array, start_ms)
    right = bisect.bisect_right(timestamp_array, end_ms)
    segment_events = events[left:right]

    if not segment_events:
        return None

    event_types = [
        e.get("event_type", "UNKNOWN") for e in segment_events
    ]
    apps = [get_active_app(e) for e in segment_events]
    layers = [e.get("layer", "UNKNOWN") for e in segment_events]

    duration = max((end_ms - start_ms) / 1000.0, 0.001)
    count = len(segment_events)

    f = {
        "duration_log": math.log1p(duration),
        "event_count_log": math.log1p(count),
        "event_rate": count / duration,
        "unique_event_types": len(set(event_types)),
        "event_type_entropy": entropy(Counter(event_types)),
        "unique_apps": len(set(apps)),
        "app_entropy": entropy(Counter(apps)),
        "unique_layers": len(set(layers)),
        "layer_entropy": entropy(Counter(layers)),
    }

    event_counter = Counter(event_types)
    for event_type in EVENT_TYPES:
        f[f"type_ratio__{event_type}"] = event_counter[event_type] / count

    layer_counter = Counter(layers)
    for layer in LAYERS:
        f[f"layer_ratio__{layer}"] = layer_counter[layer] / count

    event_pairs = make_ngrams(event_types, 2)
    app_pairs = make_ngrams(apps, 2)
    layer_pairs = make_ngrams(layers, 2)

    f["event_transition_diversity"] = len(set(event_pairs))
    f["app_transition_diversity"] = len(set(app_pairs))
    f["layer_transition_diversity"] = len(set(layer_pairs))

    for kind, values in (
        ("event", event_types),
        ("app", apps),
        ("layer", layers),
    ):
        for n in NGRAM_SIZES:
            grams = make_ngrams(values, n)
            denominator = max(len(grams), 1)
            counter = Counter(grams)

            for i, gram in enumerate(vocab[kind][n]):
                f[f"{kind}_ngram_{n}_{i}"] = (
                    counter[tuple(gram)] / denominator
                )

    if len(segment_events) >= 2:
        timestamps = np.asarray(
            [e["_ts_ms"] for e in segment_events],
            dtype=float,
        )
        gaps = np.diff(timestamps) / 1000.0
        f["median_event_gap"] = float(np.median(gaps))
        f["p90_event_gap"] = float(np.percentile(gaps, 90))
        f["max_event_gap"] = float(np.max(gaps))
    else:
        f["median_event_gap"] = duration
        f["p90_event_gap"] = duration
        f["max_event_gap"] = duration

    motifs = {
        "click_to_keystroke": ("mouse_click", "keystroke"),
        "keystroke_to_click": ("keystroke", "mouse_click"),
        "clipboard_to_keystroke": ("clipboard_change", "keystroke"),
        "keystroke_to_clipboard": ("keystroke", "clipboard_change"),
        "browser_input_to_click": (
            "browser_form_input", "browser_click"
        ),
        "click_to_navigation": (
            "browser_click", "browser_navigation"
        ),
        "navigation_to_input": (
            "browser_navigation", "browser_form_input"
        ),
        "app_switch_to_click": (
            "app_switch", "mouse_click"
        ),
    }

    pair_count = max(len(event_pairs), 1)
    for name, pair in motifs.items():
        f[f"motif__{name}"] = event_pairs.count(pair) / pair_count

    # Deterministic categorical encoding is preferable to Python's hash(),
    # whose randomization can change across processes.
    def stable_code(value):
        return sum((i + 1) * ord(c) for i, c in enumerate(str(value))) % 997

    f["first_event_hash"] = stable_code(event_types[0])
    f["last_event_hash"] = stable_code(event_types[-1])
    f["first_app_hash"] = stable_code(apps[0])
    f["last_app_hash"] = stable_code(apps[-1])

    return f


def build_feature_dataset(manifests, events_by_session, vocab):
    rows = []
    diagnostics = Counter()
    manifest_total = 0

    for manifest_path, executions in manifests:
        session_id = manifest_path.parent.name
        manifest_total += len(executions)

        events = events_by_session.get(session_id, [])
        if not events:
            diagnostics["session_without_events"] += len(executions)
            continue

        timestamp_array = [e["_ts_ms"] for e in events]

        for index, record in enumerate(executions):
            start_value = record.get("start_ts")
            end_value = record.get("end_ts")
            label = record.get("process_code")

            if start_value is None:
                diagnostics["missing_start"] += 1
                continue
            if end_value is None:
                diagnostics["missing_end"] += 1
                continue
            if label is None:
                diagnostics["missing_label"] += 1
                continue

            start_ms = iso_to_ms(start_value)
            end_ms = iso_to_ms(end_value)

            if start_ms is None:
                diagnostics["invalid_start"] += 1
                continue
            if end_ms is None:
                diagnostics["invalid_end"] += 1
                continue
            if start_ms >= end_ms:
                diagnostics["invalid_interval"] += 1
                continue

            features = make_features(
                events, timestamp_array, start_ms, end_ms, vocab
            )
            if not features:
                diagnostics["no_events_inside_window"] += 1
                continue

            row = dict(features)
            row["session_id"] = session_id
            row["execution_index"] = index
            row["label"] = label
            row["variant"] = record.get("variant") or "UNKNOWN"
            rows.append(row)

    diagnostics["total_manifest_records"] = manifest_total
    diagnostics["usable_feature_rows"] = len(rows)
    return rows, diagnostics


def evaluate_model(rows, feature_names, model, folds):
    X = np.asarray(
        [[float(row.get(feature, 0.0)) for feature in feature_names]
         for row in rows],
        dtype=float,
    )
    y = np.asarray([row["label"] for row in rows])
    groups = np.asarray([row["session_id"] for row in rows])

    n_splits = min(folds, len(set(groups)))
    if n_splits < 2:
        raise ValueError("Need at least 2 session groups for CV.")

    splitter = GroupKFold(n_splits=n_splits)
    all_true, all_pred, fold_results = [], [], []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        all_true.extend(y[test_idx])
        all_pred.extend(pred)

        fold_results.append({
            "fold": fold,
            "accuracy": float(
                accuracy_score(y[test_idx], pred)
            ),
            "macro_f1": float(
                f1_score(
                    y[test_idx], pred,
                    average="macro", zero_division=0
                )
            ),
            "weighted_f1": float(
                f1_score(
                    y[test_idx], pred,
                    average="weighted", zero_division=0
                )
            ),
        })

    return {
        "accuracy": float(accuracy_score(all_true, all_pred)),
        "macro_f1": float(
            f1_score(
                all_true, all_pred,
                average="macro", zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                all_true, all_pred,
                average="weighted", zero_division=0
            )
        ),
        "folds": fold_results,
        "y_true": all_true,
        "y_pred": all_pred,
    }


def main():
    args = parse_args()
    dataset = Path(args.dataset_a)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 6.5 - SEQUENCE + CONTEXT IDENTITY ANALYSIS")
    print("=" * 60)

    manifests = load_manifest(dataset)
    events_by_session = load_events(dataset)

    print("Sessions with GT manifest :", len(manifests))
    print("Sessions with raw events  :", len(events_by_session))

    manifest_total = sum(len(x[1]) for x in manifests)
    print("Manifest executions found :", manifest_total)

    # Expected result from Phase 6.2: 2009 executions across 15 families.
    if manifest_total != 2009:
        print()
        print(
            "WARNING: Phase 6.2 expected 2009 GT executions, "
            f"but this run extracted {manifest_total}."
        )
        print(
            "The script is using the same session -> process -> "
            "execution extraction logic as Phase 6.2."
        )

    vocab = collect_vocab(events_by_session)
    rows, diagnostics = build_feature_dataset(
        manifests, events_by_session, vocab
    )

    print()
    print("FEATURE EXTRACTION DIAGNOSTICS")
    print("------------------------------")
    for key, value in diagnostics.items():
        print(f"{key:30s}: {value}")

    if not rows:
        print()
        print("ERROR: Zero usable executions after feature extraction.")
        return

    labels = sorted({row["label"] for row in rows})
    sessions = sorted({row["session_id"] for row in rows})

    excluded = {
        "session_id", "execution_index", "label", "variant"
    }
    feature_names = sorted({
        key
        for row in rows
        for key, value in row.items()
        if key not in excluded and isinstance(value, (int, float))
    })

    sequence_features = [
        feature for feature in feature_names
        if (
            "ngram" in feature
            or "transition" in feature
            or "motif__" in feature
            or feature in {
                "median_event_gap", "p90_event_gap",
                "max_event_gap", "first_event_hash",
                "last_event_hash", "first_app_hash",
                "last_app_hash",
            }
        )
    ]

    context_features = [
        feature for feature in feature_names
        if (
            feature.startswith("app_ngram_")
            or feature.startswith("layer_ngram_")
            or "app_transition" in feature
            or "layer_transition" in feature
        )
    ]

    combined_features = sorted(
        set(sequence_features + context_features)
    )

    print()
    print("=" * 60)
    print("PHASE 6.5 DATASET")
    print("=" * 60)
    print("Complete timed executions :", len(rows))
    print("Process classes            :", len(labels))
    print("Sessions/groups            :", len(sessions))
    print("Total numeric features     :", len(feature_names))
    print("Sequence features          :", len(sequence_features))
    print("Combined features          :", len(combined_features))

    models = {
        "sequence_only_logistic": (
            sequence_features,
            Pipeline([
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(
                    max_iter=3000,
                    C=1.0,
                    class_weight="balanced",
                )),
            ]),
        ),
        "sequence_context_random_forest": (
            combined_features,
            RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
            ),
        ),
    }

    results = {}
    first_confusion = None

    for name, (features, model) in models.items():
        if not features:
            print(f"\n{name}: SKIPPED - no features")
            continue

        print()
        print(name)
        print("Features :", len(features))

        result = evaluate_model(
            rows, features, model, args.folds
        )

        results[name] = {
            k: v for k, v in result.items()
            if k not in {"y_true", "y_pred"}
        }

        print(f"Accuracy    : {result['accuracy']:.4f}")
        print(f"Macro F1    : {result['macro_f1']:.4f}")
        print(f"Weighted F1 : {result['weighted_f1']:.4f}")

        if first_confusion is None:
            first_confusion = (
                result["y_true"], result["y_pred"]
            )

    # Save features.
    feature_csv = output / "phase6_5_sequence_features.csv"
    with feature_csv.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "session_id", "execution_index",
            "label", "variant"
        ] + feature_names
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: row.get(key, 0) for key in fields
            })

    metrics = {
        "dataset": str(dataset),
        "manifest_executions": manifest_total,
        "complete_timed_executions": len(rows),
        "process_classes": labels,
        "session_count": len(sessions),
        "feature_count": len(feature_names),
        "sequence_feature_count": len(sequence_features),
        "combined_feature_count": len(combined_features),
        "diagnostics": dict(diagnostics),
        "models": results,
    }

    (output / "phase6_5_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    if first_confusion is not None:
        y_true, y_pred = first_confusion
        matrix = confusion_matrix(
            y_true, y_pred, labels=labels
        )

        with (output / "phase6_5_confusion_matrix.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["true\\pred"] + labels)
            for label, values in zip(labels, matrix):
                writer.writerow([label] + values.tolist())

    best_model = None
    for name, result in results.items():
        if (
            best_model is None
            or result["macro_f1"] > best_model[1]["macro_f1"]
        ):
            best_model = (name, result)

    summary = [
        "# Phase 6.5 — Sequence + Context Identity Analysis",
        "",
        "## Objective",
        "",
        "Test whether local event ordering and short behavioral "
        "patterns provide additional logical process identity "
        "information beyond aggregate execution-level features.",
        "",
        "## Evaluation",
        "",
        f"- Manifest executions: {manifest_total}",
        f"- Complete timed executions: {len(rows)}",
        f"- Process families: {len(labels)}",
        f"- Session groups: {len(sessions)}",
        "- Session-grouped cross-validation",
        "- GT process labels used only as targets",
        "- No previous/next GT process labels used as features",
        "",
        "## Results",
        "",
    ]

    for name, result in results.items():
        summary += [
            f"### {name}",
            f"- Accuracy: {result['accuracy']:.4f}",
            f"- Macro F1: {result['macro_f1']:.4f}",
            f"- Weighted F1: {result['weighted_f1']:.4f}",
            "",
        ]

    if best_model:
        summary += [
            "## Best model",
            f"- Model: {best_model[0]}",
            f"- Accuracy: {best_model[1]['accuracy']:.4f}",
            f"- Macro F1: {best_model[1]['macro_f1']:.4f}",
            "",
        ]

    summary += [
        "## Interpretation",
        "",
        "Phase 6.5 evaluates whether local event ordering, "
        "transition structure, short n-grams, and interaction "
        "motifs add process-identity information beyond "
        "whole-execution aggregates.",
        "",
        "The result should be compared with the Phase 6.4 "
        "behavioral Random Forest baseline "
        "(29.28% accuracy, 0.2828 Macro F1).",
        "",
        "A meaningful improvement supports incorporating local "
        "sequence/context information into final identity and "
        "segmentation. A weak improvement supports a conservative "
        "identity mechanism based on process history, transition "
        "structure, variants, and confidence rather than "
        "standalone classification.",
    ]

    (output / "phase6_5_summary.md").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("PHASE 6.5 COMPLETE")
    print("=" * 60)

    if best_model:
        print("Best model :", best_model[0])
        print(f"Accuracy   : {best_model[1]['accuracy']:.4f}")
        print(f"Macro F1   : {best_model[1]['macro_f1']:.4f}")

    print()
    print("Outputs written to:", output)


if __name__ == "__main__":
    main()
