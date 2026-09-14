import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold


# ============================================================
# PHASE 6.6
# TRANSITION-WEIGHT SWEEP FOR IDENTITY RESOLUTION
# ============================================================

WEIGHTS = [
    0.80,
    0.85,
    0.90,
    0.95,
    1.00,
    1.05,
    1.10,
    1.15,
    1.20,
    1.30,
    1.40,
    1.50
]

RANDOM_STATE = 42
N_SPLITS = 5


# ============================================================
# UTILITIES
# ============================================================

def parse_ts(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except Exception:
        return None


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_manifest_files(dataset_a):
    return sorted(
        Path(dataset_a).rglob("gt_manifest.json")
    )


# ============================================================
# EXACT GT MANIFEST EXTRACTION
# ============================================================

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

            item = {
                "process_code": process_code,
                "family_name": family_name,
                "domain": domain,
                "variant": execution.get("variant"),
                "case_id": execution.get("case_id"),

                "start_ts": execution.get("start_ts"),
                "end_ts": execution.get("end_ts"),

                "start": parse_ts(
                    execution.get("start_ts")
                ),

                "end": parse_ts(
                    execution.get("end_ts")
                ),

                "phase": execution.get("phase"),
                "seq": execution.get("seq"),

                "continues_from_prev":
                    execution.get(
                        "continues_from_prev",
                        False
                    ),

                "continues_to_next":
                    execution.get(
                        "continues_to_next",
                        False
                    ),

                "apps": execution.get("apps", []),

                "session_start": session_start,
                "session_end": session_end,

                "source_file": str(manifest_path)
            }

            executions.append(item)

    return executions


# ============================================================
# RAW EVENT LOADING
# ============================================================

def find_event_files(dataset_a):
    return sorted(
        Path(dataset_a).rglob("events.jsonl")
    )


def load_events(path):

    events = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                event = json.loads(line)

                ts = parse_ts(
                    event.get("timestamp_iso")
                )

                if ts is not None:
                    event["_parsed_ts"] = ts
                    events.append(event)

            except Exception:
                continue

    events.sort(
        key=lambda x: x["_parsed_ts"]
    )

    return events


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def entropy_from_counts(counter):

    total = sum(counter.values())

    if total == 0:
        return 0.0

    result = 0.0

    for count in counter.values():

        if count <= 0:
            continue

        p = count / total
        result -= p * math.log2(p)

    return result


def active_app_name(event):

    context = event.get("context", {})

    active_app = context.get(
        "active_app",
        {}
    )

    if isinstance(active_app, dict):

        return (
            active_app.get("app_name")
            or active_app.get("process_name")
            or "UNKNOWN"
        )

    return "UNKNOWN"


def extract_features(events, execution):

    start = execution["start"]
    end = execution["end"]

    if start is None or end is None:
        return None

    selected = [
        e for e in events
        if start <= e["_parsed_ts"] <= end
    ]

    if not selected:
        return None

    duration = (
        end - start
    ).total_seconds()

    duration = max(duration, 0.001)

    event_count = len(selected)

    event_types = Counter()
    layers = Counter()
    apps = Counter()

    counts = Counter()

    for event in selected:

        etype = event.get(
            "event_type",
            "UNKNOWN"
        )

        layer = event.get(
            "layer",
            "UNKNOWN"
        )

        app = active_app_name(event)

        event_types[etype] += 1
        layers[layer] += 1
        apps[app] += 1

        counts[etype] += 1

    features = {}

    # --------------------------------------------------------
    # Basic logarithmic features
    # --------------------------------------------------------

    features["duration_log"] = math.log1p(
        duration
    )

    features["event_count_log"] = math.log1p(
        event_count
    )

    features["event_rate"] = (
        event_count / duration
    )

    # --------------------------------------------------------
    # Diversity
    # --------------------------------------------------------

    features["unique_event_types"] = len(
        event_types
    )

    features["unique_layers"] = len(
        layers
    )

    features["unique_apps"] = len(
        apps
    )

    features["event_type_entropy"] = (
        entropy_from_counts(event_types)
    )

    features["layer_entropy"] = (
        entropy_from_counts(layers)
    )

    features["app_entropy"] = (
        entropy_from_counts(apps)
    )

    # --------------------------------------------------------
    # Event rates
    # --------------------------------------------------------

    rate_events = {
        "keystroke_rate": [
            "keystroke"
        ],

        "shortcut_rate": [
            "shortcut"
        ],

        "click_rate": [
            "mouse_click",
            "mouse_double_click"
        ],

        "scroll_rate": [
            "scroll"
        ],

        "drag_drop_rate": [
            "drag_drop"
        ],

        "clipboard_rate": [
            "clipboard_change"
        ],

        "screenshot_rate": [
            "screenshot_smart"
        ],

        "app_switch_rate": [
            "app_switch"
        ],

        "window_rate": [
            "window_change",
            "window_open",
            "window_close"
        ],

        "dialog_rate": [
            "dialog_open",
            "dialog_close"
        ],

        "browser_click_rate": [
            "browser_click"
        ],

        "browser_input_rate": [
            "browser_form_input"
        ],

        "browser_navigation_rate": [
            "browser_navigation"
        ],

        "browser_tab_rate": [
            "browser_tab"
        ]
    }

    for feature_name, types in rate_events.items():

        count = sum(
            counts[t]
            for t in types
        )

        features[feature_name] = (
            count / duration
        )

    # --------------------------------------------------------
    # Type composition ratios
    # --------------------------------------------------------

    if event_count > 0:

        for etype, count in counts.items():

            safe_name = (
                str(etype)
                .replace(" ", "_")
                .replace("/", "_")
            )

            features[
                f"type_ratio__{safe_name}"
            ] = count / event_count

    # --------------------------------------------------------
    # Layer composition
    # --------------------------------------------------------

    if event_count > 0:

        for layer, count in layers.items():

            safe_name = (
                str(layer)
                .replace(" ", "_")
                .replace("/", "_")
            )

            features[
                f"layer_ratio__{safe_name}"
            ] = count / event_count

    # --------------------------------------------------------
    # App composition
    # --------------------------------------------------------

    if event_count > 0:

        for app, count in apps.items():

            safe_name = (
                str(app)
                .replace(" ", "_")
                .replace("/", "_")
            )

            features[
                f"app_ratio__{safe_name}"
            ] = count / event_count

    return features


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(dataset_a):

    manifest_files = find_manifest_files(
        dataset_a
    )

    print(
        f"Manifest files found : "
        f"{len(manifest_files)}"
    )

    all_executions = []

    for manifest_path in manifest_files:

        manifest = load_json(
            manifest_path
        )

        session_executions = (
            extract_executions(
                manifest,
                manifest_path
            )
        )

        all_executions.extend(
            session_executions
        )

    print(
        f"GT executions found : "
        f"{len(all_executions)}"
    )

    event_files = find_event_files(
        dataset_a
    )

    events_by_session = {}

    for event_file in event_files:

        events = load_events(
            event_file
        )

        if not events:
            continue

        session_id = events[0].get(
            "session_id"
        )

        if session_id is not None:
            events_by_session[
                str(session_id)
            ] = events

    print(
        f"Sessions with raw events : "
        f"{len(events_by_session)}"
    )

    rows = []

    diagnostics = Counter()

    for execution in all_executions:

        if execution["end"] is None:

            diagnostics[
                "missing_end"
            ] += 1

            continue

        if execution["start"] is None:

            diagnostics[
                "missing_start"
            ] += 1

            continue

        if execution["start"] >= execution["end"]:

            diagnostics[
                "invalid_window"
            ] += 1

            continue

        session_id = None

        source = Path(
            execution["source_file"]
        )

        # ----------------------------------------------------
        # Session ID from manifest path / events
        # ----------------------------------------------------

        possible = []

        for sid in events_by_session:

            if sid in str(source):
                possible.append(sid)

        if possible:

            session_id = possible[0]

        else:

            # Match using session timestamps
            for sid, events in events_by_session.items():

                if not events:
                    continue

                first_ts = events[0]["_parsed_ts"]
                last_ts = events[-1]["_parsed_ts"]

                if (
                    first_ts <= execution["start"]
                    and last_ts >= execution["end"]
                ):
                    session_id = sid
                    break

        if session_id is None:

            diagnostics[
                "session_not_found"
            ] += 1

            continue

        events = events_by_session[
            session_id
        ]

        features = extract_features(
            events,
            execution
        )

        if features is None:

            diagnostics[
                "no_events_inside_window"
            ] += 1

            continue

        row = dict(features)

        row["process_code"] = (
            execution["process_code"]
        )

        row["session_id"] = session_id

        rows.append(row)

    print("\nFEATURE EXTRACTION DIAGNOSTICS")

    for key, value in diagnostics.items():

        print(
            f"{key:30s}: {value}"
        )

    df = pd.DataFrame(rows)

    return df


# ============================================================
# TRANSITION PROBABILITIES
# ============================================================

def learn_transition_matrix(
    train_df,
    labels
):

    transition_counts = defaultdict(
        Counter
    )

    # --------------------------------------------------------
    # Group executions by session and preserve temporal order
    # --------------------------------------------------------

    for session_id, group in train_df.groupby(
        "session_id"
    ):

        sequence = list(
            group["process_code"]
        )

        for a, b in zip(
            sequence[:-1],
            sequence[1:]
        ):

            transition_counts[a][b] += 1

    matrix = {}

    smoothing = 1.0

    n_classes = len(labels)

    for a in labels:

        total = sum(
            transition_counts[a].values()
        )

        matrix[a] = {}

        for b in labels:

            count = transition_counts[a][b]

            probability = (
                count + smoothing
            ) / (
                total
                + smoothing * n_classes
            )

            matrix[a][b] = probability

    return matrix


# ============================================================
# TRANSITION-AWARE DECODING
# ============================================================

def transition_decode(
    probabilities,
    classes,
    session_ids,
    transition_matrix,
    weight
):

    predictions = []

    previous_prediction = {}

    class_list = list(classes)

    for i in range(
        len(probabilities)
    ):

        session = session_ids[i]

        probs = probabilities[i]

        best_class = None
        best_score = -float("inf")

        prev = previous_prediction.get(
            session
        )

        for j, cls in enumerate(
            class_list
        ):

            base_score = math.log(
                max(
                    probs[j],
                    1e-12
                )
            )

            transition_score = 0.0

            if prev is not None:

                transition_score = math.log(
                    max(
                        transition_matrix[
                            prev
                        ][cls],
                        1e-12
                    )
                )

            score = (
                base_score
                + weight * transition_score
            )

            if score > best_score:

                best_score = score
                best_class = cls

        predictions.append(
            best_class
        )

        previous_prediction[
            session
        ] = best_class

    return predictions


# ============================================================
# MAIN EVALUATION
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

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # BUILD DATASET
    # --------------------------------------------------------

    df = build_dataset(
        args.dataset_a
    )

    if df.empty:

        raise RuntimeError(
            "No usable executions found."
        )

    feature_columns = [
        c for c in df.columns
        if c not in [
            "process_code",
            "session_id"
        ]
    ]

    X = (
        df[feature_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0.0)
    )

    y = df["process_code"].values

    groups = df[
        "session_id"
    ].values

    classes = sorted(
        df["process_code"].unique()
    )

    print("\nPHASE 6.6 DATASET")

    print(
        f"Usable executions       : {len(df)}"
    )

    print(
        f"Process classes         : {len(classes)}"
    )

    print(
        f"Session groups          : "
        f"{len(set(groups))}"
    )

    print(
        f"Numeric features        : "
        f"{X.shape[1]}"
    )

    print(
        f"Transition weights      : "
        f"{WEIGHTS}"
    )

    # Save feature dataset
    df.to_csv(
        output_dir /
        "phase6_6_transition_sweep_features.csv",
        index=False
    )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    base_results = []
    sweep_results = {
        weight: []
        for weight in WEIGHTS
    }

    gkf = GroupKFold(
        n_splits=N_SPLITS
    )

    # --------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        gkf.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        print(
            f"\nFold {fold}/{N_SPLITS}"
        )

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        train_df = df.iloc[
            train_idx
        ]

        test_df = df.iloc[
            test_idx
        ]

        # ----------------------------------------------------
        # Base Random Forest
        # ----------------------------------------------------

        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
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

        base_pred = model.predict(
            X_test
        )

        base_accuracy = accuracy_score(
            y_test,
            base_pred
        )

        base_macro_f1 = f1_score(
            y_test,
            base_pred,
            average="macro",
            zero_division=0
        )

        base_weighted_f1 = f1_score(
            y_test,
            base_pred,
            average="weighted",
            zero_division=0
        )

        base_results.append({
            "fold": fold,
            "accuracy": base_accuracy,
            "macro_f1": base_macro_f1,
            "weighted_f1": base_weighted_f1
        })

        print(
            f"Base RF              "
            f"Acc={base_accuracy:.4f} "
            f"MacroF1={base_macro_f1:.4f}"
        )

        # ----------------------------------------------------
        # Learn transitions ONLY from training fold
        # ----------------------------------------------------

        transition_matrix = (
            learn_transition_matrix(
                train_df,
                classes
            )
        )

        session_ids = (
            test_df[
                "session_id"
            ].tolist()
        )

        # ----------------------------------------------------
        # Test every transition weight
        # ----------------------------------------------------

        for weight in WEIGHTS:

            pred = transition_decode(
                probabilities,
                classes,
                session_ids,
                transition_matrix,
                weight
            )

            accuracy = accuracy_score(
                y_test,
                pred
            )

            macro_f1 = f1_score(
                y_test,
                pred,
                average="macro",
                zero_division=0
            )

            weighted_f1 = f1_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )

            sweep_results[
                weight
            ].append({
                "fold": fold,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1
            })

            print(
                f"Weight {weight:>4.2f}      "
                f"Acc={accuracy:.4f} "
                f"MacroF1={macro_f1:.4f}"
            )

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    base_accuracy = np.mean([
        x["accuracy"]
        for x in base_results
    ])

    base_macro_f1 = np.mean([
        x["macro_f1"]
        for x in base_results
    ])

    base_weighted_f1 = np.mean([
        x["weighted_f1"]
        for x in base_results
    ])

    summary_rows = []

    for weight in WEIGHTS:

        rows = sweep_results[
            weight
        ]

        summary_rows.append({
            "transition_weight": weight,

            "accuracy_mean": np.mean([
                x["accuracy"]
                for x in rows
            ]),

            "accuracy_std": np.std([
                x["accuracy"]
                for x in rows
            ]),

            "macro_f1_mean": np.mean([
                x["macro_f1"]
                for x in rows
            ]),

            "macro_f1_std": np.std([
                x["macro_f1"]
                for x in rows
            ]),

            "weighted_f1_mean": np.mean([
                x["weighted_f1"]
                for x in rows
            ]),

            "weighted_f1_std": np.std([
                x["weighted_f1"]
                for x in rows
            ])
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        output_dir /
        "phase6_6_transition_weight_sweep.csv",
        index=False
    )

    # ========================================================
    # FIND BEST WEIGHTS
    # ========================================================

    best_accuracy_row = (
        summary_df
        .sort_values(
            "accuracy_mean",
            ascending=False
        )
        .iloc[0]
    )

    best_macro_row = (
        summary_df
        .sort_values(
            "macro_f1_mean",
            ascending=False
        )
        .iloc[0]
    )

    best_weight = float(
        best_macro_row[
            "transition_weight"
        ]
    )

    best_accuracy = float(
        best_accuracy_row[
            "accuracy_mean"
        ]
    )

    best_macro_f1 = float(
        best_macro_row[
            "macro_f1_mean"
        ]
    )

    # ========================================================
    # DELTAS FROM BASE
    # ========================================================

    summary_df[
        "accuracy_delta_vs_base"
    ] = (
        summary_df["accuracy_mean"]
        - base_accuracy
    )

    summary_df[
        "macro_f1_delta_vs_base"
    ] = (
        summary_df["macro_f1_mean"]
        - base_macro_f1
    )

    summary_df.to_csv(
        output_dir /
        "phase6_6_transition_weight_sweep.csv",
        index=False
    )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print(
        "PHASE 6.6 TRANSITION WEIGHT SWEEP RESULTS"
    )
    print("=" * 70)

    print(
        f"\nBASE MODEL"
    )

    print(
        f"Accuracy    : "
        f"{base_accuracy:.4f}"
    )

    print(
        f"Macro F1    : "
        f"{base_macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : "
        f"{base_weighted_f1:.4f}"
    )

    print(
        "\nWEIGHT RESULTS"
    )

    print(
        summary_df[
            [
                "transition_weight",
                "accuracy_mean",
                "macro_f1_mean",
                "accuracy_delta_vs_base",
                "macro_f1_delta_vs_base"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nBEST BY ACCURACY"
    )

    print(
        f"Weight      : "
        f"{best_accuracy_row['transition_weight']:.2f}"
    )

    print(
        f"Accuracy    : "
        f"{best_accuracy:.4f}"
    )

    print(
        f"Delta       : "
        f"{best_accuracy - base_accuracy:+.4f}"
    )

    print(
        "\nBEST BY MACRO F1"
    )

    print(
        f"Weight      : "
        f"{best_weight:.2f}"
    )

    print(
        f"Macro F1    : "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Delta       : "
        f"{best_macro_f1 - base_macro_f1:+.4f}"
    )

    # ========================================================
    # VERDICT
    # ========================================================

    improvement = (
        best_macro_f1
        - base_macro_f1
    )

    if best_weight == 0.0:

        verdict = (
            "NOT SUPPORTED: "
            "all tested positive transition weights "
            "failed to improve Macro F1 over the base model."
        )

    elif improvement > 0:

        verdict = (
            f"SUPPORTED: transition weight "
            f"{best_weight:.2f} improved Macro F1 "
            f"by {improvement:+.4f}."
        )

    else:

        verdict = (
            "NOT SUPPORTED: "
            "transition weighting did not improve "
            "the base model."
        )

    print(
        f"\nVERDICT: {verdict}"
    )

    # ========================================================
    # SAVE JSON SUMMARY
    # ========================================================

    result = {
        "phase": "6.6",
        "description":
            "Transition-weight sweep for identity resolution",

        "usable_executions": int(len(df)),
        "process_classes": int(len(classes)),
        "session_groups": int(len(set(groups))),

        "base_model": {
            "accuracy": float(base_accuracy),
            "macro_f1": float(base_macro_f1),
            "weighted_f1": float(base_weighted_f1)
        },

        "tested_weights": WEIGHTS,

        "best_accuracy_weight":
            float(
                best_accuracy_row[
                    "transition_weight"
                ]
            ),

        "best_accuracy":
            best_accuracy,

        "best_macro_f1_weight":
            best_weight,

        "best_macro_f1":
            best_macro_f1,

        "macro_f1_improvement":
            float(improvement),

        "verdict": verdict
    }

    with open(
        output_dir /
        "phase6_6_transition_weight_sweep_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )

    print(
        "\nOutputs saved to:"
    )

    print(
        output_dir
    )


if __name__ == "__main__":
    main()