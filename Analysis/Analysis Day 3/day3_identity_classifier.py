"""
PHASE 6.4 - PROCESS IDENTITY CLASSIFICATION

Purpose
-------
Test whether logical business-process identity can be inferred
from observed behavioral patterns in the raw operation log.

Dataset A:
    - 2009 GT executions in gt_manifest.json
    - 1752 executions have both start_ts and end_ts
    - Only those 1752 complete executions are used for temporal
      feature extraction.
    - No timestamps are fabricated.

Models
------
1. APP-ONLY BASELINE
   Uses the applications observed during an execution.

2. EVENT-BEHAVIOR MODEL
   Uses event-type, layer, interaction, application and temporal
   behavior features.

Validation
----------
GroupKFold by session/manifest to prevent executions from the
same recording session appearing in both train and validation.
"""

import argparse
import bisect
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-a",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    return parser.parse_args()


# ============================================================
# TIMESTAMP
# ============================================================

def parse_ts(value):

    if not value:
        return None

    try:

        return pd.to_datetime(
            value,
            utc=True
        )

    except Exception:

        return None


# ============================================================
# LOAD COMPLETE GT EXECUTIONS
# ============================================================

def load_executions(dataset_a):

    executions = []

    manifest_files = sorted(
        Path(dataset_a).rglob(
            "gt_manifest.json"
        )
    )

    for manifest_path in manifest_files:

        with open(
            manifest_path,
            "r",
            encoding="utf-8"
        ) as f:

            manifest = json.load(f)

        # Session directory is the grouping unit.
        session_id = manifest_path.parent.name

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

            domain = process.get(
                "domain"
            )

            for execution in process.get(
                "executions",
                []
            ):

                start_raw = execution.get(
                    "start_ts"
                )

                end_raw = execution.get(
                    "end_ts"
                )

                if not start_raw or not end_raw:
                    continue

                start = parse_ts(
                    start_raw
                )

                end = parse_ts(
                    end_raw
                )

                if start is None or end is None:
                    continue

                if start >= end:
                    continue

                executions.append({

                    "process_code":
                        process_code,

                    "family_name":
                        family_name,

                    "domain":
                        domain,

                    "variant":
                        execution.get(
                            "variant"
                        ),

                    "case_id":
                        execution.get(
                            "case_id"
                        ),

                    "start":
                        start,

                    "end":
                        end,

                    "session_id":
                        session_id,

                    "manifest":
                        str(manifest_path)

                })

    return executions


# ============================================================
# LOAD RAW EVENTS
# ============================================================

def load_events(dataset_a):

    events = []

    event_files = sorted(
        Path(dataset_a).rglob(
            "events.jsonl"
        )
    )

    print(
        f"events.jsonl files found : "
        f"{len(event_files)}"
    )

    for event_file in event_files:

        with open(
            event_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:

                    event = json.loads(
                        line
                    )

                except json.JSONDecodeError:

                    continue

                timestamp = event.get(
                    "timestamp_iso"
                )

                if not timestamp:
                    continue

                parsed = parse_ts(
                    timestamp
                )

                if parsed is None:
                    continue

                event["_parsed_ts"] = parsed

                events.append(
                    event
                )

    events.sort(
        key=lambda x:
        x["_parsed_ts"]
    )

    print(
        f"Raw events loaded    : "
        f"{len(events)}"
    )

    return events


# ============================================================
# EVENT ACCESSORS
# ============================================================

def event_type(event):

    return str(
        event.get(
            "event_type",
            "UNKNOWN"
        )
    )


def event_layer(event):

    return str(
        event.get(
            "layer",
            "UNKNOWN"
        )
    )


def active_app(event):

    context = event.get(
        "context",
        {}
    )

    if not isinstance(
        context,
        dict
    ):
        return "UNKNOWN"

    app_info = context.get(
        "active_app",
        {}
    )

    if isinstance(
        app_info,
        dict
    ):

        value = (
            app_info.get("app_name")
            or app_info.get("process_name")
        )

        if value:
            return str(value)

    if isinstance(
        app_info,
        str
    ):

        return app_info

    return "UNKNOWN"


# ============================================================
# ENTROPY
# ============================================================

def entropy(values):

    if not values:
        return 0.0

    counts = Counter(
        values
    )

    total = len(values)

    result = 0.0

    for count in counts.values():

        p = count / total

        result -= (
            p * math.log2(p)
        )

    return result


# ============================================================
# EVENT INDEX
# ============================================================

def build_event_index(events):

    timestamps = [
        event["_parsed_ts"]
        for event in events
    ]

    return timestamps


# ============================================================
# GET EVENTS IN WINDOW
# ============================================================

def get_window_events(
    events,
    timestamps,
    start,
    end
):

    left = bisect.bisect_left(
        timestamps,
        start
    )

    right = bisect.bisect_right(
        timestamps,
        end
    )

    return events[
        left:right
    ]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    execution,
    window
):

    duration = (
        execution["end"]
        -
        execution["start"]
    ).total_seconds()

    duration = max(
        duration,
        0.001
    )

    event_types = [
        event_type(e)
        for e in window
    ]

    layers = [
        event_layer(e)
        for e in window
    ]

    apps = [
        active_app(e)
        for e in window
    ]

    type_counts = Counter(
        event_types
    )

    layer_counts = Counter(
        layers
    )

    app_counts = Counter(
        apps
    )

    # --------------------------------------------------------
    # Individual event categories
    # --------------------------------------------------------

    def count_type(
        name
    ):

        return type_counts.get(
            name,
            0
        )

    keystrokes = count_type(
        "keystroke"
    )

    shortcuts = count_type(
        "shortcut"
    )

    clicks = (
        count_type(
            "mouse_click"
        )
        +
        count_type(
            "mouse_double_click"
        )
    )

    scrolls = count_type(
        "mouse_scroll"
    )

    drag_drop = count_type(
        "mouse_drag_drop"
    )

    clipboard = count_type(
        "clipboard_change"
    )

    screenshots = count_type(
        "screenshot_smart"
    )

    app_switches = count_type(
        "app_switch"
    )

    window_changes = (
        count_type(
            "window_title_change"
        )
        +
        count_type(
            "window_state_change"
        )
    )

    dialogs = (
        count_type(
            "dialog_opened"
        )
        +
        count_type(
            "dialog_closed"
        )
    )

    browser_clicks = count_type(
        "browser_click"
    )

    browser_inputs = count_type(
        "browser_form_input"
    )

    browser_navigation = count_type(
        "browser_navigation"
    )

    browser_tabs = count_type(
        "browser_tab_event"
    )

    browser_errors = count_type(
        "browser_error"
    )

    browser_alerts = count_type(
        "browser_alert"
    )

    # --------------------------------------------------------
    # Rate helper
    # --------------------------------------------------------

    def rate(value):

        return value / duration

    # --------------------------------------------------------
    # Core numeric features
    # --------------------------------------------------------

    features = {

        "duration_log":
            math.log1p(duration),

        "event_count_log":
            math.log1p(
                len(window)
            ),

        "event_rate":
            rate(len(window)),

        "unique_event_types":
            len(type_counts),

        "event_type_entropy":
            entropy(event_types),

        "unique_layers":
            len(layer_counts),

        "layer_entropy":
            entropy(layers),

        "unique_apps":
            len(app_counts),

        "app_entropy":
            entropy(apps),

        "keystroke_rate":
            rate(keystrokes),

        "shortcut_rate":
            rate(shortcuts),

        "click_rate":
            rate(clicks),

        "scroll_rate":
            rate(scrolls),

        "drag_drop_rate":
            rate(drag_drop),

        "clipboard_rate":
            rate(clipboard),

        "screenshot_rate":
            rate(screenshots),

        "app_switch_rate":
            rate(app_switches),

        "window_change_rate":
            rate(window_changes),

        "dialog_rate":
            rate(dialogs),

        "browser_click_rate":
            rate(browser_clicks),

        "browser_input_rate":
            rate(browser_inputs),

        "browser_navigation_rate":
            rate(browser_navigation),

        "browser_tab_rate":
            rate(browser_tabs),

        "browser_error_rate":
            rate(browser_errors),

        "browser_alert_rate":
            rate(browser_alerts),
    }

    # --------------------------------------------------------
    # Relative event-type composition
    # --------------------------------------------------------

    total = max(
        len(window),
        1
    )

    for name, count in type_counts.items():

        safe_name = (
            name
            .lower()
            .replace(
                " ",
                "_"
            )
            .replace(
                "/",
                "_"
            )
        )

        features[
            "type_ratio__" +
            safe_name
        ] = count / total

    # --------------------------------------------------------
    # Layer composition
    # --------------------------------------------------------

    for name, count in layer_counts.items():

        safe_name = (
            name
            .lower()
            .replace(
                " ",
                "_"
            )
        )

        features[
            "layer_ratio__" +
            safe_name
        ] = count / total

    # --------------------------------------------------------
    # Application composition
    # --------------------------------------------------------

    for name, count in app_counts.items():

        safe_name = (
            name
            .lower()
            .replace(
                " ",
                "_"
            )
            .replace(
                "/",
                "_"
            )
        )

        features[
            "app_ratio__" +
            safe_name
        ] = count / total

    return features


# ============================================================
# BUILD FEATURE DATASET
# ============================================================

def build_dataset(
    executions,
    events
):

    timestamps = build_event_index(
        events
    )

    rows = []

    total = len(
        executions
    )

    for i, execution in enumerate(
        executions,
        start=1
    ):

        window = get_window_events(
            events,
            timestamps,
            execution["start"],
            execution["end"]
        )

        features = extract_features(
            execution,
            window
        )

        features[
            "process_code"
        ] = execution[
            "process_code"
        ]

        features[
            "family_name"
        ] = execution[
            "family_name"
        ]

        features[
            "variant"
        ] = execution[
            "variant"
        ]

        features[
            "case_id"
        ] = execution[
            "case_id"
        ]

        features[
            "session_id"
        ] = execution[
            "session_id"
        ]

        rows.append(
            features
        )

        if (
            i % 250 == 0
            or i == total
        ):

            print(
                f"Feature rows processed : "
                f"{i}/{total}"
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# APP-ONLY DATASET
# ============================================================

def build_app_features(
    executions,
    events
):

    timestamps = build_event_index(
        events
    )

    rows = []

    for execution in executions:

        window = get_window_events(
            events,
            timestamps,
            execution["start"],
            execution["end"]
        )

        apps = Counter(
            active_app(e)
            for e in window
        )

        total = max(
            len(window),
            1
        )

        row = {}

        for app, count in apps.items():

            safe_name = (
                app
                .lower()
                .replace(
                    " ",
                    "_"
                )
                .replace(
                    "/",
                    "_"
                )
            )

            row[
                "app__" +
                safe_name
            ] = count / total

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    ).fillna(0)


# ============================================================
# GROUPED CROSS VALIDATION
# ============================================================

def grouped_predictions(
    model,
    X,
    y,
    groups
):

    unique_groups = np.unique(
        groups
    )

    n_splits = min(
        5,
        len(unique_groups)
    )

    cv = GroupKFold(
        n_splits=n_splits
    )

    predictions = np.empty(
        len(y),
        dtype=object
    )

    for train_idx, test_idx in cv.split(
        X,
        y,
        groups
    ):

        model.fit(
            X.iloc[train_idx]
            if isinstance(X, pd.DataFrame)
            else X[train_idx],

            y[train_idx]
        )

        predictions[test_idx] = (
            model.predict(
                X.iloc[test_idx]
                if isinstance(X, pd.DataFrame)
                else X[test_idx]
            )
        )

    return predictions


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    name,
    model,
    X,
    y,
    groups,
    labels,
    output_dir
):

    print()
    print(
        f"Evaluating model: "
        f"{name}"
    )

    predictions = grouped_predictions(
        model,
        X,
        y,
        groups
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(
        f"Accuracy    : "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro F1    : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : "
        f"{weighted_f1:.4f}"
    )

    report = classification_report(
        y,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    with open(
        output_dir /
        f"phase6_4_{name}_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "accuracy":
                    float(accuracy),

                "macro_f1":
                    float(macro_f1),

                "weighted_f1":
                    float(weighted_f1),

                "classification_report":
                    report
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    return predictions


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

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

    print("=" * 60)
    print(
        "PHASE 6.4 - PROCESS IDENTITY CLASSIFICATION"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    executions = load_executions(
        dataset_a
    )

    print()
    print(
        f"Complete GT executions : "
        f"{len(executions)}"
    )

    events = load_events(
        dataset_a
    )

    # --------------------------------------------------------
    # Build behavioral dataset
    # --------------------------------------------------------

    df = build_dataset(
        executions,
        events
    )

    print()
    print(
        f"Feature dataset shape : "
        f"{df.shape}"
    )

    # --------------------------------------------------------
    # Distribution
    # --------------------------------------------------------

    print()
    print(
        "PROCESS DISTRIBUTION"
    )
    print("-" * 40)

    distribution = (
        df["process_code"]
        .value_counts()
        .sort_index()
    )

    for process, count in distribution.items():

        print(
            f"{process:<10} : {count}"
        )

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------

    df.to_csv(
        output_dir /
        "phase6_4_identity_features.csv",
        index=False
    )

    # --------------------------------------------------------
    # Numeric behavioral features
    # --------------------------------------------------------

    excluded = {
        "process_code",
        "family_name",
        "variant",
        "case_id",
        "session_id"
    }

    feature_columns = [

        column

        for column in df.columns

        if column not in excluded

        and pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    X = (
        df[
            feature_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    y = df[
        "process_code"
    ].values

    groups = df[
        "session_id"
    ].values

    labels = sorted(
        df[
            "process_code"
        ].unique()
    )

    print()
    print(
        f"Behavioral features : "
        f"{len(feature_columns)}"
    )

    print(
        f"Process classes      : "
        f"{len(labels)}"
    )

    print(
        f"Sessions/groups      : "
        f"{len(np.unique(groups))}"
    )

    # --------------------------------------------------------
    # Behavioral logistic model
    # --------------------------------------------------------

    behavioral_model = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=42
            )
        )

    ])

    behavioral_predictions = evaluate(
        "behavioral_logistic",
        behavioral_model,
        X,
        y,
        groups,
        labels,
        output_dir
    )

    # --------------------------------------------------------
    # Random forest model
    # --------------------------------------------------------

    rf_model = RandomForestClassifier(

        n_estimators=400,

        random_state=42,

        class_weight="balanced",

        n_jobs=-1,

        max_features="sqrt"
    )

    rf_predictions = evaluate(
        "behavioral_random_forest",
        rf_model,
        X,
        y,
        groups,
        labels,
        output_dir
    )

    # --------------------------------------------------------
    # Confusion matrix for RF
    # --------------------------------------------------------

    rf_matrix = confusion_matrix(
        y,
        rf_predictions,
        labels=labels
    )

    pd.DataFrame(
        rf_matrix,
        index=labels,
        columns=labels
    ).to_csv(
        output_dir /
        "phase6_4_random_forest_confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # RF feature importance
    # --------------------------------------------------------

    rf_model.fit(
        X,
        y
    )

    importance = pd.DataFrame({

        "feature":
            feature_columns,

        "importance":
            rf_model.feature_importances_

    }).sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        output_dir /
        "phase6_4_feature_importance.csv",
        index=False
    )

    print()
    print(
        "TOP FEATURE IMPORTANCE"
    )
    print("-" * 50)

    for _, row in importance.head(
        20
    ).iterrows():

        print(
            f"{row['feature']:<40} "
            f"{row['importance']:.5f}"
        )

    # --------------------------------------------------------
    # Top confusion pairs
    # --------------------------------------------------------

    confusion_pairs = []

    for i, actual in enumerate(
        labels
    ):

        for j, predicted in enumerate(
            labels
        ):

            if i == j:
                continue

            count = int(
                rf_matrix[i, j]
            )

            if count > 0:

                confusion_pairs.append({

                    "actual":
                        actual,

                    "predicted":
                        predicted,

                    "count":
                        count

                })

    confusion_pairs.sort(
        key=lambda x:
        x["count"],
        reverse=True
    )

    print()
    print(
        "TOP CONFUSED PROCESS PAIRS"
    )
    print("-" * 50)

    for item in confusion_pairs[:20]:

        print(
            f"{item['actual']} -> "
            f"{item['predicted']} : "
            f"{item['count']}"
        )

    with open(
        output_dir /
        "phase6_4_confusion_pairs.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            confusion_pairs,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # APP-ONLY BASELINE
    # --------------------------------------------------------

    print()
    print(
        "Building app-only baseline..."
    )

    X_app = build_app_features(
        executions,
        events
    )

    X_app = (
        X_app
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .fillna(0)
    )

    print(
        f"App-only features    : "
        f"{X_app.shape[1]}"
    )

    app_model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    )

    app_predictions = evaluate(
        "app_only_baseline",
        app_model,
        X_app,
        y,
        groups,
        labels,
        output_dir
    )

    # --------------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------------

    behavioral_accuracy = accuracy_score(
        y,
        behavioral_predictions
    )

    behavioral_f1 = f1_score(
        y,
        behavioral_predictions,
        average="macro",
        zero_division=0
    )

    rf_accuracy = accuracy_score(
        y,
        rf_predictions
    )

    rf_f1 = f1_score(
        y,
        rf_predictions,
        average="macro",
        zero_division=0
    )

    app_accuracy = accuracy_score(
        y,
        app_predictions
    )

    app_f1 = f1_score(
        y,
        app_predictions,
        average="macro",
        zero_division=0
    )

    comparison = {

        "complete_gt_executions":
            len(executions),

        "process_classes":
            len(labels),

        "session_groups":
            len(np.unique(groups)),

        "behavioral_logistic": {

            "accuracy":
                float(
                    behavioral_accuracy
                ),

            "macro_f1":
                float(
                    behavioral_f1
                )
        },

        "behavioral_random_forest": {

            "accuracy":
                float(
                    rf_accuracy
                ),

            "macro_f1":
                float(
                    rf_f1
                )
        },

        "app_only_baseline": {

            "accuracy":
                float(
                    app_accuracy
                ),

            "macro_f1":
                float(
                    app_f1
                )
        }
    }

    with open(
        output_dir /
        "phase6_4_model_comparison.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            comparison,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    summary = {

        "phase":
            "6.4",

        "manifest_executions":
            2009,

        "complete_executions_used":
            len(executions),

        "incomplete_executions_excluded":
            2009 - len(executions),

        "process_families":
            len(labels),

        "session_groups":
            len(np.unique(groups)),

        "behavioral_feature_count":
            len(feature_columns),

        "model_comparison":
            comparison,

        "top_features":
            importance.head(
                20
            ).to_dict(
                orient="records"
            ),

        "top_confusion_pairs":
            confusion_pairs[:20]
    }

    with open(
        output_dir /
        "phase6_4_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "PHASE 6.4 COMPLETE"
    )
    print("=" * 60)

    print(
        f"Executions used      : "
        f"{len(executions)}"
    )

    print(
        f"Process families     : "
        f"{len(labels)}"
    )

    print(
        f"Session groups       : "
        f"{len(np.unique(groups))}"
    )

    print()
    print(
        "BEHAVIORAL LOGISTIC"
    )

    print(
        f"Accuracy             : "
        f"{behavioral_accuracy:.4f}"
    )

    print(
        f"Macro F1             : "
        f"{behavioral_f1:.4f}"
    )

    print()
    print(
        "BEHAVIORAL RANDOM FOREST"
    )

    print(
        f"Accuracy             : "
        f"{rf_accuracy:.4f}"
    )

    print(
        f"Macro F1             : "
        f"{rf_f1:.4f}"
    )

    print()
    print(
        "APP-ONLY BASELINE"
    )

    print(
        f"Accuracy             : "
        f"{app_accuracy:.4f}"
    )

    print(
        f"Macro F1             : "
        f"{app_f1:.4f}"
    )

    print()
    print(
        f"Outputs written to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()