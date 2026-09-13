from pathlib import Path
import json
import math
import random

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(r"Dataset A\dataset_a")

BOUNDARY_FILE = Path(
    r"Outputs\Day 2\day2_gt_boundary_analysis\gt_switch_boundaries.csv"
)

OUTPUT_DIR = Path(
    r"Outputs\Day 2\day2_boundary_vs_normal"
)

WINDOWS_SEC = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

# Number of negative examples.
# Keep this manageable while preserving coverage.
NEGATIVE_MULTIPLIER = 1

# A negative timestamp must be at least this far away
# from every GT boundary.
MIN_NEGATIVE_DISTANCE_SEC = 5.0

RANDOM_SEED = 42

SIGNAL_FIELDS = [
    "event_type",
    "layer",
    "source",
    "app_name",
    "process_name",
    "window_title",
    "browser_domain",
    "browser_path",
    "browser_element",
]


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(record, dict):
                continue

            records.append(record)

    return records


def safe_string(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        if value:
            return value

        return None

    return str(value)


def extract_signal(record, field):
    """
    Extract a normalized signal from the actual raw-event schema.
    """

    if field == "event_type":
        return safe_string(record.get("event_type"))

    if field == "layer":
        return safe_string(record.get("layer"))

    if field == "source":
        source = record.get("source")

        if isinstance(source, dict):
            # Use relatively stable source characteristics.
            value = (
                source.get("agent_version")
                or source.get("os")
            )
            return safe_string(value)

        return safe_string(source)

    context = record.get("context")

    if not isinstance(context, dict):
        context = {}

    active_app = context.get("active_app")

    if not isinstance(active_app, dict):
        active_app = {}

    if field == "app_name":
        return safe_string(active_app.get("app_name"))

    if field == "process_name":
        return safe_string(active_app.get("process_name"))

    if field == "window_title":
        return safe_string(active_app.get("window_title"))

    # --------------------------------------------------------
    # Browser context
    # --------------------------------------------------------

    browser_tab = context.get("active_browser_tab")

    if not isinstance(browser_tab, dict):
        browser_tab = {}

    if field == "browser_domain":
        value = (
            browser_tab.get("domain")
            or browser_tab.get("host")
        )

        return safe_string(value)

    if field == "browser_path":
        value = (
            browser_tab.get("path")
            or browser_tab.get("url_path")
        )

        return safe_string(value)

    if field == "browser_element":
        value = (
            browser_tab.get("element")
            or context.get("browser_element")
        )

        return safe_string(value)

    return None


def normalize_event(record):
    """
    Convert raw event into compact representation.
    """

    timestamp_ms = record.get("timestamp_ms")

    try:
        timestamp_ms = int(timestamp_ms)
    except (TypeError, ValueError):
        timestamp_ms = None

    if timestamp_ms is None:
        return None

    signals = {}

    for field in SIGNAL_FIELDS:
        signals[field] = extract_signal(record, field)

    return {
        "timestamp_ms": timestamp_ms,
        "signals": signals,
    }


def jaccard_similarity(a, b):
    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


def signal_set(events, field):
    values = set()

    for event in events:
        value = event["signals"].get(field)

        if value is not None:
            values.add(value)

    return values


def count_signal_changes(pre_events, post_events, field):
    pre = signal_set(pre_events, field)
    post = signal_set(post_events, field)

    return {
        "jaccard": jaccard_similarity(pre, post),
        "new_count": len(post - pre),
        "disappeared_count": len(pre - post),
        "changed": int(pre != post),
    }


def event_rate(events, window_sec):
    return len(events) / window_sec


def shannon_entropy(events, field):
    values = [
        event["signals"].get(field)
        for event in events
        if event["signals"].get(field) is not None
    ]

    if not values:
        return 0.0

    counts = {}

    for value in values:
        counts[value] = counts.get(value, 0) + 1

    total = len(values)

    entropy = 0.0

    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


# ============================================================
# LOAD GT BOUNDARIES
# ============================================================

print("=" * 70)
print("DAY 2 — PHASE 3")
print("BOUNDARY VS NORMAL ACTIVITY")
print("=" * 70)

print("\n" + "-" * 70)
print("LOADING GT BOUNDARIES")
print("-" * 70)

boundaries_df = pd.read_csv(BOUNDARY_FILE)

required_boundary_columns = {
    "session_id",
    "timestamp_ms",
    "from_process",
    "to_process",
}

missing = required_boundary_columns - set(boundaries_df.columns)

if missing:
    raise ValueError(
        f"Missing required boundary columns: {sorted(missing)}"
    )

boundaries_df["timestamp_ms"] = pd.to_numeric(
    boundaries_df["timestamp_ms"],
    errors="coerce"
)

boundaries_df = boundaries_df.dropna(
    subset=["timestamp_ms"]
).copy()

boundaries_df["timestamp_ms"] = (
    boundaries_df["timestamp_ms"].astype(np.int64)
)

print(f"GT boundaries: {len(boundaries_df):,}")


# ============================================================
# LOAD RAW EVENTS
# ============================================================

print("\n" + "-" * 70)
print("LOADING RAW EVENTS")
print("-" * 70)

session_events = {}

session_dirs = sorted(
    p for p in DATASET_ROOT.iterdir()
    if p.is_dir() and p.name.startswith("ses_")
)

print(f"Sessions found: {len(session_dirs)}")

total_raw_events = 0
total_event_files = 0

for session_dir in session_dirs:

    session_id = session_dir.name

    event_files = sorted(
        session_dir.rglob("events.jsonl")
    )

    events = []

    for event_file in event_files:

        total_event_files += 1

        records = load_jsonl(event_file)

        for record in records:

            normalized = normalize_event(record)

            if normalized is not None:
                events.append(normalized)

    events.sort(key=lambda x: x["timestamp_ms"])

    session_events[session_id] = events

    total_raw_events += len(events)

    print(
        f"{session_id}: "
        f"{len(events):,} raw events "
        f"({len(event_files)} events.jsonl files)"
    )


print("\n" + "=" * 70)
print("RAW EVENT LOADING SUMMARY")
print("=" * 70)

print(f"Sessions loaded:       {len(session_events)}")
print(f"events.jsonl files:    {total_event_files}")
print(f"Total raw events:      {total_raw_events:,}")


# ============================================================
# BUILD NEGATIVE CANDIDATES
# ============================================================

print("\n" + "-" * 70)
print("BUILDING NORMAL-ACTIVITY NEGATIVES")
print("-" * 70)

random.seed(RANDOM_SEED)

# Group GT boundaries by session.
boundary_times_by_session = {}

for _, row in boundaries_df.iterrows():

    session_id = row["session_id"]
    timestamp_ms = int(row["timestamp_ms"])

    boundary_times_by_session.setdefault(
        session_id, []
    ).append(timestamp_ms)

for session_id in boundary_times_by_session:
    boundary_times_by_session[session_id].sort()


def far_from_boundary(timestamp_ms, boundary_times, min_distance_ms):
    """
    Check whether timestamp is sufficiently far from
    every known GT boundary.
    """

    if not boundary_times:
        return True

    # Binary-search would be faster, but the boundary count
    # per session is small enough for this analysis.
    return all(
        abs(timestamp_ms - b) >= min_distance_ms
        for b in boundary_times
    )


negative_candidates = []

for session_id, events in session_events.items():

    boundary_times = boundary_times_by_session.get(
        session_id, []
    )

    min_distance_ms = int(
        MIN_NEGATIVE_DISTANCE_SEC * 1000
    )

    for event in events:

        timestamp_ms = event["timestamp_ms"]

        if far_from_boundary(
            timestamp_ms,
            boundary_times,
            min_distance_ms,
        ):
            negative_candidates.append(
                {
                    "session_id": session_id,
                    "timestamp_ms": timestamp_ms,
                }
            )


print(
    f"Potential normal timestamps: "
    f"{len(negative_candidates):,}"
)


# ============================================================
# SAMPLE NEGATIVES
# ============================================================

desired_negatives = (
    len(boundaries_df) * NEGATIVE_MULTIPLIER
)

if len(negative_candidates) < desired_negatives:
    print(
        "WARNING: fewer negative candidates than requested."
    )

    sampled_negatives = negative_candidates
else:
    sampled_negatives = random.sample(
        negative_candidates,
        desired_negatives,
    )

print(
    f"Selected normal examples: "
    f"{len(sampled_negatives):,}"
)


# ============================================================
# CREATE EXAMPLE TABLE
# ============================================================

positive_examples = []

for _, row in boundaries_df.iterrows():

    positive_examples.append(
        {
            "label": 1,
            "example_type": "boundary",
            "session_id": row["session_id"],
            "timestamp_ms": int(row["timestamp_ms"]),
            "from_process": row["from_process"],
            "to_process": row["to_process"],
        }
    )


negative_examples = []

for item in sampled_negatives:

    negative_examples.append(
        {
            "label": 0,
            "example_type": "normal",
            "session_id": item["session_id"],
            "timestamp_ms": int(item["timestamp_ms"]),
            "from_process": None,
            "to_process": None,
        }
    )


examples = positive_examples + negative_examples

print(
    f"Total examples: {len(examples):,}"
)


# ============================================================
# ANALYZE WINDOWS
# ============================================================

print("\n" + "-" * 70)
print("EXTRACTING BOUNDARY VS NORMAL FEATURES")
print("-" * 70)

rows = []

for example_index, example in enumerate(examples):

    session_id = example["session_id"]
    center_ms = example["timestamp_ms"]
    label = example["label"]

    events = session_events.get(session_id, [])

    if not events:
        continue

    for window_sec in WINDOWS_SEC:

        window_ms = int(window_sec * 1000)

        pre_start = center_ms - window_ms
        post_end = center_ms + window_ms

        pre_events = [
            e for e in events
            if pre_start <= e["timestamp_ms"] < center_ms
        ]

        post_events = [
            e for e in events
            if center_ms <= e["timestamp_ms"] <= post_end
        ]

        row = {
            "example_index": example_index,
            "label": label,
            "example_type": example["example_type"],
            "session_id": session_id,
            "timestamp_ms": center_ms,
            "window_sec": window_sec,

            "pre_event_count": len(pre_events),
            "post_event_count": len(post_events),

            "pre_event_rate": event_rate(
                pre_events, window_sec
            ),

            "post_event_rate": event_rate(
                post_events, window_sec
            ),

            "event_rate_ratio": (
                (len(post_events) / window_sec + 1e-9)
                /
                (len(pre_events) / window_sec + 1e-9)
            ),

            "pre_event_type_entropy":
                shannon_entropy(
                    pre_events,
                    "event_type"
                ),

            "post_event_type_entropy":
                shannon_entropy(
                    post_events,
                    "event_type"
                ),
        }

        # ----------------------------------------------------
        # Signal-level features
        # ----------------------------------------------------

        for field in SIGNAL_FIELDS:

            stats = count_signal_changes(
                pre_events,
                post_events,
                field
            )

            prefix = field

            row[f"{prefix}_jaccard"] = stats["jaccard"]
            row[f"{prefix}_new_count"] = stats["new_count"]
            row[f"{prefix}_disappeared_count"] = (
                stats["disappeared_count"]
            )
            row[f"{prefix}_changed"] = stats["changed"]

        rows.append(row)


features_df = pd.DataFrame(rows)

print(
    f"Feature observations generated: "
    f"{len(features_df):,}"
)


# ============================================================
# SUMMARY BY WINDOW
# ============================================================

print("\n" + "-" * 70)
print("BOUNDARY VS NORMAL SUMMARY")
print("-" * 70)

summary_rows = []

for window_sec in WINDOWS_SEC:

    subset = features_df[
        features_df["window_sec"] == window_sec
    ]

    boundary = subset[
        subset["label"] == 1
    ]

    normal = subset[
        subset["label"] == 0
    ]

    summary = {
        "window_sec": window_sec,
        "boundary_examples": len(boundary),
        "normal_examples": len(normal),
    }

    feature_columns = [
        c for c in features_df.columns
        if c not in {
            "example_index",
            "label",
            "example_type",
            "session_id",
            "timestamp_ms",
            "window_sec",
        }
    ]

    for feature in feature_columns:

        b = pd.to_numeric(
            boundary[feature],
            errors="coerce"
        ).dropna()

        n = pd.to_numeric(
            normal[feature],
            errors="coerce"
        ).dropna()

        if len(b) == 0 or len(n) == 0:
            continue

        b_mean = b.mean()
        n_mean = n.mean()

        b_std = b.std(ddof=1)
        n_std = n.std(ddof=1)

        pooled_std = math.sqrt(
            (
                (len(b) - 1) * (b_std ** 2)
                +
                (len(n) - 1) * (n_std ** 2)
            )
            /
            max(len(b) + len(n) - 2, 1)
        )

        if pooled_std > 0:
            smd = (b_mean - n_mean) / pooled_std
        else:
            smd = 0.0

        summary[f"{feature}_boundary_mean"] = b_mean
        summary[f"{feature}_normal_mean"] = n_mean
        summary[f"{feature}_smd"] = smd

    summary_rows.append(summary)


summary_df = pd.DataFrame(summary_rows)


# ============================================================
# PRINT IMPORTANT SIGNALS
# ============================================================

for window_sec in WINDOWS_SEC:

    row = summary_df[
        summary_df["window_sec"] == window_sec
    ]

    if row.empty:
        continue

    row = row.iloc[0]

    print("\n" + "=" * 70)
    print(f"WINDOW: ±{window_sec} seconds")
    print("=" * 70)

    important_features = [
        "pre_event_rate",
        "post_event_rate",
        "event_rate_ratio",

        "event_type_jaccard",
        "event_type_changed",

        "app_name_jaccard",
        "app_name_changed",

        "process_name_jaccard",
        "process_name_changed",

        "window_title_jaccard",
        "window_title_changed",

        "browser_domain_jaccard",
        "browser_domain_changed",
    ]

    for feature in important_features:

        smd_key = f"{feature}_smd"

        if smd_key not in row:
            continue

        boundary_mean = row.get(
            f"{feature}_boundary_mean",
            np.nan
        )

        normal_mean = row.get(
            f"{feature}_normal_mean",
            np.nan
        )

        smd = row.get(
            smd_key,
            np.nan
        )

        print(
            f"{feature:35s} "
            f"boundary={boundary_mean:8.4f} "
            f"normal={normal_mean:8.4f} "
            f"SMD={smd:8.4f}"
        )


# ============================================================
# LOGISTIC REGRESSION BASELINE
# ============================================================

logistic_results = []

if SKLEARN_AVAILABLE:

    print("\n" + "-" * 70)
    print("SESSION-GROUPED LOGISTIC REGRESSION")
    print("-" * 70)

    exclude_columns = {
        "example_index",
        "label",
        "example_type",
        "session_id",
        "timestamp_ms",
        "window_sec",
    }

    for window_sec in WINDOWS_SEC:

        subset = features_df[
            features_df["window_sec"] == window_sec
        ].copy()

        X = subset.drop(
            columns=list(exclude_columns),
            errors="ignore"
        )

        y = subset["label"]
        groups = subset["session_id"]

        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        X = X.fillna(0.0)

        # Need enough unique sessions.
        unique_groups = groups.nunique()

        if unique_groups < 3:
            continue

        n_splits = min(
            5,
            unique_groups
        )

        cv = GroupKFold(
            n_splits=n_splits
        )

        model = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                ),
                (
                    "logistic",
                    LogisticRegression(
                        max_iter=2000
                    )
                ),
            ]
        )

        try:
            scores = cross_val_score(
                model,
                X,
                y,
                groups=groups,
                cv=cv,
                scoring="roc_auc",
            )

            mean_auc = scores.mean()
            std_auc = scores.std()

        except Exception as exc:

            print(
                f"Window {window_sec}s failed: {exc}"
            )

            mean_auc = np.nan
            std_auc = np.nan

        logistic_results.append(
            {
                "window_sec": window_sec,
                "mean_grouped_roc_auc": mean_auc,
                "std_grouped_roc_auc": std_auc,
                "sessions": unique_groups,
            }
        )

        print(
            f"±{window_sec:4.1f}s : "
            f"ROC-AUC = "
            f"{mean_auc:.4f} ± {std_auc:.4f}"
        )

else:

    print(
        "scikit-learn not available; "
        "skipping logistic regression."
    )


logistic_df = pd.DataFrame(
    logistic_results
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

features_path = (
    OUTPUT_DIR /
    "boundary_vs_normal_features.csv"
)

summary_path = (
    OUTPUT_DIR /
    "boundary_vs_normal_summary.csv"
)

logistic_path = (
    OUTPUT_DIR /
    "boundary_vs_normal_logistic.csv"
)

json_path = (
    OUTPUT_DIR /
    "day2_boundary_vs_normal_summary.json"
)

features_df.to_csv(
    features_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

logistic_df.to_csv(
    logistic_path,
    index=False
)


# ============================================================
# FINAL SUMMARY JSON
# ============================================================

best_window_by_smd = {}

for feature in [
    "post_event_rate",
    "event_rate_ratio",
    "app_name_changed",
    "process_name_changed",
    "window_title_changed",
    "event_type_changed",
]:

    smd_column = f"{feature}_smd"

    if smd_column not in summary_df.columns:
        continue

    temp = summary_df[
        ["window_sec", smd_column]
    ].copy()

    temp["abs_smd"] = temp[smd_column].abs()

    temp = temp.dropna(
        subset=["abs_smd"]
    )

    if temp.empty:
        continue

    best = temp.loc[
        temp["abs_smd"].idxmax()
    ]

    best_window_by_smd[feature] = {
        "window_sec": float(
            best["window_sec"]
        ),
        "smd": float(
            best[smd_column]
        ),
    }


summary_json = {
    "gt_boundaries": int(
        len(boundaries_df)
    ),

    "normal_examples": int(
        len(negative_examples)
    ),

    "sessions": int(
        len(session_events)
    ),

    "raw_events": int(
        total_raw_events
    ),

    "negative_distance_sec":
        MIN_NEGATIVE_DISTANCE_SEC,

    "windows_sec":
        WINDOWS_SEC,

    "best_window_by_absolute_smd":
        best_window_by_smd,

    "logistic_regression":
        logistic_results,
}


with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary_json,
        f,
        indent=2
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3 COMPLETE")
print("=" * 70)

print(f"\n{features_path}")
print(f"{summary_path}")
print(f"{logistic_path}")
print(f"{json_path}")

print("\nNext: inspect BOUNDARY VS NORMAL SUMMARY.")