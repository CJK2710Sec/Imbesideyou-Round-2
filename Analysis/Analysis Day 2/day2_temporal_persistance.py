from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# DAY 2 — PHASE 5D v2
# TEMPORAL PERSISTENCE + BOUNDARY REFINEMENT
# ============================================================


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(r"Dataset A\dataset_a")

PHASE3_FEATURES = Path(
    r"Outputs\Day 2\day2_boundary_vs_normal\boundary_vs_normal_features.csv"
)

GT_BOUNDARIES = Path(
    r"Outputs\Day 2\day2_gt_boundary_analysis\gt_switch_boundaries.csv"
)

HARD_NEGATIVE_FEATURES = Path(
    r"Outputs\Day 2\day2_boundary_hard_negative\hard_negative_features.csv"
)

OUTPUT_DIR = Path(
    r"Outputs\Day 2\day2_temporal_persistence"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEMPORAL CONFIGURATION
# ============================================================

# Detector feature windows.
# These are the same windows used in previous phases.

WINDOWS = [
    0.5,
    1.0,
    2.0,
    3.0,
]


# Temporal profile around each center.
#
# Example for a GT boundary at t:
#
#       -2s -1.5s -1s -0.5s 0 +0.5s +1s +1.5s +2s
#
# We score each center using the fixed detector.

PROFILE_OFFSETS = [
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
]


# Thresholds used for persistence measurements.

THRESHOLDS = [
    0.50,
    0.70,
    0.90,
    0.95,
]


# ============================================================
# FIXED D_LEAN_FULL FEATURE SET
# ============================================================

FEATURES = [
    "pre_event_rate",
    "post_event_rate",

    "event_type_jaccard",
    "event_type_changed",
    "event_type_new_count",
    "event_type_disappeared_count",

    "layer_jaccard",
    "layer_changed",
    "layer_new_count",
    "layer_disappeared_count",

    "pre_event_type_entropy",
    "post_event_type_entropy",

    "window_title_jaccard",
    "window_title_changed",
    "window_title_new_count",

    "app_name_jaccard",
    "app_name_changed",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def entropy(values):
    """
    Shannon entropy of categorical values.
    """

    if not values:
        return 0.0

    counts = {}

    for value in values:
        value = str(value)
        counts[value] = counts.get(value, 0) + 1

    total = sum(counts.values())

    if total == 0:
        return 0.0

    result = 0.0

    for count in counts.values():

        p = count / total

        if p > 0:
            result -= p * math.log2(p)

    return result


def jaccard(a, b):
    """
    Jaccard similarity between two sets.
    """

    a = set(a)
    b = set(b)

    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


def changed(a, b):
    """
    Whether two categorical sets differ.
    """

    return int(set(a) != set(b))


def new_count(a, b):
    """
    Values appearing in post but not pre.
    """

    return len(set(b) - set(a))


def disappeared_count(a, b):
    """
    Values appearing in pre but not post.
    """

    return len(set(a) - set(b))


def get_active_app(event):
    """
    Extract active_app fields from raw event context.
    """

    context = event.get("context", {})

    if not isinstance(context, dict):
        return {
            "app_name": "",
            "process_name": "",
            "window_title": "",
        }

    active_app = context.get("active_app", {})

    if not isinstance(active_app, dict):
        return {
            "app_name": "",
            "process_name": "",
            "window_title": "",
        }

    return {
        "app_name": str(
            active_app.get("app_name", "")
        ),
        "process_name": str(
            active_app.get("process_name", "")
        ),
        "window_title": str(
            active_app.get("window_title", "")
        ),
    }


def event_layer(event):
    value = event.get("layer", "")

    if value is None:
        return ""

    return str(value)


def event_type(event):
    value = event.get("event_type", "")

    if value is None:
        return ""

    return str(value)


# ============================================================
# RAW EVENT LOADING
# ============================================================

def load_session_events(session_dir):
    """
    Load all events.jsonl files belonging to one session.
    """

    events = []

    event_files = sorted(
        session_dir.rglob("events.jsonl")
    )

    for path in event_files:

        try:

            with path.open(
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    if not isinstance(obj, dict):
                        continue

                    timestamp = safe_float(
                        obj.get("timestamp_ms")
                    )

                    if np.isnan(timestamp):
                        continue

                    obj["_timestamp_ms"] = timestamp

                    events.append(obj)

        except Exception as exc:

            print(
                f"WARNING: could not read {path}: {exc}"
            )

    events.sort(
        key=lambda x: x["_timestamp_ms"]
    )

    return events


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features_at_center(
    events,
    center_ms,
    window_sec
):
    """
    Reconstruct the same D_lean_full-style features
    around a temporal center.

    pre  = [center-window, center)
    post = [center, center+window]
    """

    half_ms = window_sec * 1000.0

    start_pre = center_ms - half_ms
    end_pre = center_ms

    start_post = center_ms
    end_post = center_ms + half_ms

    pre_events = []
    post_events = []

    for event in events:

        ts = event["_timestamp_ms"]

        if start_pre <= ts < end_pre:

            pre_events.append(event)

        elif start_post <= ts <= end_post:

            post_events.append(event)

    # --------------------------------------------------------
    # Basic event rates
    # --------------------------------------------------------

    pre_rate = len(pre_events) / window_sec

    post_rate = len(post_events) / window_sec

    # --------------------------------------------------------
    # Event types
    # --------------------------------------------------------

    pre_event_types = [
        event_type(e)
        for e in pre_events
        if event_type(e)
    ]

    post_event_types = [
        event_type(e)
        for e in post_events
        if event_type(e)
    ]

    event_type_jaccard = jaccard(
        pre_event_types,
        post_event_types
    )

    event_type_changed = changed(
        pre_event_types,
        post_event_types
    )

    event_type_new = new_count(
        pre_event_types,
        post_event_types
    )

    event_type_disappeared = disappeared_count(
        pre_event_types,
        post_event_types
    )

    # --------------------------------------------------------
    # Layers
    # --------------------------------------------------------

    pre_layers = [
        event_layer(e)
        for e in pre_events
        if event_layer(e)
    ]

    post_layers = [
        event_layer(e)
        for e in post_events
        if event_layer(e)
    ]

    layer_jaccard = jaccard(
        pre_layers,
        post_layers
    )

    layer_changed = changed(
        pre_layers,
        post_layers
    )

    layer_new = new_count(
        pre_layers,
        post_layers
    )

    layer_disappeared = disappeared_count(
        pre_layers,
        post_layers
    )

    # --------------------------------------------------------
    # Event-type entropy
    # --------------------------------------------------------

    pre_entropy = entropy(
        pre_event_types
    )

    post_entropy = entropy(
        post_event_types
    )

    # --------------------------------------------------------
    # Window titles
    # --------------------------------------------------------

    pre_titles = []
    post_titles = []

    for event in pre_events:

        app = get_active_app(event)

        if app["window_title"]:
            pre_titles.append(
                app["window_title"]
            )

    for event in post_events:

        app = get_active_app(event)

        if app["window_title"]:
            post_titles.append(
                app["window_title"]
            )

    title_jaccard = jaccard(
        pre_titles,
        post_titles
    )

    title_changed = changed(
        pre_titles,
        post_titles
    )

    title_new = new_count(
        pre_titles,
        post_titles
    )

    # --------------------------------------------------------
    # Applications
    # --------------------------------------------------------

    pre_apps = []
    post_apps = []

    for event in pre_events:

        app = get_active_app(event)

        if app["app_name"]:
            pre_apps.append(
                app["app_name"]
            )

    for event in post_events:

        app = get_active_app(event)

        if app["app_name"]:
            post_apps.append(
                app["app_name"]
            )

    app_jaccard = jaccard(
        pre_apps,
        post_apps
    )

    app_changed = changed(
        pre_apps,
        post_apps
    )

    # --------------------------------------------------------
    # Final feature vector
    # --------------------------------------------------------

    features = {

        "pre_event_rate":
            pre_rate,

        "post_event_rate":
            post_rate,

        "event_type_jaccard":
            event_type_jaccard,

        "event_type_changed":
            event_type_changed,

        "event_type_new_count":
            event_type_new,

        "event_type_disappeared_count":
            event_type_disappeared,

        "layer_jaccard":
            layer_jaccard,

        "layer_changed":
            layer_changed,

        "layer_new_count":
            layer_new,

        "layer_disappeared_count":
            layer_disappeared,

        "pre_event_type_entropy":
            pre_entropy,

        "post_event_type_entropy":
            post_entropy,

        "window_title_jaccard":
            title_jaccard,

        "window_title_changed":
            title_changed,

        "window_title_new_count":
            title_new,

        "app_name_jaccard":
            app_jaccard,

        "app_name_changed":
            app_changed,
    }

    return features


# ============================================================
# TRAIN FIXED MODEL
# ============================================================

print("=" * 70)
print("DAY 2 — PHASE 5D v2")
print("TEMPORAL PERSISTENCE + BOUNDARY REFINEMENT")
print("=" * 70)

print(
    "\n[1/7] Training fixed D_lean_full detector..."
)

train_df = pd.read_csv(
    PHASE3_FEATURES
)

train_df = train_df.dropna(
    subset=FEATURES + ["label"]
).copy()

X_train = train_df[FEATURES].astype(float)

y_train = (
    train_df["label"]
    .astype(int)
    .values
)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_train.values
)

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(
    X_scaled,
    y_train
)

print(
    f"Training rows: {len(train_df):,}"
)


# ============================================================
# LOAD GT
# ============================================================

print(
    "\n[2/7] Loading GT boundaries..."
)

gt = pd.read_csv(
    GT_BOUNDARIES
)

gt["timestamp_ms"] = pd.to_numeric(
    gt["timestamp_ms"],
    errors="coerce"
)

gt = gt.dropna(
    subset=[
        "session_id",
        "timestamp_ms"
    ]
).copy()

print(
    f"GT boundaries: {len(gt):,}"
)


# ============================================================
# LOAD HARD NEGATIVE CENTERS
# ============================================================

print(
    "\n[3/7] Loading hard-negative centers..."
)

hard = pd.read_csv(
    HARD_NEGATIVE_FEATURES
)

# The Phase 5B file should contain these.
# We only need the center timestamps here.

required_hard_columns = [
    "session_id",
    "center_timestamp_ms",
]

missing = [
    c
    for c in required_hard_columns
    if c not in hard.columns
]

if missing:

    raise RuntimeError(
        "Hard-negative feature file is missing "
        f"required columns: {missing}"
    )

hard["center_timestamp_ms"] = pd.to_numeric(
    hard["center_timestamp_ms"],
    errors="coerce"
)

hard = hard.dropna(
    subset=[
        "session_id",
        "center_timestamp_ms"
    ]
).copy()

# Remove duplicate centers.
hard_centers = (
    hard[
        [
            "session_id",
            "center_timestamp_ms"
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(
    f"Unique hard-negative centers: "
    f"{len(hard_centers):,}"
)


# ============================================================
# LOAD RAW EVENTS
# ============================================================

print(
    "\n[4/7] Loading raw session events..."
)

session_events = {}

session_dirs = sorted(
    [
        p
        for p in DATASET_ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith("ses_")
    ]
)

print(
    f"Sessions found: {len(session_dirs)}"
)

for i, session_dir in enumerate(
    session_dirs,
    start=1
):

    session_id = session_dir.name

    events = load_session_events(
        session_dir
    )

    session_events[session_id] = events

    if i % 10 == 0 or i == len(session_dirs):

        print(
            f"  loaded {i}/{len(session_dirs)} "
            f"sessions"
        )


# ============================================================
# SCORE TEMPORAL PROFILES
# ============================================================

print(
    "\n[5/7] Building temporal score profiles..."
)


profile_rows = []


def score_center(
    events,
    center_ms,
    window_sec
):

    features = extract_features_at_center(
        events=events,
        center_ms=center_ms,
        window_sec=window_sec
    )

    X = pd.DataFrame(
        [[
            features[f]
            for f in FEATURES
        ]],
        columns=FEATURES
    )

    score = model.predict_proba(
        scaler.transform(
            X.values
        )
    )[0, 1]

    return score


# ------------------------------------------------------------
# Genuine GT boundaries
# ------------------------------------------------------------

total_gt = len(gt)

for idx, boundary in gt.iterrows():

    session_id = str(
        boundary["session_id"]
    )

    gt_ts = float(
        boundary["timestamp_ms"]
    )

    events = session_events.get(
        session_id,
        []
    )

    if not events:
        continue

    for window_sec in WINDOWS:

        scores = []

        for offset in PROFILE_OFFSETS:

            center_ms = (
                gt_ts
                + offset * 1000.0
            )

            score = score_center(
                events,
                center_ms,
                window_sec
            )

            scores.append(score)

            profile_rows.append({

                "kind": "gt_boundary",

                "session_id":
                    session_id,

                "reference_timestamp_ms":
                    gt_ts,

                "center_offset_sec":
                    offset,

                "center_timestamp_ms":
                    center_ms,

                "window_sec":
                    window_sec,

                "boundary_score":
                    score,
            })

    if (
        (idx + 1) % 100 == 0
        or idx + 1 == total_gt
    ):

        print(
            f"  GT profiles: "
            f"{idx + 1}/{total_gt}"
        )


# ------------------------------------------------------------
# Hard negatives
# ------------------------------------------------------------

total_hard = len(hard_centers)

for idx, row in hard_centers.iterrows():

    session_id = str(
        row["session_id"]
    )

    center_ts = float(
        row["center_timestamp_ms"]
    )

    events = session_events.get(
        session_id,
        []
    )

    if not events:
        continue

    for window_sec in WINDOWS:

        for offset in PROFILE_OFFSETS:

            center_ms = (
                center_ts
                + offset * 1000.0
            )

            score = score_center(
                events,
                center_ms,
                window_sec
            )

            profile_rows.append({

                "kind": "hard_negative",

                "session_id":
                    session_id,

                "reference_timestamp_ms":
                    center_ts,

                "center_offset_sec":
                    offset,

                "center_timestamp_ms":
                    center_ms,

                "window_sec":
                    window_sec,

                "boundary_score":
                    score,
            })

    if (
        (idx + 1) % 500 == 0
        or idx + 1 == total_hard
    ):

        print(
            f"  hard-negative profiles: "
            f"{idx + 1}/{total_hard}"
        )


profiles = pd.DataFrame(
    profile_rows
)

profiles_path = (
    OUTPUT_DIR /
    "temporal_score_profiles.csv"
)

profiles.to_csv(
    profiles_path,
    index=False
)

print(
    f"\nSaved: {profiles_path}"
)


# ============================================================
# COMPUTE PERSISTENCE METRICS
# ============================================================

print(
    "\n[6/7] Computing persistence metrics..."
)


summary_rows = []


group_columns = [
    "kind",
    "session_id",
    "reference_timestamp_ms",
    "window_sec",
]


for group_key, group in profiles.groupby(
    group_columns
):

    kind = group_key[0]
    session_id = group_key[1]
    reference_ts = group_key[2]
    window_sec = group_key[3]

    group = group.sort_values(
        "center_offset_sec"
    )

    scores = group[
        "boundary_score"
    ].to_numpy()

    offsets = group[
        "center_offset_sec"
    ].to_numpy()

    if len(scores) == 0:
        continue

    peak_idx = int(
        np.argmax(scores)
    )

    peak_score = float(
        scores[peak_idx]
    )

    peak_offset = float(
        offsets[peak_idx]
    )

    row = {

        "kind":
            kind,

        "session_id":
            session_id,

        "reference_timestamp_ms":
            reference_ts,

        "window_sec":
            window_sec,

        "peak_score":
            peak_score,

        "peak_offset_sec":
            peak_offset,

        "mean_score":
            float(
                np.mean(scores)
            ),

        "median_score":
            float(
                np.median(scores)
            ),

        "score_std":
            float(
                np.std(scores)
            ),

        "score_max_minus_mean":
            peak_score
            - float(
                np.mean(scores)
            ),
    }

    # --------------------------------------------------------
    # Persistence for each threshold
    # --------------------------------------------------------

    for threshold in THRESHOLDS:

        high = (
            scores >= threshold
        )

        fraction_high = (
            np.mean(high)
        )

        row[
            f"fraction_high_{threshold}"
        ] = float(
            fraction_high
        )

        row[
            f"count_high_{threshold}"
        ] = int(
            np.sum(high)
        )

    # --------------------------------------------------------
    # Boundary localization
    # --------------------------------------------------------

    if kind == "gt_boundary":

        row["peak_error_sec"] = abs(
            peak_offset
        )

    else:

        row["peak_error_sec"] = np.nan

    summary_rows.append(
        row
    )


metrics = pd.DataFrame(
    summary_rows
)

metrics_path = (
    OUTPUT_DIR /
    "temporal_persistence_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)

print(
    f"Saved: {metrics_path}"
)


# ============================================================
# GT VS HARD-NEGATIVE COMPARISON
# ============================================================

comparison_rows = []


for window_sec in WINDOWS:

    gt_window = metrics[
        (metrics["kind"] == "gt_boundary")
        &
        (
            np.isclose(
                metrics["window_sec"],
                window_sec
            )
        )
    ]

    hard_window = metrics[
        (metrics["kind"] == "hard_negative")
        &
        (
            np.isclose(
                metrics["window_sec"],
                window_sec
            )
        )
    ]

    for threshold in THRESHOLDS:

        gt_col = (
            f"fraction_high_{threshold}"
        )

        if gt_col in gt_window:

            gt_values = (
                gt_window[gt_col]
                .dropna()
            )

        else:

            gt_values = pd.Series(
                dtype=float
            )

        if gt_col in hard_window:

            hard_values = (
                hard_window[gt_col]
                .dropna()
            )

        else:

            hard_values = pd.Series(
                dtype=float
            )

        gt_mean = (
            gt_values.mean()
            if len(gt_values)
            else np.nan
        )

        hard_mean = (
            hard_values.mean()
            if len(hard_values)
            else np.nan
        )

        comparison_rows.append({

            "window_sec":
                window_sec,

            "threshold":
                threshold,

            "gt_cases":
                len(gt_values),

            "hard_negative_cases":
                len(hard_values),

            "gt_mean_fraction_high":
                gt_mean,

            "hard_negative_mean_fraction_high":
                hard_mean,

            "persistence_gap":
                (
                    gt_mean - hard_mean
                    if not np.isnan(gt_mean)
                    and not np.isnan(hard_mean)
                    else np.nan
                ),

            "gt_median_fraction_high":
                (
                    gt_values.median()
                    if len(gt_values)
                    else np.nan
                ),

            "hard_negative_median_fraction_high":
                (
                    hard_values.median()
                    if len(hard_values)
                    else np.nan
                ),
        })


comparison = pd.DataFrame(
    comparison_rows
)

comparison_path = (
    OUTPUT_DIR /
    "gt_vs_hard_negative_persistence.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# GT LOCALIZATION SUMMARY
# ============================================================

gt_metrics = metrics[
    metrics["kind"] == "gt_boundary"
].copy()


localization_rows = []


for window_sec in WINDOWS:

    subset = gt_metrics[
        np.isclose(
            gt_metrics["window_sec"],
            window_sec
        )
    ]

    if subset.empty:
        continue

    errors = (
        subset["peak_error_sec"]
        .dropna()
    )

    localization_rows.append({

        "window_sec":
            window_sec,

        "cases":
            len(errors),

        "mean_peak_error_sec":
            errors.mean(),

        "median_peak_error_sec":
            errors.median(),

        "within_0_5_sec":
            np.mean(
                errors <= 0.5
            ),

        "within_1_sec":
            np.mean(
                errors <= 1.0
            ),

        "within_2_sec":
            np.mean(
                errors <= 2.0
            ),
    })


localization = pd.DataFrame(
    localization_rows
)

localization_path = (
    OUTPUT_DIR /
    "boundary_peak_localization.csv"
)

localization.to_csv(
    localization_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "GT VS HARD-NEGATIVE TEMPORAL PERSISTENCE"
)

print(
    "=" * 70
)

if comparison.empty:

    print(
        "No comparison rows were produced."
    )

else:

    print(
        comparison.to_string(
            index=False
        )
    )


print(
    "\n" + "=" * 70
)

print(
    "GT BOUNDARY PEAK LOCALIZATION"
)

print(
    "=" * 70
)

if localization.empty:

    print(
        "No GT localization results."
    )

else:

    print(
        localization.to_string(
            index=False
        )
    )


print(
    "\n" + "=" * 70
)

print(
    "PHASE 5D v2 OUTPUTS"
)

print(
    "=" * 70
)

print(
    OUTPUT_DIR
)

print(
    "\nCreated:"
)

print(
    "  temporal_score_profiles.csv"
)

print(
    "  temporal_persistence_metrics.csv"
)

print(
    "  gt_vs_hard_negative_persistence.csv"
)

print(
    "  boundary_peak_localization.csv"
)

print(
    "\n" + "=" * 70
)

print(
    "PHASE 5D v2 COMPLETE"
)

print(
    "=" * 70
)