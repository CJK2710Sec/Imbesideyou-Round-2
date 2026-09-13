from pathlib import Path
import json
import math

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# DAY 2 — PHASE 5E
# MULTI-SCALE TRANSITION SCORING
# ============================================================

DATASET_ROOT = Path(
    r"Dataset A\dataset_a"
)

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
    r"Outputs\Day 2\day2_multiscale_transition"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SCALES
# ============================================================

SCALES = [
    0.5,
    1.0,
    2.0,
    3.0,
]


# ============================================================
# FIXED D_LEAN_FULL FEATURES
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
# MULTI-SCALE COMBINATIONS
# ============================================================

COMBINATIONS = {

    # Each scale independently.
    "single_0_5": [0.5],
    "single_1_0": [1.0],
    "single_2_0": [2.0],
    "single_3_0": [3.0],

    # Simple mean across scales.
    "mean_all": [
        0.5,
        1.0,
        2.0,
        3.0,
    ],

    # Short + medium.
    "mean_0_5_1": [
        0.5,
        1.0,
    ],

    # Medium + localization-friendly scales.
    "mean_1_2": [
        1.0,
        2.0,
    ],

    "mean_2_3": [
        2.0,
        3.0,
    ],

    # Short + long context.
    "mean_0_5_2_3": [
        0.5,
        2.0,
        3.0,
    ],
}


# ============================================================
# THRESHOLDS
# ============================================================

THRESHOLDS = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
]


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):

    try:
        return float(value)

    except Exception:

        return np.nan


def entropy(values):

    if not values:
        return 0.0

    counts = {}

    for value in values:

        value = str(value)

        counts[value] = (
            counts.get(value, 0) + 1
        )

    total = sum(
        counts.values()
    )

    if total == 0:
        return 0.0

    result = 0.0

    for count in counts.values():

        p = count / total

        if p > 0:

            result -= (
                p * math.log2(p)
            )

    return result


def jaccard(a, b):

    a = set(a)
    b = set(b)

    if not a and not b:

        return 1.0

    union = a | b

    if not union:

        return 1.0

    return (
        len(a & b) /
        len(union)
    )


def changed(a, b):

    return int(
        set(a) != set(b)
    )


def new_count(a, b):

    return len(
        set(b) - set(a)
    )


def disappeared_count(a, b):

    return len(
        set(a) - set(b)
    )


def get_active_app(event):

    context = event.get(
        "context",
        {}
    )

    if not isinstance(
        context,
        dict
    ):

        return {
            "app_name": "",
            "window_title": "",
        }

    active_app = context.get(
        "active_app",
        {}
    )

    if not isinstance(
        active_app,
        dict
    ):

        return {
            "app_name": "",
            "window_title": "",
        }

    return {

        "app_name":
            str(
                active_app.get(
                    "app_name",
                    ""
                )
            ),

        "window_title":
            str(
                active_app.get(
                    "window_title",
                    ""
                )
            ),
    }


def event_type(event):

    value = event.get(
        "event_type",
        ""
    )

    if value is None:

        return ""

    return str(value)


def event_layer(event):

    value = event.get(
        "layer",
        ""
    )

    if value is None:

        return ""

    return str(value)


# ============================================================
# LOAD RAW EVENTS
# ============================================================

def load_session_events(
    session_dir
):

    events = []

    files = sorted(
        session_dir.rglob(
            "events.jsonl"
        )
    )

    for path in files:

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

                        event = json.loads(
                            line
                        )

                    except Exception:

                        continue

                    if not isinstance(
                        event,
                        dict
                    ):

                        continue

                    ts = safe_float(
                        event.get(
                            "timestamp_ms"
                        )
                    )

                    if np.isnan(ts):

                        continue

                    event[
                        "_timestamp_ms"
                    ] = ts

                    events.append(
                        event
                    )

        except Exception as exc:

            print(
                f"WARNING reading {path}: "
                f"{exc}"
            )

    events.sort(
        key=lambda x:
        x["_timestamp_ms"]
    )

    return events


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    events,
    center_ms,
    window_sec
):

    half_ms = (
        window_sec * 1000.0
    )

    pre_start = (
        center_ms - half_ms
    )

    pre_end = center_ms

    post_start = center_ms

    post_end = (
        center_ms + half_ms
    )

    pre = []
    post = []

    for event in events:

        ts = event[
            "_timestamp_ms"
        ]

        if (
            pre_start
            <= ts
            < pre_end
        ):

            pre.append(event)

        elif (
            post_start
            <= ts
            <= post_end
        ):

            post.append(event)

    # --------------------------------------------------------
    # Event types
    # --------------------------------------------------------

    pre_types = [
        event_type(e)
        for e in pre
        if event_type(e)
    ]

    post_types = [
        event_type(e)
        for e in post
        if event_type(e)
    ]

    # --------------------------------------------------------
    # Layers
    # --------------------------------------------------------

    pre_layers = [
        event_layer(e)
        for e in pre
        if event_layer(e)
    ]

    post_layers = [
        event_layer(e)
        for e in post
        if event_layer(e)
    ]

    # --------------------------------------------------------
    # Titles
    # --------------------------------------------------------

    pre_titles = []
    post_titles = []

    for event in pre:

        app = get_active_app(
            event
        )

        if app["window_title"]:

            pre_titles.append(
                app["window_title"]
            )

    for event in post:

        app = get_active_app(
            event
        )

        if app["window_title"]:

            post_titles.append(
                app["window_title"]
            )

    # --------------------------------------------------------
    # Apps
    # --------------------------------------------------------

    pre_apps = []
    post_apps = []

    for event in pre:

        app = get_active_app(
            event
        )

        if app["app_name"]:

            pre_apps.append(
                app["app_name"]
            )

    for event in post:

        app = get_active_app(
            event
        )

        if app["app_name"]:

            post_apps.append(
                app["app_name"]
            )

    # --------------------------------------------------------
    # Feature dictionary
    # --------------------------------------------------------

    return {

        "pre_event_rate":
            len(pre) / window_sec,

        "post_event_rate":
            len(post) / window_sec,

        "event_type_jaccard":
            jaccard(
                pre_types,
                post_types
            ),

        "event_type_changed":
            changed(
                pre_types,
                post_types
            ),

        "event_type_new_count":
            new_count(
                pre_types,
                post_types
            ),

        "event_type_disappeared_count":
            disappeared_count(
                pre_types,
                post_types
            ),

        "layer_jaccard":
            jaccard(
                pre_layers,
                post_layers
            ),

        "layer_changed":
            changed(
                pre_layers,
                post_layers
            ),

        "layer_new_count":
            new_count(
                pre_layers,
                post_layers
            ),

        "layer_disappeared_count":
            disappeared_count(
                pre_layers,
                post_layers
            ),

        "pre_event_type_entropy":
            entropy(
                pre_types
            ),

        "post_event_type_entropy":
            entropy(
                post_types
            ),

        "window_title_jaccard":
            jaccard(
                pre_titles,
                post_titles
            ),

        "window_title_changed":
            changed(
                pre_titles,
                post_titles
            ),

        "window_title_new_count":
            new_count(
                pre_titles,
                post_titles
            ),

        "app_name_jaccard":
            jaccard(
                pre_apps,
                post_apps
            ),

        "app_name_changed":
            changed(
                pre_apps,
                post_apps
            ),
    }


# ============================================================
# TRAIN FIXED MODEL
# ============================================================

print("=" * 70)
print("DAY 2 — PHASE 5E")
print("MULTI-SCALE TRANSITION SCORING")
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

X_train = (
    train_df[
        FEATURES
    ]
    .astype(float)
)

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
    f"Training rows: "
    f"{len(train_df):,}"
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
        "timestamp_ms",
    ]
).copy()

print(
    f"GT boundaries: "
    f"{len(gt):,}"
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

required = [
    "session_id",
    "center_timestamp_ms",
]

missing = [
    c
    for c in required
    if c not in hard.columns
]

if missing:

    raise RuntimeError(
        "Missing required columns: "
        f"{missing}"
    )

hard["center_timestamp_ms"] = pd.to_numeric(
    hard["center_timestamp_ms"],
    errors="coerce"
)

hard = hard.dropna(
    subset=[
        "session_id",
        "center_timestamp_ms",
    ]
).copy()

hard_centers = (
    hard[
        [
            "session_id",
            "center_timestamp_ms",
        ]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)

print(
    f"Hard-negative centers: "
    f"{len(hard_centers):,}"
)


# ============================================================
# LOAD RAW EVENTS
# ============================================================

print(
    "\n[4/7] Loading raw events..."
)

session_dirs = sorted(
    [
        p
        for p in DATASET_ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith("ses_")
    ]
)

session_events = {}

for i, session_dir in enumerate(
    session_dirs,
    start=1
):

    session_id = (
        session_dir.name
    )

    events = load_session_events(
        session_dir
    )

    session_events[
        session_id
    ] = events

    if (
        i % 10 == 0
        or i == len(session_dirs)
    ):

        print(
            f"  loaded "
            f"{i}/{len(session_dirs)}"
        )


# ============================================================
# SCORE ALL CENTERS
# ============================================================

print(
    "\n[5/7] Scoring temporal centers..."
)


score_rows = []


def score_one(
    events,
    center_ms,
    window_sec
):

    feature_dict = extract_features(
        events,
        center_ms,
        window_sec
    )

    X = pd.DataFrame(
        [[
            feature_dict[f]
            for f in FEATURES
        ]],
        columns=FEATURES
    )

    X_scaled = scaler.transform(
        X.values
    )

    return float(
        model.predict_proba(
            X_scaled
        )[0, 1]
    )


# ------------------------------------------------------------
# GT centers
# ------------------------------------------------------------

for idx, row in gt.iterrows():

    session_id = str(
        row["session_id"]
    )

    center_ms = float(
        row["timestamp_ms"]
    )

    events = session_events.get(
        session_id,
        []
    )

    if not events:

        continue

    for window_sec in SCALES:

        score = score_one(
            events,
            center_ms,
            window_sec
        )

        score_rows.append({

            "kind":
                "gt_boundary",

            "session_id":
                session_id,

            "reference_timestamp_ms":
                center_ms,

            "window_sec":
                window_sec,

            "boundary_score":
                score,
        })

    if (
        (idx + 1) % 100 == 0
        or idx + 1 == len(gt)
    ):

        print(
            f"  GT centers: "
            f"{idx + 1}/{len(gt)}"
        )


# ------------------------------------------------------------
# Hard negatives
# ------------------------------------------------------------

for idx, row in hard_centers.iterrows():

    session_id = str(
        row["session_id"]
    )

    center_ms = float(
        row["center_timestamp_ms"]
    )

    events = session_events.get(
        session_id,
        []
    )

    if not events:

        continue

    for window_sec in SCALES:

        score = score_one(
            events,
            center_ms,
            window_sec
        )

        score_rows.append({

            "kind":
                "hard_negative",

            "session_id":
                session_id,

            "reference_timestamp_ms":
                center_ms,

            "window_sec":
                window_sec,

            "boundary_score":
                score,
        })

    if (
        (idx + 1) % 500 == 0
        or idx + 1 == len(hard_centers)
    ):

        print(
            f"  hard-negative centers: "
            f"{idx + 1}/{len(hard_centers)}"
        )


scores = pd.DataFrame(
    score_rows
)

scores_path = (
    OUTPUT_DIR /
    "multiscale_component_scores.csv"
)

scores.to_csv(
    scores_path,
    index=False
)

print(
    f"\nSaved: {scores_path}"
)


# ============================================================
# PIVOT SCALE SCORES
# ============================================================

pivot = scores.pivot_table(
    index=[
        "kind",
        "session_id",
        "reference_timestamp_ms",
    ],
    columns="window_sec",
    values="boundary_score",
    aggfunc="first"
).reset_index()


for scale in SCALES:

    if scale not in pivot.columns:

        pivot[scale] = np.nan


# ============================================================
# BUILD MULTI-SCALE SCORES
# ============================================================

print(
    "\n[6/7] Building multi-scale scores..."
)


for name, scales in COMBINATIONS.items():

    columns = [
        pivot[scale]
        for scale in scales
    ]

    pivot[name] = pd.concat(
        columns,
        axis=1
    ).mean(
        axis=1
    )


multiscale_path = (
    OUTPUT_DIR /
    "multiscale_scores.csv"
)

pivot.to_csv(
    multiscale_path,
    index=False
)

print(
    f"Saved: {multiscale_path}"
)


# ============================================================
# EVALUATE SCORES
# ============================================================

print(
    "\n[7/7] Evaluating multi-scale scoring..."
)


evaluation_rows = []


for name in COMBINATIONS:

    temp = pivot[
        [
            "kind",
            name,
        ]
    ].dropna()

    y_true = (
        temp["kind"]
        .eq("gt_boundary")
        .astype(int)
        .values
    )

    y_score = (
        temp[name]
        .astype(float)
        .values
    )

    if len(
        np.unique(y_true)
    ) < 2:

        continue

    roc = roc_auc_score(
        y_true,
        y_score
    )

    pr = average_precision_score(
        y_true,
        y_score
    )

    for threshold in THRESHOLDS:

        y_pred = (
            y_score >= threshold
        ).astype(int)

        evaluation_rows.append({

            "configuration":
                name,

            "threshold":
                threshold,

            "roc_auc":
                roc,

            "average_precision":
                pr,

            "precision":
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                ),

            "gt_cases":
                int(
                    np.sum(
                        y_true == 1
                    )
                ),

            "hard_negative_cases":
                int(
                    np.sum(
                        y_true == 0
                    )
                ),
        })


evaluation = pd.DataFrame(
    evaluation_rows
)

evaluation_path = (
    OUTPUT_DIR /
    "multiscale_evaluation.csv"
)

evaluation.to_csv(
    evaluation_path,
    index=False
)


# ============================================================
# SIMPLE SCORE DISTRIBUTION SUMMARY
# ============================================================

distribution_rows = []


for name in COMBINATIONS:

    gt_scores = pivot.loc[
        pivot["kind"]
        == "gt_boundary",
        name
    ].dropna()

    hard_scores = pivot.loc[
        pivot["kind"]
        == "hard_negative",
        name
    ].dropna()

    if (
        gt_scores.empty
        or hard_scores.empty
    ):

        continue

    distribution_rows.append({

        "configuration":
            name,

        "gt_mean":
            gt_scores.mean(),

        "gt_median":
            gt_scores.median(),

        "gt_p10":
            gt_scores.quantile(0.10),

        "gt_p90":
            gt_scores.quantile(0.90),

        "hard_mean":
            hard_scores.mean(),

        "hard_median":
            hard_scores.median(),

        "hard_p90":
            hard_scores.quantile(0.90),

        "hard_p95":
            hard_scores.quantile(0.95),

        "hard_p99":
            hard_scores.quantile(0.99),
    })


distribution = pd.DataFrame(
    distribution_rows
)

distribution_path = (
    OUTPUT_DIR /
    "multiscale_score_distribution.csv"
)

distribution.to_csv(
    distribution_path,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "MULTI-SCALE ROC-AUC / PR-AUC"
)

print(
    "=" * 70
)

if evaluation.empty:

    print(
        "No evaluation results."
    )

else:

    best = (
        evaluation
        .sort_values(
            [
                "roc_auc",
                "f1",
            ],
            ascending=False
        )
        .drop_duplicates(
            "configuration"
        )
    )

    print(
        best[
            [
                "configuration",
                "roc_auc",
                "average_precision",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False
        )
    )


print(
    "\n" + "=" * 70
)

print(
    "SCORE DISTRIBUTIONS"
)

print(
    "=" * 70
)

if distribution.empty:

    print(
        "No distribution results."
    )

else:

    print(
        distribution.to_string(
            index=False
        )
    )


print(
    "\n" + "=" * 70
)

print(
    "OUTPUTS"
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
    "  multiscale_component_scores.csv"
)

print(
    "  multiscale_scores.csv"
)

print(
    "  multiscale_evaluation.csv"
)

print(
    "  multiscale_score_distribution.csv"
)

print(
    "\n" + "=" * 70
)

print(
    "PHASE 5E COMPLETE"
)

print(
    "=" * 70
)