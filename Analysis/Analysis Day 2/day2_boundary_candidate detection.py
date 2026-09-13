"""
Day 2 - Phase 5F v2
===================

Continuous Boundary Candidate Detection

Pipeline
--------
Raw events
    ↓
Fast temporal feature extraction
    ↓
D_lean Logistic Regression
    ↓
Multiscale transition score
    ↓
Persistence filtering
    ↓
Peak selection
    ↓
Boundary candidates
    ↓
STRICT one-to-one GT matching
    ↓
Precision / Recall / F1
    ↓
Localization error

Major optimizations
-------------------
1. Each session's raw events are loaded only once.
2. Events are stored in numpy arrays.
3. Temporal windows use np.searchsorted().
4. Features are accumulated into arrays.
5. ML inference is performed in batches.
6. GT matching is strict one-to-one.
7. No pandas DataFrame is created for every timestamp.

IMPORTANT
---------
The D_lean feature definitions are kept consistent with our
Day 2 Phase 5A model.

This script is intended to replace the previous slow
day2_boundary_candidate_detection.py.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(
    r"Dataset A\dataset_a"
)

PHASE3_FEATURE_FILE = Path(
    r"Outputs\Day 2\day2_boundary_vs_normal"
    r"\boundary_vs_normal_features.csv"
)

GT_BOUNDARY_FILE = Path(
    r"Outputs\Day 2\day2_gt_boundary_analysis"
    r"\gt_switch_boundaries.csv"
)

OUTPUT_DIR = Path(
    r"Outputs\Day 2\day2_boundary_candidate_detection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEMPORAL CONFIGURATION
# ============================================================

# Multiscale windows used in Phase 5E
WINDOWS = [
    0.5,
    1.0,
    2.0,
    3.0,
]

# Continuous sampling interval
STEP_MS = 100

# Transition score threshold
SCORE_THRESHOLD = 0.70

# Number of consecutive high-score samples
MIN_PERSISTENCE_POINTS = 3

# Minimum distance between final candidates
MIN_SEPARATION_SEC = 1.0

# Maximum GT matching distance
MATCH_TOLERANCE_SEC = 2.0


# ============================================================
# D_LEAN FEATURE SET
# ============================================================

D_LEAN_FEATURES = [
    "pre_event_rate",
    "post_event_rate",

    "pre_event_type_entropy",
    "post_event_type_entropy",

    "event_type_jaccard",
    "event_type_changed",
    "event_type_new_count",
    "event_type_disappeared_count",

    "layer_jaccard",
    "layer_changed",
    "layer_new_count",
    "layer_disappeared_count",

    "app_name_jaccard",
    "app_name_changed",

    "window_title_jaccard",
    "window_title_changed",
    "window_title_new_count",
]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_value(value):
    """
    Convert a categorical value into a stable representation.
    """

    if value is None:
        return None

    if isinstance(value, (dict, list)):

        try:
            return json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False
            )

        except Exception:
            return str(value)

    return str(value)


def safe_entropy(values):
    """
    Shannon entropy for categorical values.
    """

    if len(values) == 0:
        return 0.0

    values = [
        x for x in values
        if x is not None
    ]

    if len(values) == 0:
        return 0.0

    counts = {}

    for value in values:
        counts[value] = (
            counts.get(value, 0) + 1
        )

    total = len(values)

    entropy = 0.0

    for count in counts.values():

        p = count / total

        entropy -= (
            p * np.log2(p + 1e-12)
        )

    return float(entropy)


def jaccard(set_a, set_b):
    """
    Jaccard similarity.
    """

    if not set_a and not set_b:
        return 1.0

    union = set_a | set_b

    if not union:
        return 1.0

    return len(set_a & set_b) / len(union)


# ============================================================
# RAW EVENT EXTRACTION
# ============================================================

def extract_event(raw_event):
    """
    Extract only fields required by the detector.
    """

    context = (
        raw_event.get("context")
        or {}
    )

    active_app = (
        context.get("active_app")
        or {}
    )

    timestamp = raw_event.get(
        "timestamp_ms"
    )

    if timestamp is None:
        return None

    return {
        "timestamp_ms": int(timestamp),

        "event_type": normalize_value(
            raw_event.get("event_type")
        ),

        "layer": normalize_value(
            raw_event.get("layer")
        ),

        "app_name": normalize_value(
            active_app.get("app_name")
        ),

        "window_title": normalize_value(
            active_app.get("window_title")
        ),
    }


# ============================================================
# LOAD SESSION
# ============================================================

def load_session(session_dir):
    """
    Load a complete session once.

    Returns numpy arrays for fast temporal access.
    """

    records = []

    event_files = sorted(
        session_dir.rglob(
            "events.jsonl"
        )
    )

    for event_file in event_files:

        try:

            with event_file.open(
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    try:
                        raw = json.loads(line)

                    except Exception:
                        continue

                    if not isinstance(
                        raw,
                        dict
                    ):
                        continue

                    event = extract_event(
                        raw
                    )

                    if event is not None:
                        records.append(
                            event
                        )

        except Exception as exc:

            print(
                f"WARNING: {event_file}: {exc}"
            )

    if not records:
        return None

    # --------------------------------------------------------
    # Sort once
    # --------------------------------------------------------

    records.sort(
        key=lambda x:
            x["timestamp_ms"]
    )

    # --------------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------------

    return {
        "timestamps": np.asarray(
            [
                x["timestamp_ms"]
                for x in records
            ],
            dtype=np.int64
        ),

        "event_type": np.asarray(
            [
                x["event_type"]
                for x in records
            ],
            dtype=object
        ),

        "layer": np.asarray(
            [
                x["layer"]
                for x in records
            ],
            dtype=object
        ),

        "app_name": np.asarray(
            [
                x["app_name"]
                for x in records
            ],
            dtype=object
        ),

        "window_title": np.asarray(
            [
                x["window_title"]
                for x in records
            ],
            dtype=object
        ),
    }


# ============================================================
# FAST WINDOW BOUNDS
# ============================================================

def window_bounds(
    timestamps,
    center_ms,
    window_ms
):
    """
    Return event-array boundaries using binary search.

    This avoids scanning all events.
    """

    left = np.searchsorted(
        timestamps,
        center_ms - window_ms,
        side="left"
    )

    center_left = np.searchsorted(
        timestamps,
        center_ms,
        side="left"
    )

    center_right = np.searchsorted(
        timestamps,
        center_ms,
        side="right"
    )

    right = np.searchsorted(
        timestamps,
        center_ms + window_ms,
        side="right"
    )

    return (
        left,
        center_left,
        center_right,
        right
    )


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_feature_vector(
    session,
    center_ms,
    window_sec
):
    """
    Calculate one D_lean feature vector.

    This function avoids DataFrames and works directly
    on numpy arrays.
    """

    timestamps = session[
        "timestamps"
    ]

    window_ms = int(
        window_sec * 1000
    )

    (
        left,
        center_left,
        center_right,
        right
    ) = window_bounds(
        timestamps,
        center_ms,
        window_ms
    )

    # --------------------------------------------------------
    # Slices
    # --------------------------------------------------------

    pre_events = session[
        "event_type"
    ][left:center_left]

    post_events = session[
        "event_type"
    ][center_right:right]

    pre_layers = session[
        "layer"
    ][left:center_left]

    post_layers = session[
        "layer"
    ][center_right:right]

    pre_apps = session[
        "app_name"
    ][left:center_left]

    post_apps = session[
        "app_name"
    ][center_right:right]

    pre_titles = session[
        "window_title"
    ][left:center_left]

    post_titles = session[
        "window_title"
    ][center_right:right]

    # --------------------------------------------------------
    # Sets
    # --------------------------------------------------------

    pre_event_set = {
        x for x in pre_events
        if x is not None
    }

    post_event_set = {
        x for x in post_events
        if x is not None
    }

    pre_layer_set = {
        x for x in pre_layers
        if x is not None
    }

    post_layer_set = {
        x for x in post_layers
        if x is not None
    }

    pre_app_set = {
        x for x in pre_apps
        if x is not None
    }

    post_app_set = {
        x for x in post_apps
        if x is not None
    }

    pre_title_set = {
        x for x in pre_titles
        if x is not None
    }

    post_title_set = {
        x for x in post_titles
        if x is not None
    }

    # --------------------------------------------------------
    # Rates
    # --------------------------------------------------------

    duration = max(
        window_sec,
        1e-9
    )

    pre_rate = (
        len(pre_events) / duration
    )

    post_rate = (
        len(post_events) / duration
    )

    # --------------------------------------------------------
    # Event structure
    # --------------------------------------------------------

    event_jaccard = jaccard(
        pre_event_set,
        post_event_set
    )

    event_changed = int(
        pre_event_set != post_event_set
    )

    event_new = len(
        post_event_set
        - pre_event_set
    )

    event_disappeared = len(
        pre_event_set
        - post_event_set
    )

    # --------------------------------------------------------
    # Layer structure
    # --------------------------------------------------------

    layer_jaccard = jaccard(
        pre_layer_set,
        post_layer_set
    )

    layer_changed = int(
        pre_layer_set != post_layer_set
    )

    layer_new = len(
        post_layer_set
        - pre_layer_set
    )

    layer_disappeared = len(
        pre_layer_set
        - post_layer_set
    )

    # --------------------------------------------------------
    # Application structure
    # --------------------------------------------------------

    app_jaccard = jaccard(
        pre_app_set,
        post_app_set
    )

    app_changed = int(
        pre_app_set != post_app_set
    )

    # --------------------------------------------------------
    # Window title
    # --------------------------------------------------------

    title_jaccard = jaccard(
        pre_title_set,
        post_title_set
    )

    title_changed = int(
        pre_title_set != post_title_set
    )

    title_new = len(
        post_title_set
        - pre_title_set
    )

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    pre_entropy = safe_entropy(
        pre_events
    )

    post_entropy = safe_entropy(
        post_events
    )

    # --------------------------------------------------------
    # Return in EXACT feature order
    # --------------------------------------------------------

    return [
        pre_rate,
        post_rate,

        pre_entropy,
        post_entropy,

        event_jaccard,
        event_changed,
        event_new,
        event_disappeared,

        layer_jaccard,
        layer_changed,
        layer_new,
        layer_disappeared,

        app_jaccard,
        app_changed,

        title_jaccard,
        title_changed,
        title_new,
    ]


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():
    """
    Train D_lean Logistic Regression once.
    """

    print(
        "\nLoading Phase 3 feature data..."
    )

    df = pd.read_csv(
        PHASE3_FEATURE_FILE
    )

    missing = [
        feature
        for feature in D_LEAN_FEATURES
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing features:\n"
            + "\n".join(missing)
        )

    X = df[
        D_LEAN_FEATURES
    ].copy()

    y = df[
        "label"
    ].astype(int)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0.0)

    model = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    print(
        f"Training on "
        f"{len(X):,} examples..."
    )

    model.fit(
        X,
        y
    )

    print(
        "Model trained."
    )

    return model


# ============================================================
# BATCH CONTINUOUS SCORING
# ============================================================

def score_session(
    session_id,
    session,
    model
):
    """
    Generate continuous scores.

    IMPORTANT OPTIMIZATION:

    Feature vectors are first collected into numpy arrays.
    The model is then called in batches rather than once
    per timestamp.
    """

    timestamps = session[
        "timestamps"
    ]

    if len(timestamps) == 0:
        return pd.DataFrame()

    start_ms = int(
        timestamps[0]
    )

    end_ms = int(
        timestamps[-1]
    )

    centers = np.arange(
        start_ms,
        end_ms + 1,
        STEP_MS,
        dtype=np.int64
    )

    all_results = []

    # --------------------------------------------------------
    # Calculate each scale separately
    #
    # This lets us batch the ML inference.
    # --------------------------------------------------------

    scale_scores = []

    for scale_number, window_sec in enumerate(
        WINDOWS
    ):

        print(
            f"      scale ±{window_sec}s "
            f"feature extraction...",
            flush=True
        )

        feature_rows = []

        for center_ms in centers:

            row = calculate_feature_vector(
                session,
                int(center_ms),
                window_sec
            )

            feature_rows.append(
                row
            )

        X = np.asarray(
            feature_rows,
            dtype=np.float64
        )

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ----------------------------------------------------
        # ONE batch prediction
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(X)[:, 1]
        )

        scale_scores.append(
            probabilities
        )

    # --------------------------------------------------------
    # Multiscale mean
    # --------------------------------------------------------

    scale_scores = np.asarray(
        scale_scores
    )

    mean_scores = np.mean(
        scale_scores,
        axis=0
    )

    # --------------------------------------------------------
    # Build result dataframe once
    # --------------------------------------------------------

    result = pd.DataFrame(
        {
            "session_id":
                session_id,

            "timestamp_ms":
                centers,

            "timestamp_sec":
                centers / 1000.0,

            "score_0_5":
                scale_scores[0],

            "score_1_0":
                scale_scores[1],

            "score_2_0":
                scale_scores[2],

            "score_3_0":
                scale_scores[3],

            "mean_all_score":
                mean_scores,
        }
    )

    return result


# ============================================================
# PERSISTENCE DETECTION
# ============================================================

def persistent_regions(
    scores,
    timestamps
):
    """
    Find contiguous regions where the score remains above
    threshold.

    A region must contain at least
    MIN_PERSISTENCE_POINTS samples.
    """

    above = (
        scores >= SCORE_THRESHOLD
    )

    regions = []

    start = None

    for i, flag in enumerate(
        above
    ):

        if flag and start is None:

            start = i

        elif not flag and start is not None:

            end = i - 1

            length = (
                end - start + 1
            )

            if (
                length
                >= MIN_PERSISTENCE_POINTS
            ):
                regions.append(
                    (
                        start,
                        end
                    )
                )

            start = None

    # --------------------------------------------------------
    # Region reaching end
    # --------------------------------------------------------

    if start is not None:

        end = len(
            above
        ) - 1

        length = (
            end - start + 1
        )

        if (
            length
            >= MIN_PERSISTENCE_POINTS
        ):
            regions.append(
                (
                    start,
                    end
                )
            )

    return regions


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    score_df
):
    """
    Convert persistent high-score regions into peaks.

    Then merge peaks that are too close.
    """

    candidates = []

    for session_id, group in score_df.groupby(
        "session_id"
    ):

        group = group.sort_values(
            "timestamp_ms"
        ).reset_index(drop=True)

        scores = group[
            "mean_all_score"
        ].to_numpy()

        timestamps = group[
            "timestamp_ms"
        ].to_numpy()

        regions = persistent_regions(
            scores,
            timestamps
        )

        session_candidates = []

        # ----------------------------------------------------
        # Peak per persistent region
        # ----------------------------------------------------

        for start_idx, end_idx in regions:

            region_scores = scores[
                start_idx:end_idx + 1
            ]

            peak_offset = int(
                np.argmax(
                    region_scores
                )
            )

            peak_idx = (
                start_idx
                + peak_offset
            )

            session_candidates.append(
                {
                    "session_id":
                        session_id,

                    "candidate_timestamp_ms":
                        int(
                            timestamps[
                                peak_idx
                            ]
                        ),

                    "candidate_timestamp_sec":
                        float(
                            timestamps[
                                peak_idx
                            ] / 1000.0
                        ),

                    "boundary_score":
                        float(
                            scores[
                                peak_idx
                            ]
                        ),

                    "region_start_ms":
                        int(
                            timestamps[
                                start_idx
                            ]
                        ),

                    "region_end_ms":
                        int(
                            timestamps[
                                end_idx
                            ]
                        ),

                    "persistence_points":
                        int(
                            end_idx
                            - start_idx
                            + 1
                        ),

                    "persistence_sec":
                        float(
                            (
                                timestamps[
                                    end_idx
                                ]
                                -
                                timestamps[
                                    start_idx
                                ]
                            )
                            / 1000.0
                        ),
                }
            )

        # ----------------------------------------------------
        # Sort
        # ----------------------------------------------------

        session_candidates.sort(
            key=lambda x:
                x[
                    "candidate_timestamp_ms"
                ]
        )

        # ----------------------------------------------------
        # Merge close candidates
        # ----------------------------------------------------

        merged = []

        for candidate in session_candidates:

            if not merged:

                merged.append(
                    candidate
                )

                continue

            previous = merged[-1]

            distance_sec = (
                (
                    candidate[
                        "candidate_timestamp_ms"
                    ]
                    -
                    previous[
                        "candidate_timestamp_ms"
                    ]
                )
                / 1000.0
            )

            if (
                distance_sec
                < MIN_SEPARATION_SEC
            ):

                # Keep stronger peak
                if (
                    candidate[
                        "boundary_score"
                    ]
                    >
                    previous[
                        "boundary_score"
                    ]
                ):
                    merged[-1] = candidate

            else:

                merged.append(
                    candidate
                )

        candidates.extend(
            merged
        )

    return pd.DataFrame(
        candidates
    )


# ============================================================
# LOAD GT
# ============================================================

def load_gt():
    """
    Load canonical Phase 2 GT boundary file.
    """

    gt = pd.read_csv(
        GT_BOUNDARY_FILE
    )

    required = [
        "session_id",
        "timestamp_ms"
    ]

    for column in required:

        if column not in gt.columns:

            raise ValueError(
                f"GT missing column: "
                f"{column}"
            )

    gt = gt[
        required
    ].copy()

    gt["timestamp_ms"] = pd.to_numeric(
        gt["timestamp_ms"],
        errors="coerce"
    )

    gt = gt.dropna(
        subset=[
            "timestamp_ms"
        ]
    )

    gt["timestamp_ms"] = (
        gt["timestamp_ms"]
        .astype(np.int64)
    )

    return gt


# ============================================================
# STRICT ONE-TO-ONE MATCHING
# ============================================================

def match_one_to_one(
    candidates,
    gt
):
    """
    Strict one-to-one candidate ↔ GT matching.

    Rules
    -----
    - One candidate can match at most one GT.
    - One GT can match at most one candidate.
    - Match must be within MATCH_TOLERANCE_SEC.
    - Closest candidate-GT pair is preferred.

    This guarantees:

        matched <= GT count
        matched <= candidate count

    Therefore recall can never exceed 1.0.
    """

    tolerance_ms = int(
        MATCH_TOLERANCE_SEC * 1000
    )

    if candidates.empty:

        return pd.DataFrame(
            columns=[
                "session_id",
                "candidate_timestamp_ms",
                "candidate_timestamp_sec",
                "boundary_score",
                "matched",
                "gt_timestamp_ms",
                "error_ms",
                "error_sec",
            ]
        )

    matches = []

    # --------------------------------------------------------
    # Process each session independently
    # --------------------------------------------------------

    for session_id in sorted(
        candidates[
            "session_id"
        ].unique()
    ):

        session_candidates = (
            candidates[
                candidates[
                    "session_id"
                ]
                == session_id
            ]
            .sort_values(
                "candidate_timestamp_ms"
            )
            .reset_index(drop=True)
        )

        session_gt = (
            gt[
                gt[
                    "session_id"
                ]
                == session_id
            ]
            .sort_values(
                "timestamp_ms"
            )
            .reset_index(drop=True)
        )

        gt_times = session_gt[
            "timestamp_ms"
        ].to_numpy(
            dtype=np.int64
        )

        used_gt = set()

        # ----------------------------------------------------
        # Candidate matching
        # ----------------------------------------------------

        for candidate_index, candidate in (
            session_candidates.iterrows()
        ):

            candidate_ts = int(
                candidate[
                    "candidate_timestamp_ms"
                ]
            )

            if len(gt_times) == 0:

                matches.append(
                    {
                        **candidate.to_dict(),
                        "matched": False,
                        "gt_timestamp_ms":
                            np.nan,
                        "error_ms":
                            np.nan,
                        "error_sec":
                            np.nan,
                    }
                )

                continue

            # Binary search
            insertion = np.searchsorted(
                gt_times,
                candidate_ts
            )

            possible_indices = []

            if insertion < len(gt_times):
                possible_indices.append(
                    insertion
                )

            if insertion > 0:
                possible_indices.append(
                    insertion - 1
                )

            # ------------------------------------------------
            # Find nearest UNUSED GT
            # ------------------------------------------------

            best_gt_index = None
            best_error = None

            for gt_index in possible_indices:

                if gt_index in used_gt:
                    continue

                gt_ts = int(
                    gt_times[
                        gt_index
                    ]
                )

                error = abs(
                    candidate_ts
                    - gt_ts
                )

                if (
                    best_error is None
                    or error < best_error
                ):

                    best_error = error
                    best_gt_index = (
                        gt_index
                    )

            # ------------------------------------------------
            # Accept only within tolerance
            # ------------------------------------------------

            if (
                best_gt_index is not None
                and best_error <= tolerance_ms
            ):

                used_gt.add(
                    best_gt_index
                )

                gt_ts = int(
                    gt_times[
                        best_gt_index
                    ]
                )

                matches.append(
                    {
                        **candidate.to_dict(),
                        "matched": True,
                        "gt_timestamp_ms":
                            gt_ts,
                        "error_ms":
                            best_error,
                        "error_sec":
                            best_error / 1000.0,
                    }
                )

            else:

                matches.append(
                    {
                        **candidate.to_dict(),
                        "matched": False,
                        "gt_timestamp_ms":
                            np.nan,
                        "error_ms":
                            np.nan,
                        "error_sec":
                            np.nan,
                    }
                )

    return pd.DataFrame(
        matches
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    candidates,
    gt,
    matches
):
    """
    Calculate strict candidate detection metrics.
    """

    gt_count = len(
        gt
    )

    candidate_count = len(
        candidates
    )

    if matches.empty:

        tp = 0

    else:

        tp = int(
            matches[
                "matched"
            ]
            .fillna(False)
            .sum()
        )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    assert tp <= gt_count, (
        "Internal error: TP > GT"
    )

    assert tp <= candidate_count, (
        "Internal error: TP > candidates"
    )

    fp = (
        candidate_count
        - tp
    )

    fn = (
        gt_count
        - tp
    )

    precision = (
        tp / candidate_count
        if candidate_count > 0
        else 0.0
    )

    recall = (
        tp / gt_count
        if gt_count > 0
        else 0.0
    )

    if (
        precision + recall
        > 0
    ):

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0

    if (
        not matches.empty
        and "error_sec" in matches.columns
    ):

        errors = matches.loc[
            matches[
                "matched"
            ] == True,
            "error_sec"
        ].dropna()

    else:

        errors = pd.Series(
            dtype=float
        )

    return {
        "gt_boundaries":
            gt_count,

        "detected_candidates":
            candidate_count,

        "true_positives":
            tp,

        "false_positives":
            fp,

        "false_negatives":
            fn,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "mean_localization_error_sec":
            errors.mean()
            if len(errors)
            else np.nan,

        "median_localization_error_sec":
            errors.median()
            if len(errors)
            else np.nan,

        "p90_localization_error_sec":
            errors.quantile(0.90)
            if len(errors)
            else np.nan,
    }


# ============================================================
# SESSION-LEVEL RECALL
# ============================================================

def session_recall(
    candidates,
    gt,
    matches
):
    """
    Calculate GT recall independently for each session.
    """

    rows = []

    sessions = sorted(
        set(
            gt["session_id"]
        )
        |
        set(
            candidates["session_id"]
        )
    )

    for session_id in sessions:

        gt_session = gt[
            gt[
                "session_id"
            ]
            == session_id
        ]

        candidate_session = candidates[
            candidates[
                "session_id"
            ]
            == session_id
        ]

        if matches.empty:

            matched_count = 0

        else:

            match_session = matches[
                matches[
                    "session_id"
                ]
                == session_id
            ]

            matched_count = int(
                match_session[
                    "matched"
                ]
                .fillna(False)
                .sum()
            )

        gt_count = len(
            gt_session
        )

        rows.append(
            {
                "session_id":
                    session_id,

                "gt_boundaries":
                    gt_count,

                "detected_candidates":
                    len(
                        candidate_session
                    ),

                "matched_boundaries":
                    matched_count,

                "recall":
                    (
                        matched_count
                        / gt_count
                        if gt_count > 0
                        else 0.0
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "DAY 2 - PHASE 5F v2"
    )
    print(
        "OPTIMIZED CONTINUOUS "
        "BOUNDARY DETECTION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Train model ONCE
    # --------------------------------------------------------

    model = train_model()

    # --------------------------------------------------------
    # Load GT
    # --------------------------------------------------------

    print(
        "\nLoading GT boundaries..."
    )

    gt = load_gt()

    print(
        f"GT boundaries: "
        f"{len(gt):,}"
    )

    # --------------------------------------------------------
    # Discover sessions
    # --------------------------------------------------------

    session_dirs = sorted(
        [
            path
            for path in DATASET_ROOT.iterdir()
            if (
                path.is_dir()
                and path.name.startswith(
                    "ses_"
                )
            )
        ]
    )

    total_sessions = len(
        session_dirs
    )

    print(
        f"Sessions: "
        f"{total_sessions}"
    )

    # --------------------------------------------------------
    # Continuous scoring
    # --------------------------------------------------------

    all_scores = []

    for index, session_dir in enumerate(
        session_dirs,
        start=1
    ):

        session_id = (
            session_dir.name
        )

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"[{index}/{total_sessions}] "
            f"{session_id}"
        )

        # ----------------------------------------------------
        # Load once
        # ----------------------------------------------------

        session = load_session(
            session_dir
        )

        if session is None:

            print(
                "    No valid events."
            )

            continue

        event_count = len(
            session[
                "timestamps"
            ]
        )

        duration = (
            (
                session[
                    "timestamps"
                ][-1]
                -
                session[
                    "timestamps"
                ][0]
            )
            / 1000.0
        )

        print(
            f"    Events: "
            f"{event_count:,}"
        )

        print(
            f"    Duration: "
            f"{duration:.1f}s"
        )

        # ----------------------------------------------------
        # Score session
        # ----------------------------------------------------

        scores = score_session(
            session_id,
            session,
            model
        )

        all_scores.append(
            scores
        )

        print(
            f"    Score points: "
            f"{len(scores):,}"
        )

    # --------------------------------------------------------
    # Combine continuous scores
    # --------------------------------------------------------

    print(
        "\nCombining score data..."
    )

    if all_scores:

        score_df = pd.concat(
            all_scores,
            ignore_index=True
        )

    else:

        score_df = pd.DataFrame()

    score_file = (
        OUTPUT_DIR
        / "continuous_transition_scores.csv"
    )

    score_df.to_csv(
        score_file,
        index=False
    )

    print(
        f"Saved:\n{score_file}"
    )

    # --------------------------------------------------------
    # Candidate detection
    # --------------------------------------------------------

    print(
        "\nRunning persistence "
        "and peak detection..."
    )

    candidates = generate_candidates(
        score_df
    )

    candidate_file = (
        OUTPUT_DIR
        / "detected_boundary_candidates.csv"
    )

    candidates.to_csv(
        candidate_file,
        index=False
    )

    print(
        f"Candidates detected: "
        f"{len(candidates):,}"
    )

    # --------------------------------------------------------
    # Strict one-to-one matching
    # --------------------------------------------------------

    print(
        "\nRunning STRICT "
        "one-to-one GT matching..."
    )

    matches = match_one_to_one(
        candidates,
        gt
    )

    match_file = (
        OUTPUT_DIR
        / "candidate_gt_matches.csv"
    )

    matches.to_csv(
        match_file,
        index=False
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        candidates,
        gt,
        matches
    )

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_file = (
        OUTPUT_DIR
        / "boundary_detection_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_file,
        index=False
    )

    # --------------------------------------------------------
    # Per-session recall
    # --------------------------------------------------------

    recall_df = session_recall(
        candidates,
        gt,
        matches
    )

    recall_file = (
        OUTPUT_DIR
        / "gt_detection_recall.csv"
    )

    recall_df.to_csv(
        recall_file,
        index=False
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PHASE 5F v2 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"GT boundaries       : "
        f"{metrics['gt_boundaries']:,}"
    )

    print(
        f"Candidates detected : "
        f"{metrics['detected_candidates']:,}"
    )

    print(
        f"True positives      : "
        f"{metrics['true_positives']:,}"
    )

    print(
        f"False positives     : "
        f"{metrics['false_positives']:,}"
    )

    print(
        f"False negatives     : "
        f"{metrics['false_negatives']:,}"
    )

    print(
        f"Precision            : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall               : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1                   : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"Mean localization    : "
        f"{metrics['mean_localization_error_sec']:.3f}s"
    )

    print(
        f"Median localization  : "
        f"{metrics['median_localization_error_sec']:.3f}s"
    )

    print(
        f"P90 localization     : "
        f"{metrics['p90_localization_error_sec']:.3f}s"
    )

    print(
        "\nOutputs:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nSanity check:"
    )

    print(
        "Recall <= 1.0       : "
        f"{metrics['recall'] <= 1.0}"
    )

    print(
        "TP <= GT            : "
        f"{metrics['true_positives'] <= metrics['gt_boundaries']}"
    )

    print(
        "TP <= candidates    : "
        f"{metrics['true_positives'] <= metrics['detected_candidates']}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()