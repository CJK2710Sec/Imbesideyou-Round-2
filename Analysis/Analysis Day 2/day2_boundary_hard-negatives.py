from pathlib import Path
import json
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(r"Dataset A\dataset_a")

PHASE3_FEATURES = Path(
    r"Outputs\Day 2\day2_boundary_vs_normal\boundary_vs_normal_features.csv"
)

PHASE2_BOUNDARIES = Path(
    r"Outputs\Day 2\day2_gt_boundary_analysis\gt_switch_boundaries.csv"
)

OUTPUT_DIR = Path(
    r"Outputs\Day 2\day2_boundary_hard_negative"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


WINDOWS = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

# We use the same feature family selected for D_lean_full.
D_LEAN_FEATURES = [
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

# Minimum distance between a hard-negative center and a GT event.
# This prevents the actual GT transition from entering the feature window.
GUARD_SECONDS = 0.25

# Candidate distances from a real process boundary.
#
# Example:
#   ±0.5 detector -> hard negative must be > 0.5 sec away
#   ±1 detector   -> hard negative must be > 1 sec away
#   etc.
#
# We additionally sample a band immediately outside the detector
# window to make the test difficult.
HARD_NEGATIVE_GAP = 0.25

RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def safe_entropy(values):
    """Shannon entropy of categorical values."""
    if not values:
        return 0.0

    counts = pd.Series(values).value_counts().values.astype(float)

    probs = counts / counts.sum()

    return float(
        -(probs * np.log2(probs + 1e-12)).sum()
    )


def jaccard(a, b):
    """Jaccard similarity between two sets."""
    a = set(a)
    b = set(b)

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


def clean_value(value):
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


# ============================================================
# LOAD RAW EVENTS
# ============================================================

def load_raw_events():
    rows = []

    session_dirs = sorted(
        p for p in DATASET_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("ses_")
    )

    print(f"Sessions found: {len(session_dirs)}")

    for session_dir in session_dirs:

        event_files = list(
            session_dir.rglob("events.jsonl")
        )

        for event_file in event_files:

            with open(
                event_file,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    try:
                        event = json.loads(line)

                    except Exception:
                        continue

                    if not isinstance(event, dict):
                        continue

                    ts = event.get("timestamp_ms")

                    if ts is None:
                        continue

                    try:
                        ts = float(ts)

                    except Exception:
                        continue

                    context = event.get("context") or {}

                    if not isinstance(context, dict):
                        context = {}

                    active_app = context.get(
                        "active_app"
                    ) or {}

                    if not isinstance(active_app, dict):
                        active_app = {}

                    rows.append({
                        "session_id": event.get(
                            "session_id",
                            session_dir.name
                        ),

                        "timestamp_ms": ts,

                        "event_type": clean_value(
                            event.get("event_type")
                        ),

                        "layer": clean_value(
                            event.get("layer")
                        ),

                        "app_name": clean_value(
                            active_app.get("app_name")
                        ),

                        "process_name": clean_value(
                            active_app.get("process_name")
                        ),

                        "window_title": clean_value(
                            active_app.get("window_title")
                        ),
                    })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No raw events were loaded."
        )

    df = df.sort_values(
        ["session_id", "timestamp_ms"]
    ).reset_index(drop=True)

    return df


def load_gt_boundaries():
    """
    Load the already-validated GT switch boundaries produced by
    Day 2 Phase 1.

    Phase 1 successfully identified all 1,590
    process_switched_out candidate boundaries, so Phase 5B
    should reuse that canonical output instead of reparsing
    gt.jsonl independently.
    """

    if not PHASE2_BOUNDARIES.exists():
        raise FileNotFoundError(
            f"Phase 2 boundary file not found:\n"
            f"{PHASE2_BOUNDARIES.resolve()}\n\n"
            f"Run day2_gt_boundary_analysis.py first."
        )

    df = pd.read_csv(
        PHASE2_BOUNDARIES
    )

    print(
        f"Loaded Phase 2 boundary file: "
        f"{len(df):,} rows"
    )

    print(
        "Columns:",
        list(df.columns)
    )

    # --------------------------------------------------------
    # Identify timestamp column
    # --------------------------------------------------------

    possible_timestamp_columns = [
        "timestamp_ms",
        "boundary_timestamp_ms",
        "ts_ms",
        "timestamp",
    ]

    timestamp_col = next(
        (
            col
            for col in possible_timestamp_columns
            if col in df.columns
        ),
        None,
    )

    if timestamp_col is None:
        raise RuntimeError(
            "Could not find a timestamp column in "
            "gt_switch_boundaries.csv.\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Identify session column
    # --------------------------------------------------------

    possible_session_columns = [
        "session_id",
        "session",
    ]

    session_col = next(
        (
            col
            for col in possible_session_columns
            if col in df.columns
        ),
        None,
    )

    if session_col is None:
        raise RuntimeError(
            "Could not find session column in "
            "gt_switch_boundaries.csv.\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    result = pd.DataFrame({
        "session_id": df[session_col].astype(str),
        "timestamp_ms": pd.to_numeric(
            df[timestamp_col],
            errors="coerce",
        ),
    })

    result = result.dropna(
        subset=[
            "session_id",
            "timestamp_ms",
        ]
    )

    result = (
        result
        .drop_duplicates(
            subset=[
                "session_id",
                "timestamp_ms",
            ]
        )
        .sort_values(
            [
                "session_id",
                "timestamp_ms",
            ]
        )
        .reset_index(drop=True)
    )

    if result.empty:
        raise RuntimeError(
            "Phase 2 boundary file contained no valid "
            "boundary timestamps."
        )

    return result

# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    session_events,
    center_ms,
    window_seconds,
):
    """
    Extract exactly the type of temporal/context features
    used by the Phase 5 detector.
    """

    w = window_seconds * 1000.0

    pre_start = center_ms - w
    post_end = center_ms + w

    pre = session_events[
        (session_events["timestamp_ms"] >= pre_start)
        &
        (session_events["timestamp_ms"] < center_ms)
    ]

    post = session_events[
        (session_events["timestamp_ms"] > center_ms)
        &
        (session_events["timestamp_ms"] <= post_end)
    ]

    total_seconds = max(window_seconds, 1e-6)

    features = {}

    # --------------------------------------------------------
    # Temporal
    # --------------------------------------------------------

    features["pre_event_rate"] = (
        len(pre) / total_seconds
    )

    features["post_event_rate"] = (
        len(post) / total_seconds
    )

    # --------------------------------------------------------
    # Event type
    # --------------------------------------------------------

    pre_types = [
        x for x in pre["event_type"].tolist()
        if x is not None
    ]

    post_types = [
        x for x in post["event_type"].tolist()
        if x is not None
    ]

    pre_type_set = set(pre_types)
    post_type_set = set(post_types)

    features["event_type_jaccard"] = jaccard(
        pre_type_set,
        post_type_set,
    )

    features["event_type_changed"] = int(
        pre_type_set != post_type_set
    )

    features["event_type_new_count"] = len(
        post_type_set - pre_type_set
    )

    features["event_type_disappeared_count"] = len(
        pre_type_set - post_type_set
    )

    features["pre_event_type_entropy"] = safe_entropy(
        pre_types
    )

    features["post_event_type_entropy"] = safe_entropy(
        post_types
    )

    # --------------------------------------------------------
    # Layer
    # --------------------------------------------------------

    pre_layers = [
        x for x in pre["layer"].tolist()
        if x is not None
    ]

    post_layers = [
        x for x in post["layer"].tolist()
        if x is not None
    ]

    pre_layer_set = set(pre_layers)
    post_layer_set = set(post_layers)

    features["layer_jaccard"] = jaccard(
        pre_layer_set,
        post_layer_set,
    )

    features["layer_changed"] = int(
        pre_layer_set != post_layer_set
    )

    features["layer_new_count"] = len(
        post_layer_set - pre_layer_set
    )

    features["layer_disappeared_count"] = len(
        pre_layer_set - post_layer_set
    )

    # --------------------------------------------------------
    # Window title
    # --------------------------------------------------------

    pre_titles = set(
        x for x in pre["window_title"].tolist()
        if x is not None
    )

    post_titles = set(
        x for x in post["window_title"].tolist()
        if x is not None
    )

    features["window_title_jaccard"] = jaccard(
        pre_titles,
        post_titles,
    )

    features["window_title_changed"] = int(
        pre_titles != post_titles
    )

    features["window_title_new_count"] = len(
        post_titles - pre_titles
    )

    # --------------------------------------------------------
    # Application
    # --------------------------------------------------------

    pre_apps = set(
        x for x in pre["app_name"].tolist()
        if x is not None
    )

    post_apps = set(
        x for x in post["app_name"].tolist()
        if x is not None
    )

    features["app_name_jaccard"] = jaccard(
        pre_apps,
        post_apps,
    )

    features["app_name_changed"] = int(
        pre_apps != post_apps
    )

    return features


# ============================================================
# FIND HARD NEGATIVES
# ============================================================

def make_hard_negatives(
    raw_df,
    boundaries,
):
    """
    Select ordinary raw-event timestamps close to GT boundaries.

    Crucially:
        the hard-negative center must be sufficiently far from
        every GT boundary so that the feature window does not
        directly contain the actual transition.
    """

    hard_rows = []

    boundaries_by_session = {
        sid: grp["timestamp_ms"].values
        for sid, grp in boundaries.groupby("session_id")
    }

    for session_id, session_events in raw_df.groupby(
        "session_id",
        sort=False,
    ):

        session_events = session_events.sort_values(
            "timestamp_ms"
        )

        gt_times = boundaries_by_session.get(
            session_id,
            np.array([]),
        )

        if len(gt_times) == 0:
            continue

        raw_times = session_events[
            "timestamp_ms"
        ].values

        for gt_time in gt_times:

            # Candidate events around the transition.
            distances = np.abs(
                raw_times - gt_time
            )

            # Use a challenging region around the boundary.
            #
            # We don't take the event exactly at the boundary.
            candidate_mask = (
                (distances >= 0.75 * 1000)
                &
                (distances <= 5.0 * 1000)
            )

            candidates = raw_times[
                candidate_mask
            ]

            if len(candidates) == 0:
                continue

            # For every detector window, the candidate must be
            # outside that window + guard.
            for window in WINDOWS:

                minimum_distance = (
                    window + HARD_NEGATIVE_GAP
                ) * 1000.0

                valid = candidates[
                    np.abs(candidates - gt_time)
                    >= minimum_distance
                ]

                if len(valid) == 0:
                    continue

                # Pick the closest valid raw event.
                distances_valid = np.abs(
                    valid - gt_time
                )

                center = valid[
                    np.argmin(distances_valid)
                ]

                hard_rows.append({
                    "session_id": session_id,
                    "center_timestamp_ms": center,
                    "nearest_gt_boundary_ms": gt_time,
                    "distance_from_gt_sec":
                        abs(center - gt_time) / 1000.0,
                    "window_sec": window,
                    "label": 0,
                })

    hard_df = pd.DataFrame(hard_rows)

    if hard_df.empty:
        raise RuntimeError(
            "Could not construct hard negatives."
        )

    # Remove duplicate center/window combinations.
    hard_df = hard_df.drop_duplicates(
        subset=[
            "session_id",
            "center_timestamp_ms",
            "window_sec",
        ]
    )

    return hard_df.reset_index(drop=True)


# ============================================================
# BUILD HARD-NEGATIVE FEATURES
# ============================================================

def build_hard_features(
    raw_df,
    hard_df,
):
    rows = []

    grouped = {
        sid: grp
        for sid, grp in raw_df.groupby(
            "session_id",
            sort=False,
        )
    }

    for _, row in hard_df.iterrows():

        session_events = grouped.get(
            row["session_id"]
        )

        if session_events is None:
            continue

        features = extract_features(
            session_events,
            row["center_timestamp_ms"],
            row["window_sec"],
        )

        result = {
            "session_id": row["session_id"],
            "center_timestamp_ms":
                row["center_timestamp_ms"],
            "nearest_gt_boundary_ms":
                row["nearest_gt_boundary_ms"],
            "distance_from_gt_sec":
                row["distance_from_gt_sec"],
            "window_sec":
                row["window_sec"],
            "label": 0,
        }

        result.update(features)

        rows.append(result)

    return pd.DataFrame(rows)


# ============================================================
# TRAIN FIXED PHASE 5 DETECTOR
# ============================================================

def train_fixed_detector(phase3_df):
    """
    Train the same interpretable baseline used for Phase 5.

    IMPORTANT:
    We do not tune anything using the hard-negative dataset.
    """

    missing = [
        f for f in D_LEAN_FEATURES
        if f not in phase3_df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing D_lean_full features:\n"
            + "\n".join(missing)
        )

    X = phase3_df[
        D_LEAN_FEATURES
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    y = phase3_df["label"].astype(int)

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=RANDOM_SEED,
            )
        ),
    ])

    model.fit(X, y)

    return model


# ============================================================
# EVALUATE
# ============================================================

def evaluate_hard_negatives(
    model,
    hard_features,
):
    results = []

    for window in WINDOWS:

        df = hard_features[
            np.isclose(
                hard_features["window_sec"],
                window,
            )
        ].copy()

        if df.empty:
            continue

        X = df[
            D_LEAN_FEATURES
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0)

        hard_scores = model.predict_proba(
            X
        )[:, 1]

        df["boundary_score"] = hard_scores

        results.append({
            "window_sec": window,

            "hard_negative_count":
                len(df),

            "mean_score":
                float(np.mean(hard_scores)),

            "median_score":
                float(np.median(hard_scores)),

            "p95_score":
                float(np.percentile(
                    hard_scores,
                    95,
                )),

            "p99_score":
                float(np.percentile(
                    hard_scores,
                    99,
                )),

            "max_score":
                float(np.max(hard_scores)),

            # Threshold used by the original detector.
            "false_positive_rate_at_0_5":
                float(np.mean(
                    hard_scores >= 0.5
                )),

            "false_positive_rate_at_0_7":
                float(np.mean(
                    hard_scores >= 0.7
                )),

            "false_positive_rate_at_0_9":
                float(np.mean(
                    hard_scores >= 0.9
                )),
        })

    return pd.DataFrame(results), hard_features


# ============================================================
# BOUNDARY LOCALIZATION
# ============================================================

def boundary_localization(
    model,
    raw_df,
    boundaries,
):
    """
    Around every GT boundary, evaluate nearby candidate raw
    timestamps and determine whether the strongest detector
    response is actually close to the true boundary.
    """

    grouped = {
        sid: grp
        for sid, grp in raw_df.groupby(
            "session_id",
            sort=False,
        )
    }

    rows = []

    for _, boundary in boundaries.iterrows():

        session_id = boundary["session_id"]
        gt_time = boundary["timestamp_ms"]

        session_events = grouped.get(session_id)

        if session_events is None:
            continue

        # Candidate raw event timestamps within ±5 sec.
        nearby = session_events[
            np.abs(
                session_events["timestamp_ms"]
                - gt_time
            ) <= 5000
        ]

        if nearby.empty:
            continue

        candidate_times = (
            nearby["timestamp_ms"]
            .drop_duplicates()
            .values
        )

        # Evaluate with the shortest detector window.
        # This gives the finest localization.
        window = 0.5

        candidate_rows = []

        for center in candidate_times:

            # Don't duplicate the same center.
            features = extract_features(
                session_events,
                center,
                window,
            )

            candidate_rows.append({
                "center_timestamp_ms": center,
                **features,
            })

        candidate_df = pd.DataFrame(
            candidate_rows
        )

        X = candidate_df[
            D_LEAN_FEATURES
        ].replace(
            [np.inf, -np.inf],
            np.nan,
        ).fillna(0)

        scores = model.predict_proba(
            X
        )[:, 1]

        best_idx = int(
            np.argmax(scores)
        )

        detected_time = candidate_df.iloc[
            best_idx
        ]["center_timestamp_ms"]

        error_sec = abs(
            detected_time - gt_time
        ) / 1000.0

        rows.append({
            "session_id": session_id,

            "gt_boundary_ms":
                gt_time,

            "detected_center_ms":
                detected_time,

            "localization_error_sec":
                error_sec,

            "max_boundary_score":
                float(scores[best_idx]),

            "within_0_5s":
                int(error_sec <= 0.5),

            "within_1s":
                int(error_sec <= 1.0),

            "within_2s":
                int(error_sec <= 2.0),
        })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DAY 2 — PHASE 5B")
    print("HARD-NEGATIVE + BOUNDARY LOCALIZATION TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load Phase 3 training data
    # --------------------------------------------------------

    print("\n[1/7] Loading Phase 3 features...")

    phase3_df = pd.read_csv(
        PHASE3_FEATURES
    )

    print(
        f"Phase 3 rows: {len(phase3_df):,}"
    )

    # --------------------------------------------------------
    # Train fixed detector
    # --------------------------------------------------------

    print("\n[2/7] Training fixed D_lean_full detector...")

    model = train_fixed_detector(
        phase3_df
    )

    print(
        "Detector trained using Phase 3 data."
    )

    # --------------------------------------------------------
    # Raw events
    # --------------------------------------------------------

    print("\n[3/7] Loading raw events...")

    raw_df = load_raw_events()

    print(
        f"Raw events: {len(raw_df):,}"
    )

    # --------------------------------------------------------
    # GT boundaries
    # --------------------------------------------------------

    print("\n[4/7] Loading GT boundaries...")

    boundaries = load_gt_boundaries()

    print(
        f"GT boundaries: {len(boundaries):,}"
    )

    # --------------------------------------------------------
    # Hard negatives
    # --------------------------------------------------------

    print("\n[5/7] Constructing hard negatives...")

    hard_df = make_hard_negatives(
        raw_df,
        boundaries,
    )

    print(
        f"Hard-negative centers: {len(hard_df):,}"
    )

    hard_features = build_hard_features(
        raw_df,
        hard_df,
    )

    print(
        f"Hard-negative feature rows: "
        f"{len(hard_features):,}"
    )

    # Save raw hard-negative dataset.
    hard_df.to_csv(
        OUTPUT_DIR / "hard_negative_centers.csv",
        index=False,
    )

    hard_features.to_csv(
        OUTPUT_DIR / "hard_negative_features.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Hard-negative evaluation
    # --------------------------------------------------------

    print("\n[6/7] Evaluating fixed detector...")

    hard_summary, scored_hard = evaluate_hard_negatives(
        model,
        hard_features,
    )

    hard_summary.to_csv(
        OUTPUT_DIR / "hard_negative_summary.csv",
        index=False,
    )

    scored_hard.to_csv(
        OUTPUT_DIR / "hard_negative_scored.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Boundary localization
    # --------------------------------------------------------

    print("\n[7/7] Testing boundary localization...")

    localization = boundary_localization(
        model,
        raw_df,
        boundaries,
    )

    localization.to_csv(
        OUTPUT_DIR / "boundary_localization.csv",
        index=False,
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    report = {
        "phase": "Day 2 Phase 5B",

        "raw_events":
            int(len(raw_df)),

        "gt_boundaries":
            int(len(boundaries)),

        "hard_negative_rows":
            int(len(hard_features)),

        "hard_negative_summary":
            hard_summary.to_dict(
                orient="records"
            ),

        "localization": {
            "evaluated_boundaries":
                int(len(localization)),

            "mean_error_sec":
                float(
                    localization[
                        "localization_error_sec"
                    ].mean()
                )
                if not localization.empty
                else None,

            "median_error_sec":
                float(
                    localization[
                        "localization_error_sec"
                    ].median()
                )
                if not localization.empty
                else None,

            "within_0_5s":
                float(
                    localization[
                        "within_0_5s"
                    ].mean()
                )
                if not localization.empty
                else None,

            "within_1s":
                float(
                    localization[
                        "within_1s"
                    ].mean()
                )
                if not localization.empty
                else None,

            "within_2s":
                float(
                    localization[
                        "within_2s"
                    ].mean()
                )
                if not localization.empty
                else None,
        },
    }

    with open(
        OUTPUT_DIR / "day2_phase5b_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PHASE 5B RESULTS")
    print("=" * 70)

    if not hard_summary.empty:

        print("\nHARD-NEGATIVE FALSE POSITIVE RATES")
        print("-" * 70)

        print(
            hard_summary[
                [
                    "window_sec",
                    "hard_negative_count",
                    "mean_score",
                    "p95_score",
                    "p99_score",
                    "false_positive_rate_at_0_5",
                    "false_positive_rate_at_0_7",
                    "false_positive_rate_at_0_9",
                ]
            ].to_string(
                index=False
            )
        )

    if not localization.empty:

        print("\nBOUNDARY LOCALIZATION")
        print("-" * 70)

        print(
            f"Boundaries evaluated : "
            f"{len(localization):,}"
        )

        print(
            f"Mean error           : "
            f"{localization['localization_error_sec'].mean():.3f}s"
        )

        print(
            f"Median error         : "
            f"{localization['localization_error_sec'].median():.3f}s"
        )

        print(
            f"Within ±0.5s         : "
            f"{localization['within_0_5s'].mean():.2%}"
        )

        print(
            f"Within ±1.0s         : "
            f"{localization['within_1s'].mean():.2%}"
        )

        print(
            f"Within ±2.0s         : "
            f"{localization['within_2s'].mean():.2%}"
        )

    print("\nOutputs written to:")
    print(OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("PHASE 5B COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()