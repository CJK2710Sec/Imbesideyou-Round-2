"""
Day 2 - Phase 5G
================

Boundary Confirmation / False-Positive Reduction

Input:
    Phase 5F continuous transition scores
    Phase 5F detected boundary candidates
    Raw event stream
    Ground-truth process boundaries

Goal:
    Determine whether a 5F candidate represents a genuine
    process transition or only a short-lived activity burst.

Confirmation evidence
---------------------
1. Transition score persistence
2. Pre/post event-structure change
3. Post-boundary state stability
4. Application/window state change

The script evaluates multiple confirmation thresholds instead
of choosing one arbitrarily.

Outputs
-------
candidate_confirmation_features.csv
candidate_confirmation_scores.csv
confirmation_threshold_evaluation.csv
confirmed_boundary_candidates.csv
confirmation_gt_matches.csv
confirmation_session_metrics.csv
"""


from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

DATASET_ROOT = Path(
    r"Dataset A\dataset_a"
)

PHASE5F_SCORE_FILE = Path(
    r"Outputs\Day 2\day2_boundary_candidate_detection"
    r"\continuous_transition_scores.csv"
)

PHASE5F_CANDIDATE_FILE = Path(
    r"Outputs\Day 2\day2_boundary_candidate_detection"
    r"\detected_boundary_candidates.csv"
)

GT_BOUNDARY_FILE = Path(
    r"Outputs\Day 2\day2_gt_boundary_analysis"
    r"\gt_switch_boundaries.csv"
)

OUTPUT_DIR = Path(
    r"Outputs\Day 2\day2_boundary_confirmation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

# Original 5F transition-score threshold.
# Used only for measuring persistence of the transition score.
SCORE_THRESHOLD = 0.70

# Candidate confirmation thresholds.
# We evaluate all of these rather than selecting one blindly.
CONFIRMATION_THRESHOLDS = [
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
]

# Windows around a candidate.
PRE_WINDOW_SEC = 2.0
POST_WINDOW_SEC = 2.0

# Smaller windows used for state comparison.
STATE_WINDOW_SEC = 1.0

# Minimum post-boundary stability duration.
STABILITY_WINDOW_SEC = 1.0

# GT matching tolerance.
MATCH_TOLERANCE_SEC = 2.0


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def normalize_value(value):
    """
    Convert categorical values to stable strings.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (dict, list)
    ):
        try:
            return json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False
            )
        except Exception:
            return str(value)

    return str(value)


def jaccard(a, b):
    """
    Jaccard similarity.
    """

    if not a and not b:
        return 1.0

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


def safe_entropy(values):
    """
    Shannon entropy.
    """

    values = [
        x for x in values
        if x is not None
    ]

    if not values:
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
            p * np.log2(
                p + 1e-12
            )
        )

    return float(entropy)


# ============================================================
# RAW EVENT EXTRACTION
# ============================================================

def load_session_events(
    session_dir
):
    """
    Load raw events for one session.

    Events are loaded once and sorted by timestamp.
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

                    timestamp = event.get(
                        "timestamp_ms"
                    )

                    if timestamp is None:
                        continue

                    context = (
                        event.get(
                            "context"
                        )
                        or {}
                    )

                    active_app = (
                        context.get(
                            "active_app"
                        )
                        or {}
                    )

                    records.append(
                        {
                            "timestamp_ms":
                                int(timestamp),

                            "event_type":
                                normalize_value(
                                    event.get(
                                        "event_type"
                                    )
                                ),

                            "layer":
                                normalize_value(
                                    event.get(
                                        "layer"
                                    )
                                ),

                            "app_name":
                                normalize_value(
                                    active_app.get(
                                        "app_name"
                                    )
                                ),

                            "process_name":
                                normalize_value(
                                    active_app.get(
                                        "process_name"
                                    )
                                ),

                            "window_title":
                                normalize_value(
                                    active_app.get(
                                        "window_title"
                                    )
                                ),
                        }
                    )

        except Exception as exc:

            print(
                f"WARNING: {event_file}: "
                f"{exc}"
            )

    if not records:
        return None

    records.sort(
        key=lambda x:
            x["timestamp_ms"]
    )

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

        "process_name": np.asarray(
            [
                x["process_name"]
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
# FAST TEMPORAL INDEXING
# ============================================================

def get_bounds(
    timestamps,
    center_ms,
    half_window_ms
):
    """
    Find temporal window using binary search.
    """

    left = np.searchsorted(
        timestamps,
        center_ms - half_window_ms,
        side="left"
    )

    right = np.searchsorted(
        timestamps,
        center_ms + half_window_ms,
        side="right"
    )

    return left, right


def get_region(
    timestamps,
    center_ms,
    start_offset_sec,
    end_offset_sec
):
    """
    Return [center+start, center+end].
    """

    start_ms = (
        center_ms
        + int(
            start_offset_sec
            * 1000
        )
    )

    end_ms = (
        center_ms
        + int(
            end_offset_sec
            * 1000
        )
    )

    left = np.searchsorted(
        timestamps,
        start_ms,
        side="left"
    )

    right = np.searchsorted(
        timestamps,
        end_ms,
        side="right"
    )

    return left, right


# ============================================================
# STATE EXTRACTION
# ============================================================

def extract_state(
    session,
    center_ms,
    start_offset_sec,
    end_offset_sec
):
    """
    Extract categorical state from a temporal interval.
    """

    timestamps = session[
        "timestamps"
    ]

    left, right = get_region(
        timestamps,
        center_ms,
        start_offset_sec,
        end_offset_sec
    )

    event_types = session[
        "event_type"
    ][left:right]

    layers = session[
        "layer"
    ][left:right]

    apps = session[
        "app_name"
    ][left:right]

    processes = session[
        "process_name"
    ][left:right]

    titles = session[
        "window_title"
    ][left:right]

    return {
        "event_types": {
            x for x in event_types
            if x is not None
        },

        "layers": {
            x for x in layers
            if x is not None
        },

        "apps": {
            x for x in apps
            if x is not None
        },

        "processes": {
            x for x in processes
            if x is not None
        },

        "titles": {
            x for x in titles
            if x is not None
        },

        "event_list":
            list(event_types),

        "count":
            len(event_types),
    }


# ============================================================
# STATE CHANGE FEATURES
# ============================================================

def calculate_state_change(
    pre,
    post
):
    """
    Compare pre-boundary and post-boundary states.
    """

    event_j = jaccard(
        pre["event_types"],
        post["event_types"]
    )

    layer_j = jaccard(
        pre["layers"],
        post["layers"]
    )

    app_j = jaccard(
        pre["apps"],
        post["apps"]
    )

    process_j = jaccard(
        pre["processes"],
        post["processes"]
    )

    title_j = jaccard(
        pre["titles"],
        post["titles"]
    )

    # --------------------------------------------------------
    # Convert similarity into change evidence.
    # --------------------------------------------------------

    event_change = (
        1.0 - event_j
    )

    layer_change = (
        1.0 - layer_j
    )

    app_change = (
        1.0 - app_j
    )

    process_change = (
        1.0 - process_j
    )

    title_change = (
        1.0 - title_j
    )

    # --------------------------------------------------------
    # Structure score.
    #
    # Event/layer changes are more trustworthy than raw
    # application changes because we already found that
    # app identity alone is insufficient.
    # --------------------------------------------------------

    structure_score = (
        0.35 * event_change
        +
        0.25 * layer_change
        +
        0.20 * title_change
        +
        0.10 * app_change
        +
        0.10 * process_change
    )

    return {
        "pre_event_count":
            pre["count"],

        "post_event_count":
            post["count"],

        "event_jaccard":
            event_j,

        "layer_jaccard":
            layer_j,

        "app_jaccard":
            app_j,

        "process_jaccard":
            process_j,

        "title_jaccard":
            title_j,

        "event_change":
            event_change,

        "layer_change":
            layer_change,

        "app_change":
            app_change,

        "process_change":
            process_change,

        "title_change":
            title_change,

        "structure_score":
            structure_score,
    }


# ============================================================
# POST-BOUNDARY STABILITY
# ============================================================

def calculate_post_stability(
    session,
    center_ms
):
    """
    Measure whether the post-boundary state persists.

    We divide the post-boundary period into two intervals:

        [0, 1 sec]
        [1, 2 sec]

    If their event/layer/app/title structure is similar,
    the new state is considered more stable.

    This is intended to reject short-lived bursts.
    """

    timestamps = session[
        "timestamps"
    ]

    # --------------------------------------------------------
    # First post-boundary state
    # --------------------------------------------------------

    first = extract_state(
        session,
        center_ms,
        0.0,
        STABILITY_WINDOW_SEC
    )

    # --------------------------------------------------------
    # Later post-boundary state
    # --------------------------------------------------------

    second = extract_state(
        session,
        center_ms,
        STABILITY_WINDOW_SEC,
        2.0
    )

    # --------------------------------------------------------
    # Similarity between post states
    # --------------------------------------------------------

    event_similarity = jaccard(
        first["event_types"],
        second["event_types"]
    )

    layer_similarity = jaccard(
        first["layers"],
        second["layers"]
    )

    app_similarity = jaccard(
        first["apps"],
        second["apps"]
    )

    title_similarity = jaccard(
        first["titles"],
        second["titles"]
    )

    # --------------------------------------------------------
    # Stability score
    # --------------------------------------------------------

    stability_score = (
        0.35 * event_similarity
        +
        0.25 * layer_similarity
        +
        0.20 * app_similarity
        +
        0.20 * title_similarity
    )

    return {
        "post_event_similarity":
            event_similarity,

        "post_layer_similarity":
            layer_similarity,

        "post_app_similarity":
            app_similarity,

        "post_title_similarity":
            title_similarity,

        "post_stability_score":
            stability_score,
    }


# ============================================================
# PERSISTENCE FEATURES
# ============================================================

def calculate_score_persistence(
    score_df,
    session_id,
    candidate_timestamp_ms
):
    """
    Extract transition-score persistence around candidate.

    Uses the already generated 5F continuous score file,
    avoiding expensive raw-event recalculation.
    """

    session_scores = score_df[
        score_df[
            "session_id"
        ]
        == session_id
    ]

    if session_scores.empty:

        return {
            "score_at_candidate": np.nan,
            "score_pre_mean": np.nan,
            "score_post_mean": np.nan,
            "score_pre_max": np.nan,
            "score_post_max": np.nan,
            "score_persistence": 0.0,
        }

    timestamps = session_scores[
        "timestamp_ms"
    ].to_numpy(
        dtype=np.int64
    )

    scores = session_scores[
        "mean_all_score"
    ].to_numpy(
        dtype=float
    )

    # --------------------------------------------------------
    # Candidate position
    # --------------------------------------------------------

    idx = np.searchsorted(
        timestamps,
        candidate_timestamp_ms
    )

    idx = min(
        max(idx, 0),
        len(timestamps) - 1
    )

    score_at_candidate = float(
        scores[idx]
    )

    # --------------------------------------------------------
    # ±2 sec region
    # --------------------------------------------------------

    pre_mask = (
        (
            timestamps
            >= candidate_timestamp_ms
            - 2000
        )
        &
        (
            timestamps
            < candidate_timestamp_ms
        )
    )

    post_mask = (
        (
            timestamps
            > candidate_timestamp_ms
        )
        &
        (
            timestamps
            <= candidate_timestamp_ms
            + 2000
        )
    )

    pre_scores = scores[
        pre_mask
    ]

    post_scores = scores[
        post_mask
    ]

    # --------------------------------------------------------
    # Fraction above threshold
    # --------------------------------------------------------

    if len(post_scores):

        persistence = float(
            np.mean(
                post_scores
                >= SCORE_THRESHOLD
            )
        )

    else:

        persistence = 0.0

    return {
        "score_at_candidate":
            score_at_candidate,

        "score_pre_mean":
            float(
                np.mean(pre_scores)
            )
            if len(pre_scores)
            else 0.0,

        "score_post_mean":
            float(
                np.mean(post_scores)
            )
            if len(post_scores)
            else 0.0,

        "score_pre_max":
            float(
                np.max(pre_scores)
            )
            if len(pre_scores)
            else 0.0,

        "score_post_max":
            float(
                np.max(post_scores)
            )
            if len(post_scores)
            else 0.0,

        "score_persistence":
            persistence,
    }


# ============================================================
# CANDIDATE CONFIRMATION FEATURES
# ============================================================

def build_confirmation_features(
    candidates,
    score_df,
    sessions
):
    """
    Calculate confirmation evidence for every 5F candidate.
    """

    rows = []

    total = len(
        candidates
    )

    for number, (_, candidate) in enumerate(
        candidates.iterrows(),
        start=1
    ):

        if (
            number == 1
            or number % 500 == 0
            or number == total
        ):

            print(
                f"    candidates analyzed: "
                f"{number:,}/{total:,}",
                flush=True
            )

        session_id = candidate[
            "session_id"
        ]

        timestamp_ms = int(
            candidate[
                "candidate_timestamp_ms"
            ]
        )

        session = sessions.get(
            session_id
        )

        if session is None:
            continue

        # ----------------------------------------------------
        # Pre/post state
        # ----------------------------------------------------

        pre_state = extract_state(
            session,
            timestamp_ms,
            -STATE_WINDOW_SEC,
            0.0
        )

        post_state = extract_state(
            session,
            timestamp_ms,
            0.0,
            STATE_WINDOW_SEC
        )

        state_features = (
            calculate_state_change(
                pre_state,
                post_state
            )
        )

        # ----------------------------------------------------
        # Post-boundary stability
        # ----------------------------------------------------

        stability = (
            calculate_post_stability(
                session,
                timestamp_ms
            )
        )

        # ----------------------------------------------------
        # Transition-score persistence
        # ----------------------------------------------------

        score_features = (
            calculate_score_persistence(
                score_df,
                session_id,
                timestamp_ms
            )
        )

        # ----------------------------------------------------
        # Additional score evidence
        # ----------------------------------------------------

        original_score = float(
            candidate[
                "boundary_score"
            ]
        )

        # ----------------------------------------------------
        # Confirmation score
        #
        # This is deliberately interpretable rather than
        # another ML model.
        # ----------------------------------------------------

        score_evidence = min(
            max(
                original_score,
                0.0
            ),
            1.0
        )

        persistence_evidence = (
            score_features[
                "score_persistence"
            ]
        )

        structure_evidence = (
            state_features[
                "structure_score"
            ]
        )

        stability_evidence = (
            stability[
                "post_stability_score"
            ]
        )

        confirmation_score = (
            0.35
            * score_evidence
            +
            0.25
            * persistence_evidence
            +
            0.25
            * structure_evidence
            +
            0.15
            * stability_evidence
        )

        row = {
            **candidate.to_dict(),

            **state_features,

            **stability,

            **score_features,

            "score_evidence":
                score_evidence,

            "persistence_evidence":
                persistence_evidence,

            "structure_evidence":
                structure_evidence,

            "stability_evidence":
                stability_evidence,

            "confirmation_score":
                confirmation_score,
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# STRICT ONE-TO-ONE MATCHING
# ============================================================

def match_candidates(
    candidates,
    gt
):
    """
    Strict one-to-one matching.

    Each GT boundary can be matched only once.
    Each candidate can be matched only once.
    """

    tolerance_ms = int(
        MATCH_TOLERANCE_SEC
        * 1000
    )

    if candidates.empty:

        return pd.DataFrame()

    rows = []

    for session_id in sorted(
        candidates[
            "session_id"
        ].unique()
    ):

        c = candidates[
            candidates[
                "session_id"
            ]
            == session_id
        ].sort_values(
            "candidate_timestamp_ms"
        ).reset_index(
            drop=True
        )

        g = gt[
            gt[
                "session_id"
            ]
            == session_id
        ].sort_values(
            "timestamp_ms"
        ).reset_index(
            drop=True
        )

        gt_times = g[
            "timestamp_ms"
        ].to_numpy(
            dtype=np.int64
        )

        # ----------------------------------------------------
        # Used GT indices
        # ----------------------------------------------------

        used_gt = set()

        for _, candidate in c.iterrows():

            candidate_ts = int(
                candidate[
                    "candidate_timestamp_ms"
                ]
            )

            insertion = np.searchsorted(
                gt_times,
                candidate_ts
            )

            possible = []

            # Search a few nearby GT boundaries.
            # This is safer than only looking at one index.
            for offset in range(
                -3,
                4
            ):

                index = (
                    insertion
                    + offset
                )

                if (
                    0
                    <= index
                    < len(gt_times)
                ):
                    possible.append(
                        index
                    )

            best_index = None
            best_error = None

            for gt_index in possible:

                if gt_index in used_gt:
                    continue

                error = abs(
                    candidate_ts
                    - int(
                        gt_times[
                            gt_index
                        ]
                    )
                )

                if (
                    best_error is None
                    or error < best_error
                ):

                    best_error = error
                    best_index = gt_index

            if (
                best_index is not None
                and best_error <= tolerance_ms
            ):

                used_gt.add(
                    best_index
                )

                gt_ts = int(
                    gt_times[
                        best_index
                    ]
                )

                rows.append(
                    {
                        **candidate.to_dict(),

                        "matched":
                            True,

                        "gt_timestamp_ms":
                            gt_ts,

                        "error_ms":
                            best_error,

                        "error_sec":
                            best_error / 1000.0,
                    }
                )

            else:

                rows.append(
                    {
                        **candidate.to_dict(),

                        "matched":
                            False,

                        "gt_timestamp_ms":
                            np.nan,

                        "error_ms":
                            np.nan,

                        "error_sec":
                            np.nan,
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

def evaluate_threshold(
    confirmation_df,
    gt,
    threshold
):
    """
    Evaluate one confirmation threshold.
    """

    confirmed = (
        confirmation_df[
            confirmation_df[
                "confirmation_score"
            ]
            >= threshold
        ]
        .copy()
    )

    matches = match_candidates(
        confirmed,
        gt
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

    gt_count = len(
        gt
    )

    candidate_count = len(
        confirmed
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
        if candidate_count
        else 0.0
    )

    recall = (
        tp / gt_count
        if gt_count
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
        and "error_sec" in matches
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
        "confirmation_threshold":
            threshold,

        "gt_boundaries":
            gt_count,

        "candidates_after_confirmation":
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
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "DAY 2 - PHASE 5G"
    )

    print(
        "BOUNDARY CONFIRMATION"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # Load 5F candidates
    # ========================================================

    print(
        "\nLoading Phase 5F candidates..."
    )

    candidates = pd.read_csv(
        PHASE5F_CANDIDATE_FILE
    )

    print(
        f"5F candidates: "
        f"{len(candidates):,}"
    )

    # ========================================================
    # Load continuous scores
    # ========================================================

    print(
        "\nLoading continuous scores..."
    )

    score_df = pd.read_csv(
        PHASE5F_SCORE_FILE
    )

    print(
        f"Score rows: "
        f"{len(score_df):,}"
    )

    # ========================================================
    # Load GT
    # ========================================================

    print(
        "\nLoading GT boundaries..."
    )

    gt = pd.read_csv(
        GT_BOUNDARY_FILE
    )

    gt["timestamp_ms"] = (
        pd.to_numeric(
            gt["timestamp_ms"],
            errors="coerce"
        )
    )

    gt = gt.dropna(
        subset=[
            "timestamp_ms"
        ]
    )

    gt["timestamp_ms"] = (
        gt[
            "timestamp_ms"
        ]
        .astype(np.int64)
    )

    print(
        f"GT boundaries: "
        f"{len(gt):,}"
    )

    # ========================================================
    # Load raw sessions
    # ========================================================

    print(
        "\nLoading raw session data..."
    )

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

    sessions = {}

    for index, session_dir in enumerate(
        session_dirs,
        start=1
    ):

        session_id = (
            session_dir.name
        )

        print(
            f"  [{index}/{len(session_dirs)}] "
            f"{session_id}",
            flush=True
        )

        sessions[
            session_id
        ] = load_session_events(
            session_dir
        )

    # ========================================================
    # Build confirmation features
    # ========================================================

    print(
        "\nCalculating confirmation "
        "features..."
    )

    confirmation_df = (
        build_confirmation_features(
            candidates,
            score_df,
            sessions
        )
    )

    feature_file = (
        OUTPUT_DIR
        / "candidate_confirmation_features.csv"
    )

    confirmation_df.to_csv(
        feature_file,
        index=False
    )

    print(
        f"\nSaved confirmation features:"
        f"\n{feature_file}"
    )

    # ========================================================
    # Evaluate thresholds
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVALUATING CONFIRMATION THRESHOLDS"
    )

    print(
        "=" * 70
    )

    threshold_rows = []

    best_f1 = -1
    best_threshold = None
    best_matches = None
    best_confirmed = None

    for threshold in (
        CONFIRMATION_THRESHOLDS
    ):

        print(
            f"\nThreshold: "
            f"{threshold:.2f}"
        )

        confirmed = (
            confirmation_df[
                confirmation_df[
                    "confirmation_score"
                ]
                >= threshold
            ]
            .copy()
        )

        matches = match_candidates(
            confirmed,
            gt
        )

        metrics = evaluate_threshold(
            confirmation_df,
            gt,
            threshold
        )

        threshold_rows.append(
            metrics
        )

        print(
            f"  Candidates : "
            f"{metrics['candidates_after_confirmation']:,}"
        )

        print(
            f"  TP         : "
            f"{metrics['true_positives']:,}"
        )

        print(
            f"  FP         : "
            f"{metrics['false_positives']:,}"
        )

        print(
            f"  FN         : "
            f"{metrics['false_negatives']:,}"
        )

        print(
            f"  Precision  : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"  Recall     : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"  F1         : "
            f"{metrics['f1']:.4f}"
        )

        if (
            metrics["f1"]
            > best_f1
        ):

            best_f1 = (
                metrics["f1"]
            )

            best_threshold = (
                threshold
            )

            best_matches = matches

            best_confirmed = (
                confirmed
            )

    # ========================================================
    # Save threshold results
    # ========================================================

    threshold_df = pd.DataFrame(
        threshold_rows
    )

    threshold_file = (
        OUTPUT_DIR
        / "confirmation_threshold_evaluation.csv"
    )

    threshold_df.to_csv(
        threshold_file,
        index=False
    )

    # ========================================================
    # Save best confirmed candidates
    # ========================================================

    confirmed_file = (
        OUTPUT_DIR
        / "confirmed_boundary_candidates.csv"
    )

    best_confirmed.to_csv(
        confirmed_file,
        index=False
    )

    # ========================================================
    # Save best GT matches
    # ========================================================

    match_file = (
        OUTPUT_DIR
        / "confirmation_gt_matches.csv"
    )

    best_matches.to_csv(
        match_file,
        index=False
    )

    # ========================================================
    # Session-level metrics
    # ========================================================

    session_rows = []

    for session_id in sorted(
        gt[
            "session_id"
        ].unique()
    ):

        gt_session = gt[
            gt[
                "session_id"
            ]
            == session_id
        ]

        candidate_session = (
            best_confirmed[
                best_confirmed[
                    "session_id"
                ]
                == session_id
            ]
        )

        match_session = (
            best_matches[
                best_matches[
                    "session_id"
                ]
                == session_id
            ]
        )

        tp = int(
            match_session[
                "matched"
            ]
            .fillna(False)
            .sum()
        )

        gt_count = len(
            gt_session
        )

        candidate_count = len(
            candidate_session
        )

        session_rows.append(
            {
                "session_id":
                    session_id,

                "gt_boundaries":
                    gt_count,

                "confirmed_candidates":
                    candidate_count,

                "true_positives":
                    tp,

                "false_positives":
                    (
                        candidate_count
                        - tp
                    ),

                "false_negatives":
                    (
                        gt_count
                        - tp
                    ),

                "recall":
                    (
                        tp / gt_count
                        if gt_count
                        else 0.0
                    ),
            }
        )

    session_df = pd.DataFrame(
        session_rows
    )

    session_file = (
        OUTPUT_DIR
        / "confirmation_session_metrics.csv"
    )

    session_df.to_csv(
        session_file,
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    best_metrics = threshold_df[
        threshold_df[
            "confirmation_threshold"
        ]
        == best_threshold
    ].iloc[0]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PHASE 5G COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Original 5F candidates : "
        f"{len(candidates):,}"
    )

    print(
        f"Best threshold         : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Confirmed candidates   : "
        f"{int(best_metrics['candidates_after_confirmation']):,}"
    )

    print(
        f"True positives         : "
        f"{int(best_metrics['true_positives']):,}"
    )

    print(
        f"False positives        : "
        f"{int(best_metrics['false_positives']):,}"
    )

    print(
        f"False negatives        : "
        f"{int(best_metrics['false_negatives']):,}"
    )

    print(
        f"Precision              : "
        f"{best_metrics['precision']:.4f}"
    )

    print(
        f"Recall                 : "
        f"{best_metrics['recall']:.4f}"
    )

    print(
        f"F1                     : "
        f"{best_metrics['f1']:.4f}"
    )

    print(
        f"Mean localization      : "
        f"{best_metrics['mean_localization_error_sec']:.3f}s"
    )

    print(
        f"Median localization    : "
        f"{best_metrics['median_localization_error_sec']:.3f}s"
    )

    print(
        f"P90 localization       : "
        f"{best_metrics['p90_localization_error_sec']:.3f}s"
    )

    print(
        "\nOutputs:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()