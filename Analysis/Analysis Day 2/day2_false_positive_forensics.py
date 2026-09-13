from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(r"Dataset A\dataset_a")

PHASE3_FEATURES = Path(
    r"Outputs\Day 2\day2_boundary_vs_normal\boundary_vs_normal_features.csv"
)

HARD_NEGATIVE_FEATURES = Path(
    r"Outputs\Day 2\day2_boundary_hard_negative\hard_negative_features.csv"
)

OUTPUT_DIR = Path(
    r"Outputs\Day 2\day2_false_positive_forensics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TOP_N_PER_WINDOW = 50

CONTEXT_SECONDS = 2.0
GT_CONTEXT_SECONDS = 3.0


# ============================================================
# D_LEAN_FULL
# ============================================================

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


# ============================================================
# LOAD RAW EVENTS
# ============================================================

def load_raw_events():

    rows = []

    session_dirs = sorted(
        p
        for p in DATASET_ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith("ses_")
    )

    print(
        f"Sessions found: {len(session_dirs)}"
    )

    for session_dir in session_dirs:

        for event_file in session_dir.rglob(
            "events.jsonl"
        ):

            with open(
                event_file,
                "r",
                encoding="utf-8",
            ) as f:

                for line in f:

                    try:
                        event = json.loads(line)
                    except Exception:
                        continue

                    if not isinstance(
                        event,
                        dict,
                    ):
                        continue

                    ts = event.get(
                        "timestamp_ms"
                    )

                    if ts is None:
                        continue

                    try:
                        ts = float(ts)
                    except Exception:
                        continue

                    context = (
                        event.get("context")
                        or {}
                    )

                    if not isinstance(
                        context,
                        dict,
                    ):
                        context = {}

                    active_app = (
                        context.get(
                            "active_app"
                        )
                        or {}
                    )

                    if not isinstance(
                        active_app,
                        dict,
                    ):
                        active_app = {}

                    rows.append({
                        "session_id":
                            event.get(
                                "session_id",
                                session_dir.name,
                            ),

                        "timestamp_ms":
                            ts,

                        "event_type":
                            event.get(
                                "event_type"
                            ),

                        "layer":
                            event.get(
                                "layer"
                            ),

                        "app_name":
                            active_app.get(
                                "app_name"
                            ),

                        "process_name":
                            active_app.get(
                                "process_name"
                            ),

                        "window_title":
                            active_app.get(
                                "window_title"
                            ),
                    })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No raw events found."
        )

    return (
        df
        .sort_values(
            [
                "session_id",
                "timestamp_ms",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# LOAD GT EVENTS
# ============================================================

def load_gt_events():

    rows = []

    session_dirs = sorted(
        p
        for p in DATASET_ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith("ses_")
    )

    for session_dir in session_dirs:

        for gt_file in session_dir.rglob(
            "gt.jsonl"
        ):

            with open(
                gt_file,
                "r",
                encoding="utf-8",
            ) as f:

                for line in f:

                    try:
                        event = json.loads(line)
                    except Exception:
                        continue

                    if not isinstance(
                        event,
                        dict,
                    ):
                        continue

                    ts = event.get(
                        "ts_utc"
                    )

                    if ts is None:
                        continue

                    try:
                        timestamp_ms = (
                            pd.to_datetime(
                                ts,
                                utc=True,
                            ).timestamp()
                            * 1000
                        )
                    except Exception:
                        continue

                    rows.append({
                        "session_id":
                            session_dir.name,

                        "timestamp_ms":
                            timestamp_ms,

                        "event_type":
                            event.get(
                                "event_type"
                            ),

                        "current_process":
                            event.get(
                                "current_process"
                            ),

                        "process_variant":
                            event.get(
                                "process_variant"
                            ),

                        "from_process":
                            event.get(
                                "from"
                            ),

                        "to_process":
                            event.get(
                                "to"
                            ),

                        "process_code":
                            event.get(
                                "process_code"
                            ),

                        "case_id":
                            event.get(
                                "case_id"
                            ),

                        "task_id":
                            event.get(
                                "task_id"
                            ),

                        "action":
                            event.get(
                                "action"
                            ),
                    })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No GT events found."
        )

    return (
        df
        .sort_values(
            [
                "session_id",
                "timestamp_ms",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# TRAIN FIXED DETECTOR
# ============================================================

def train_detector():

    if not PHASE3_FEATURES.exists():

        raise FileNotFoundError(
            f"Phase 3 feature file not found:\n"
            f"{PHASE3_FEATURES.resolve()}"
        )

    df = pd.read_csv(
        PHASE3_FEATURES
    )

    missing = [
        feature
        for feature in D_LEAN_FEATURES
        if feature not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing Phase 3 features:\n"
            + "\n".join(missing)
        )

    X = (
        df[D_LEAN_FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    y = df["label"].astype(int)

    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
        ),
    ])

    model.fit(
        X,
        y,
    )

    return model


# ============================================================
# SCORE HARD NEGATIVES
# ============================================================

def score_hard_negatives(
    model
):

    if not HARD_NEGATIVE_FEATURES.exists():

        raise FileNotFoundError(
            f"Hard-negative feature file not found:\n"
            f"{HARD_NEGATIVE_FEATURES.resolve()}"
        )

    hard = pd.read_csv(
        HARD_NEGATIVE_FEATURES
    )

    print(
        f"Hard-negative feature rows: "
        f"{len(hard):,}"
    )

    missing = [
        feature
        for feature in D_LEAN_FEATURES
        if feature not in hard.columns
    ]

    if missing:

        raise RuntimeError(
            "Hard-negative file is missing:\n"
            + "\n".join(missing)
        )

    X = (
        hard[D_LEAN_FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    hard["boundary_score"] = (
        model.predict_proba(X)[:, 1]
    )

    return hard


# ============================================================
# UNIQUE JOIN
# ============================================================

def unique_join(
    values,
    max_items=15,
):

    result = []

    for value in values:

        if value is None:
            continue

        try:
            if pd.isna(value):
                continue
        except Exception:
            pass

        value = str(value).strip()

        if not value:
            continue

        if value not in result:
            result.append(value)

    if len(result) > max_items:

        return (
            " | ".join(
                result[:max_items]
            )
            + f" | ... (+{len(result)-max_items})"
        )

    return " | ".join(result)


# ============================================================
# ANALYZE CASE
# ============================================================

def analyze_case(
    case,
    raw_events,
    gt_events,
):

    session_id = str(
        case["session_id"]
    )

    center = float(
        case["center_timestamp_ms"]
    )

    start = (
        center
        - CONTEXT_SECONDS * 1000
    )

    end = (
        center
        + CONTEXT_SECONDS * 1000
    )

    local = raw_events[
        (raw_events["session_id"].astype(str) == session_id)
        &
        (raw_events["timestamp_ms"] >= start)
        &
        (raw_events["timestamp_ms"] <= end)
    ].copy()

    event_types = (
        local["event_type"]
        .dropna()
        .astype(str)
        .tolist()
    )

    layers = (
        local["layer"]
        .dropna()
        .astype(str)
        .tolist()
    )

    apps = (
        local["app_name"]
        .dropna()
        .astype(str)
        .tolist()
    )

    processes = (
        local["process_name"]
        .dropna()
        .astype(str)
        .tolist()
    )

    titles = (
        local["window_title"]
        .dropna()
        .astype(str)
        .tolist()
    )

    event_counts = (
        local["event_type"]
        .dropna()
        .value_counts()
        .to_dict()
    )

    event_count_string = " | ".join(
        f"{k}:{v}"
        for k, v in sorted(
            event_counts.items(),
            key=lambda x: -x[1],
        )
    )

    # --------------------------------------------------------
    # Nearby GT events
    # --------------------------------------------------------

    gt_start = (
        center
        - GT_CONTEXT_SECONDS * 1000
    )

    gt_end = (
        center
        + GT_CONTEXT_SECONDS * 1000
    )

    nearby_gt = gt_events[
        (gt_events["session_id"].astype(str) == session_id)
        &
        (gt_events["timestamp_ms"] >= gt_start)
        &
        (gt_events["timestamp_ms"] <= gt_end)
    ].copy()

    gt_descriptions = []

    for _, gt in nearby_gt.iterrows():

        distance = (
            gt["timestamp_ms"]
            - center
        ) / 1000.0

        description = (
            f"{distance:+.3f}s:"
            f"{gt['event_type']}"
        )

        if pd.notna(
            gt.get("current_process")
        ):
            description += (
                f":process="
                f"{gt['current_process']}"
            )

        if pd.notna(
            gt.get("from_process")
        ):
            description += (
                f":from="
                f"{gt['from_process']}"
            )

        if pd.notna(
            gt.get("to_process")
        ):
            description += (
                f":to="
                f"{gt['to_process']}"
            )

        if pd.notna(
            gt.get("process_code")
        ):
            description += (
                f":code="
                f"{gt['process_code']}"
            )

        if pd.notna(
            gt.get("task_id")
        ):
            description += (
                f":task="
                f"{gt['task_id']}"
            )

        gt_descriptions.append(
            description
        )

    # --------------------------------------------------------
    # Nearest GT event
    # --------------------------------------------------------

    if nearby_gt.empty:

        nearest_gt_distance = np.nan
        nearest_gt_type = ""

    else:

        distances = (
            nearby_gt["timestamp_ms"]
            - center
        ).abs()

        idx = distances.idxmin()

        nearest_gt_distance = (
            nearby_gt.loc[
                idx,
                "timestamp_ms",
            ]
            - center
        ) / 1000.0

        nearest_gt_type = str(
            nearby_gt.loc[
                idx,
                "event_type",
            ]
        )

    # --------------------------------------------------------
    # Categories
    # --------------------------------------------------------

    app_switch_count = int(
        (
            local["event_type"]
            == "app_switch"
        ).sum()
    )

    browser_event_count = int(
        local["event_type"]
        .astype(str)
        .str.startswith(
            "browser_"
        )
        .sum()
    )

    input_event_count = int(
        local["event_type"].isin(
            [
                "keystroke",
                "shortcut",
                "text_input_complete",
                "clipboard_change",
            ]
        ).sum()
    )

    mouse_event_count = int(
        local["event_type"]
        .astype(str)
        .str.startswith(
            "mouse_"
        )
        .sum()
    )

    screenshot_event_count = int(
        (
            local["event_type"]
            == "screenshot_smart"
        ).sum()
    )

    return {
        "session_id":
            session_id,

        "center_timestamp_ms":
            center,

        "window_sec":
            case["window_sec"],

        "boundary_score":
            float(
                case["boundary_score"]
            ),

        "distance_from_gt_sec":
            float(
                case["distance_from_gt_sec"]
            ),

        "nearest_gt_event_distance_sec":
            nearest_gt_distance,

        "nearest_gt_event_type":
            nearest_gt_type,

        "local_event_count":
            len(local),

        "unique_event_types":
            local["event_type"].nunique(),

        "unique_layers":
            local["layer"].nunique(),

        "unique_apps":
            local["app_name"].nunique(),

        "unique_process_names":
            local["process_name"].nunique(),

        "unique_window_titles":
            local["window_title"].nunique(),

        "app_switch_count":
            app_switch_count,

        "browser_event_count":
            browser_event_count,

        "input_event_count":
            input_event_count,

        "mouse_event_count":
            mouse_event_count,

        "screenshot_event_count":
            screenshot_event_count,

        "event_type_counts":
            event_count_string,

        "event_types":
            unique_join(event_types),

        "layers":
            unique_join(layers),

        "apps":
            unique_join(apps),

        "process_names":
            unique_join(processes),

        "window_titles":
            unique_join(
                titles,
                max_items=8,
            ),

        "nearby_gt_events":
            " | ".join(
                gt_descriptions
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DAY 2 — PHASE 5C")
    print("FALSE-POSITIVE FORENSICS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Train fixed detector
    # --------------------------------------------------------

    print(
        "\n[1/5] Training fixed D_lean_full detector..."
    )

    model = train_detector()

    print(
        "Detector trained from Phase 3 data."
    )

    # --------------------------------------------------------
    # 2. Score hard negatives
    # --------------------------------------------------------

    print(
        "\n[2/5] Scoring hard negatives..."
    )

    hard = score_hard_negatives(
        model
    )

    print(
        f"Scored hard negatives: "
        f"{len(hard):,}"
    )

    # Save scored file for future phases.
    hard.to_csv(
        OUTPUT_DIR
        / "hard_negative_scored_forensics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3. Select top cases
    # --------------------------------------------------------

    print(
        "\n[3/5] Selecting highest-scoring cases..."
    )

    hard = hard.sort_values(
        "boundary_score",
        ascending=False,
    )

    selected = (
        hard
        .groupby(
            "window_sec",
            group_keys=False,
        )
        .head(
            TOP_N_PER_WINDOW
        )
        .copy()
    )

    print(
        f"Selected cases: "
        f"{len(selected):,}"
    )

    selected.to_csv(
        OUTPUT_DIR
        / "top_false_positive_candidates.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 4. Load context
    # --------------------------------------------------------

    print(
        "\n[4/5] Loading raw + GT context..."
    )

    raw_events = load_raw_events()

    gt_events = load_gt_events()

    # --------------------------------------------------------
    # 5. Forensics
    # --------------------------------------------------------

    print(
        "\n[5/5] Performing false-positive forensics..."
    )

    results = []

    for i, (_, case) in enumerate(
        selected.iterrows(),
        start=1,
    ):

        if i % 25 == 0:

            print(
                f"  analyzed "
                f"{i:,}/{len(selected):,}"
            )

        results.append(
            analyze_case(
                case,
                raw_events,
                gt_events,
            )
        )

    forensic_df = pd.DataFrame(
        results
    )

    forensic_df = forensic_df.sort_values(
        [
            "window_sec",
            "boundary_score",
        ],
        ascending=[
            True,
            False,
        ],
    )

    forensic_df.to_csv(
        OUTPUT_DIR
        / "false_positive_forensics.csv",
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_rows = []

    for window, group in forensic_df.groupby(
        "window_sec"
    ):

        summary_rows.append({
            "window_sec":
                window,

            "cases":
                len(group),

            "mean_score":
                group[
                    "boundary_score"
                ].mean(),

            "median_score":
                group[
                    "boundary_score"
                ].median(),

            "mean_local_event_count":
                group[
                    "local_event_count"
                ].mean(),

            "mean_app_switch_count":
                group[
                    "app_switch_count"
                ].mean(),

            "mean_input_event_count":
                group[
                    "input_event_count"
                ].mean(),

            "mean_mouse_event_count":
                group[
                    "mouse_event_count"
                ].mean(),

            "mean_browser_event_count":
                group[
                    "browser_event_count"
                ].mean(),

            "mean_unique_apps":
                group[
                    "unique_apps"
                ].mean(),

            "mean_unique_event_types":
                group[
                    "unique_event_types"
                ].mean(),
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "false_positive_summary.csv",
        index=False,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TOP FALSE-POSITIVE CASES")
    print("=" * 70)

    display_columns = [
        "window_sec",
        "boundary_score",
        "distance_from_gt_sec",
        "nearest_gt_event_distance_sec",
        "nearest_gt_event_type",
        "local_event_count",
        "unique_event_types",
        "unique_apps",
        "app_switch_count",
        "event_type_counts",
    ]

    print(
        forensic_df[
            display_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 70)
    print("FALSE-POSITIVE SUMMARY")
    print("=" * 70)

    print(
        summary_df.to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 70)
    print("OUTPUTS")
    print("=" * 70)

    print(
        OUTPUT_DIR.resolve()
    )

    print(
        "\nCreated:"
    )

    print(
        "  hard_negative_scored_forensics.csv"
    )

    print(
        "  top_false_positive_candidates.csv"
    )

    print(
        "  false_positive_forensics.csv"
    )

    print(
        "  false_positive_summary.csv"
    )

    print("\n")
    print("=" * 70)
    print("PHASE 5C COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()