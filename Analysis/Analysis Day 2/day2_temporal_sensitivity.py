from pathlib import Path
import json
import pandas as pd


# ============================================================
# DAY 2 — TEMPORAL SENSITIVITY ANALYSIS
# ============================================================

DATASET_ROOT = Path(
    r"Dataset A\dataset_a"
)

BOUNDARY_FILE = Path(
    r"Outputs\Day 2\day2_gt_boundary_analysis\gt_switch_boundaries.csv"
)

OUTPUT_DIR = Path(
    r"Outputs\Day 2\day2_temporal_sensitivity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Temporal windows under investigation
# ------------------------------------------------------------

WINDOWS_SEC = [
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0
]


# ------------------------------------------------------------
# Candidate raw-event signals
# ------------------------------------------------------------

SIGNAL_FIELDS = [
    "event_type",
    "layer",
    "source",
    "app_name",
    "process_name",
    "window_title",
    "browser_domain",
    "browser_path",
    "browser_element"
]


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):

    records = []

    invalid_records = 0

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_no, line in enumerate(
            f,
            1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError:

                invalid_records += 1

                continue


            if not isinstance(
                record,
                dict
            ):

                invalid_records += 1

                continue


            records.append(
                record
            )


    if invalid_records > 0:

        print(
            f"  WARNING: "
            f"{invalid_records} invalid/non-object "
            f"records skipped from {path.name}"
        )


    return records


# ============================================================
# RAW SIGNAL EXTRACTION
# ============================================================

def normalize_value(value):

    if value is None:
        return None

    if isinstance(
        value,
        (dict, list)
    ):

        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False
        )

    return str(value)


def extract_signal(
    record,
    field
):

    if not isinstance(
        record,
        dict
    ):

        return None


    # --------------------------------------------------------
    # Basic metadata
    # --------------------------------------------------------

    if field == "event_type":

        return record.get(
            "event_type"
        )


    if field == "layer":

        return record.get(
            "layer"
        )


    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    if field == "source":

        source = record.get(
            "source"
        )

        if isinstance(
            source,
            dict
        ):

            return (
                source.get(
                    "agent_version"
                )
                or
                source.get(
                    "os"
                )
            )

        return source


    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context = record.get(
        "context"
    )

    if not isinstance(
        context,
        dict
    ):

        context = {}


    active_app = context.get(
        "active_app"
    )

    if not isinstance(
        active_app,
        dict
    ):

        active_app = {}


    if field == "app_name":

        return active_app.get(
            "app_name"
        )


    if field == "process_name":

        return active_app.get(
            "process_name"
        )


    if field == "window_title":

        return active_app.get(
            "window_title"
        )


    # --------------------------------------------------------
    # Browser
    # --------------------------------------------------------

    browser_tab = context.get(
        "active_browser_tab"
    )

    if not isinstance(
        browser_tab,
        dict
    ):

        browser_tab = {}


    if field == "browser_domain":

        return (
            browser_tab.get(
                "domain"
            )
            or
            browser_tab.get(
                "host"
            )
        )


    if field == "browser_path":

        return (
            browser_tab.get(
                "path"
            )
            or
            browser_tab.get(
                "url_path"
            )
        )


    if field == "browser_element":

        return (
            browser_tab.get(
                "element"
            )
            or
            context.get(
                "browser_element"
            )
        )


    return None


# ============================================================
# START
# ============================================================

print("=" * 70)
print("DAY 2 — TEMPORAL SENSITIVITY ANALYSIS")
print("=" * 70)


# ============================================================
# LOAD GT BOUNDARIES
# ============================================================

print("\n" + "-" * 70)
print("LOADING GT BOUNDARIES")
print("-" * 70)


if not BOUNDARY_FILE.exists():

    raise FileNotFoundError(
        f"Boundary file not found:\n"
        f"{BOUNDARY_FILE.resolve()}"
    )


boundaries = pd.read_csv(
    BOUNDARY_FILE
)


print(
    "\nBoundary CSV columns:"
)

print(
    boundaries.columns.tolist()
)


print(
    f"\nCandidate GT boundaries: "
    f"{len(boundaries)}"
)


if boundaries.empty:

    raise RuntimeError(
        "GT boundary CSV is empty."
    )


# ------------------------------------------------------------
# Session column
# ------------------------------------------------------------

if "session_id" not in boundaries.columns:

    raise RuntimeError(
        "GT boundary CSV does not contain "
        "'session_id'."
    )


# ------------------------------------------------------------
# Numeric timestamp column
# ------------------------------------------------------------

if "timestamp_ms" not in boundaries.columns:

    raise RuntimeError(
        "GT boundary CSV does not contain "
        "'timestamp_ms'."
    )


print(
    "Using session column: session_id"
)

print(
    "Using timestamp column: timestamp_ms"
)


# ============================================================
# LOAD RAW EVENTS
# ============================================================

print("\n" + "-" * 70)
print("LOADING RAW EVENTS")
print("-" * 70)


session_events = {}


session_dirs = sorted(
    p
    for p in DATASET_ROOT.iterdir()
    if p.is_dir()
    and p.name.startswith("ses_")
)


print(
    f"\nSessions found: "
    f"{len(session_dirs)}"
)


total_raw_events = 0
total_events_files = 0


for session_dir in session_dirs:

    events = []


    # --------------------------------------------------------
    # Recursive search for actual events.jsonl files
    # --------------------------------------------------------

    events_files = sorted(
        session_dir.rglob(
            "events.jsonl"
        )
    )


    total_events_files += len(
        events_files
    )


    for events_file in events_files:

        records = load_jsonl(
            events_file
        )


        for record in records:

            timestamp_ms = record.get(
                "timestamp_ms"
            )


            # ------------------------------------------------
            # Require valid numeric timestamp
            # ------------------------------------------------

            try:

                timestamp_ms = int(
                    timestamp_ms
                )

            except (
                ValueError,
                TypeError,
                OverflowError
            ):

                continue


            # ------------------------------------------------
            # Create normalized event
            # ------------------------------------------------

            event = {

                "timestamp_ms":
                    timestamp_ms

            }


            for field in SIGNAL_FIELDS:

                event[field] = normalize_value(
                    extract_signal(
                        record,
                        field
                    )
                )


            events.append(
                event
            )


    # --------------------------------------------------------
    # Chronological order
    # --------------------------------------------------------

    events.sort(
        key=lambda event:
        event["timestamp_ms"]
    )


    session_events[
        session_dir.name
    ] = events


    total_raw_events += len(
        events
    )


    print(
        f"{session_dir.name}: "
        f"{len(events):,} raw events "
        f"({len(events_files)} events.jsonl files)"
    )


# ============================================================
# RAW EVENT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RAW EVENT LOADING SUMMARY")
print("=" * 70)


print(
    f"Sessions loaded:       "
    f"{len(session_events)}"
)


print(
    f"events.jsonl files:    "
    f"{total_events_files}"
)


print(
    f"Total raw events:      "
    f"{total_raw_events:,}"
)


if total_raw_events == 0:

    raise RuntimeError(
        "ZERO RAW EVENTS WERE LOADED."
    )


# ============================================================
# ANALYZE BOUNDARIES
# ============================================================

print("\n" + "-" * 70)
print("ANALYZING GT BOUNDARIES")
print("-" * 70)


rows = []


matched_boundaries = 0
unmatched_boundaries = 0
invalid_timestamps = 0


# ============================================================
# BOUNDARY LOOP
# ============================================================

for _, boundary in boundaries.iterrows():

    # --------------------------------------------------------
    # Session
    # --------------------------------------------------------

    session_id = str(
        boundary[
            "session_id"
        ]
    ).strip()


    # --------------------------------------------------------
    # GT timestamp_ms
    # --------------------------------------------------------

    raw_boundary_timestamp = boundary[
        "timestamp_ms"
    ]


    try:

        boundary_timestamp_ms = int(
            raw_boundary_timestamp
        )

    except (
        ValueError,
        TypeError,
        OverflowError
    ):

        invalid_timestamps += 1

        continue


    # --------------------------------------------------------
    # Find session raw events
    # --------------------------------------------------------

    events = session_events.get(
        session_id
    )


    if events is None:

        unmatched_boundaries += 1

        continue


    if not events:

        unmatched_boundaries += 1

        continue


    matched_boundaries += 1


    # ========================================================
    # TEMPORAL WINDOWS
    # ========================================================

    for window_sec in WINDOWS_SEC:

        window_ms = int(
            window_sec * 1000
        )


        window_start_ms = (
            boundary_timestamp_ms
            -
            window_ms
        )


        window_end_ms = (
            boundary_timestamp_ms
            +
            window_ms
        )


        # ----------------------------------------------------
        # PRE EVENTS
        # ----------------------------------------------------

        pre_events = [

            event

            for event in events

            if (
                window_start_ms
                <=
                event["timestamp_ms"]
                <
                boundary_timestamp_ms
            )

        ]


        # ----------------------------------------------------
        # POST EVENTS
        # ----------------------------------------------------

        post_events = [

            event

            for event in events

            if (
                boundary_timestamp_ms
                <=
                event["timestamp_ms"]
                <=
                window_end_ms
            )

        ]


        # ----------------------------------------------------
        # Basic features
        # ----------------------------------------------------

        row = {

            "session_id":
                session_id,

            "boundary_timestamp_ms":
                boundary_timestamp_ms,

            "window_sec":
                window_sec,

            "pre_event_count":
                len(pre_events),

            "post_event_count":
                len(post_events),

            "pre_event_rate":
                len(pre_events)
                /
                window_sec,

            "post_event_rate":
                len(post_events)
                /
                window_sec

        }


        # ====================================================
        # SIGNAL COMPARISON
        # ====================================================

        for field in SIGNAL_FIELDS:

            pre_values = {

                event[field]

                for event in pre_events

                if event.get(field) is not None

            }


            post_values = {

                event[field]

                for event in post_events

                if event.get(field) is not None

            }


            intersection = (
                pre_values
                &
                post_values
            )


            union = (
                pre_values
                |
                post_values
            )


            # ------------------------------------------------
            # Candidate Jaccard similarity
            # ------------------------------------------------

            if union:

                jaccard = (
                    len(intersection)
                    /
                    len(union)
                )

            else:

                jaccard = 1.0


            # ------------------------------------------------
            # Store features
            # ------------------------------------------------

            row[
                f"pre_unique_{field}"
            ] = len(
                pre_values
            )


            row[
                f"post_unique_{field}"
            ] = len(
                post_values
            )


            row[
                f"new_{field}_count"
            ] = len(
                post_values
                -
                pre_values
            )


            row[
                f"disappeared_{field}_count"
            ] = len(
                pre_values
                -
                post_values
            )


            row[
                f"jaccard_{field}"
            ] = jaccard


            row[
                f"change_{field}"
            ] = int(
                pre_values
                !=
                post_values
            )


        rows.append(
            row
        )


# ============================================================
# MATCHING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("BOUNDARY MATCHING")
print("=" * 70)


print(
    f"Total boundaries:       "
    f"{len(boundaries)}"
)


print(
    f"Matched boundaries:     "
    f"{matched_boundaries}"
)


print(
    f"Unmatched boundaries:   "
    f"{unmatched_boundaries}"
)


print(
    f"Invalid timestamps:     "
    f"{invalid_timestamps}"
)


if matched_boundaries == 0:

    raise RuntimeError(
        "\nZERO GT boundaries matched "
        "to raw sessions."
    )


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(
    rows
)


if results.empty:

    raise RuntimeError(
        "\nTemporal analysis produced ZERO rows."
    )


print(
    f"\nBoundary-window observations: "
    f"{len(results):,}"
)


# ============================================================
# SAVE BOUNDARY-LEVEL DATA
# ============================================================

boundary_output = (
    OUTPUT_DIR
    /
    "boundary_temporal_features.csv"
)


results.to_csv(
    boundary_output,
    index=False
)


# ============================================================
# AGGREGATE WINDOW RESULTS
# ============================================================

summary_rows = []


for window_sec, group in results.groupby(
    "window_sec"
):

    row = {

        "window_sec":
            window_sec,

        "boundaries":
            group[
                "boundary_timestamp_ms"
            ].nunique(),

        "mean_pre_event_rate":
            group[
                "pre_event_rate"
            ].mean(),

        "mean_post_event_rate":
            group[
                "post_event_rate"
            ].mean(),

        "median_pre_event_rate":
            group[
                "pre_event_rate"
            ].median(),

        "median_post_event_rate":
            group[
                "post_event_rate"
            ].median()

    }


    for field in SIGNAL_FIELDS:

        row[
            f"mean_jaccard_{field}"
        ] = group[
            f"jaccard_{field}"
        ].mean()


        row[
            f"median_jaccard_{field}"
        ] = group[
            f"jaccard_{field}"
        ].median()


        row[
            f"mean_new_{field}"
        ] = group[
            f"new_{field}_count"
        ].mean()


        row[
            f"mean_disappeared_{field}"
        ] = group[
            f"disappeared_{field}_count"
        ].mean()


        row[
            f"change_rate_{field}"
        ] = group[
            f"change_{field}"
        ].mean()


    summary_rows.append(
        row
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# SAVE WINDOW SUMMARY
# ============================================================

summary_output = (
    OUTPUT_DIR
    /
    "temporal_window_summary.csv"
)


summary.to_csv(
    summary_output,
    index=False
)


# ============================================================
# PRINT WINDOW SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TEMPORAL WINDOW SUMMARY")
print("=" * 70)


display_columns = [

    "window_sec",

    "boundaries",

    "mean_pre_event_rate",

    "mean_post_event_rate",

    "median_pre_event_rate",

    "median_post_event_rate",

    "mean_jaccard_event_type",

    "mean_jaccard_layer",

    "mean_jaccard_source",

    "mean_jaccard_app_name",

    "mean_jaccard_process_name",

    "mean_jaccard_window_title",

    "mean_jaccard_browser_domain",

    "change_rate_event_type",

    "change_rate_app_name",

    "change_rate_process_name",

    "change_rate_window_title",

    "change_rate_browser_domain"

]


available_columns = [

    column

    for column in display_columns

    if column in summary.columns

]


print(
    summary[
        available_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

global_summary = {

    "candidate_boundaries":
        int(
            len(boundaries)
        ),

    "matched_boundaries":
        int(
            matched_boundaries
        ),

    "unmatched_boundaries":
        int(
            unmatched_boundaries
        ),

    "invalid_timestamps":
        int(
            invalid_timestamps
        ),

    "total_raw_events":
        int(
            total_raw_events
        ),

    "total_events_files":
        int(
            total_events_files
        ),

    "windows_tested":
        WINDOWS_SEC,

    "signal_fields":
        SIGNAL_FIELDS

}


json_output = (
    OUTPUT_DIR
    /
    "day2_temporal_sensitivity_summary.json"
)


with open(
    json_output,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        global_summary,
        f,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("OUTPUTS")
print("=" * 70)


print(
    boundary_output
)


print(
    summary_output
)


print(
    json_output
)


print("\n" + "=" * 70)
print("DAY 2 PHASE 2 COMPLETE")
print("=" * 70)