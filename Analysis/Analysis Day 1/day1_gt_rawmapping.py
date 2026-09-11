import json
import csv
from pathlib import Path
from collections import Counter
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(r"Dataset A\dataset_a")

OUTPUT_DIR = DATASET_ROOT / "day1_gt_raw_analysis"

EXECUTION_TABLE = OUTPUT_DIR / "gt_execution_windows.csv"
EVENT_TYPE_TABLE = OUTPUT_DIR / "process_event_type_summary.csv"
SOURCE_TABLE = OUTPUT_DIR / "process_source_summary.csv"
LAYER_TABLE = OUTPUT_DIR / "process_layer_summary.csv"
APP_TABLE = OUTPUT_DIR / "process_app_summary.csv"
MAPPING_JSON = OUTPUT_DIR / "gt_raw_mapping_summary.json"


# ============================================================
# SAFE VALUE HELPERS
# ============================================================

def safe_value(value):
    """
    Convert any JSON value into a safe string.
    Prevents 'unhashable type: dict' errors.
    """

    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False
        )

    return str(value)


def parse_timestamp(ts):
    """Parse GT ISO timestamp."""

    if not ts:
        return None

    try:
        return datetime.fromisoformat(
            str(ts).replace("Z", "+00:00")
        )
    except Exception:
        return None


def timestamp_ms(event):
    """Extract raw event timestamp_ms."""

    value = event.get("timestamp_ms")

    if value is None:
        return None

    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def get_process(event):
    """Extract process code from GT event."""

    process = (
        event.get("current_process")
        or event.get("process_code")
    )

    return safe_value(process)


def get_case_id(event):
    """Extract case ID safely."""

    return safe_value(
        event.get("case_id")
    )


# ============================================================
# JSONL LOADER
# ============================================================

def load_jsonl(path):

    records = []

    if not path.exists():
        return records

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line_no, line in enumerate(
            f,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:

                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:

                print(
                    f"[WARNING] Invalid JSON: "
                    f"{path} line {line_no}"
                )

    return records


# ============================================================
# APPLICATION EXTRACTION
# ============================================================

def get_application(event):

    context = event.get("context")

    if not isinstance(context, dict):
        return "UNKNOWN"

    app = (
        context.get("active_app")
        or context.get("application")
        or context.get("app")
        or context.get("active_application")
    )

    if app is None:
        return "UNKNOWN"

    return safe_value(app)


# ============================================================
# LOAD ALL RAW EVENTS FOR SESSION
# ============================================================

def build_raw_events(session_dir):

    chunk_dirs = sorted(
        p
        for p in session_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("chunk_")
    )

    raw_events = []

    for chunk_dir in chunk_dirs:

        events_path = (
            chunk_dir / "events.jsonl"
        )

        events = load_jsonl(
            events_path
        )

        raw_events.extend(events)

    # Sort chronologically

    raw_events.sort(
        key=lambda event: (
            timestamp_ms(event)
            if timestamp_ms(event) is not None
            else float("inf")
        )
    )

    return raw_events


# ============================================================
# RAW EVENTS IN TIME WINDOW
# ============================================================

def get_events_in_window(
    raw_events,
    start_dt,
    end_dt
):

    start_ms = (
        start_dt.timestamp() * 1000
    )

    end_ms = (
        end_dt.timestamp() * 1000
    )

    result = []

    for event in raw_events:

        event_ms = timestamp_ms(event)

        if event_ms is None:
            continue

        if (
            event_ms >= start_ms
            and event_ms <= end_ms
        ):

            result.append(event)

    return result


# ============================================================
# BUILD EXECUTION ROW
# ============================================================

def build_execution_row(
    session_id,
    execution_id,
    process,
    case_id,
    start_dt,
    end_dt,
    raw_events,
    previous_process="",
    next_process=""
):

    window_events = get_events_in_window(
        raw_events,
        start_dt,
        end_dt
    )

    event_types = Counter()
    sources = Counter()
    layers = Counter()
    applications = Counter()

    for event in window_events:

        event_type = (
            event.get("event_type")
            or event.get("event")
            or "UNKNOWN"
        )

        event_types[
            safe_value(event_type)
        ] += 1

        source = safe_value(
            event.get(
                "source",
                "UNKNOWN"
            )
        )

        sources[source] += 1

        layer = safe_value(
            event.get(
                "layer",
                "UNKNOWN"
            )
        )

        layers[layer] += 1

        app = get_application(event)

        applications[app] += 1

    duration_seconds = (
        end_dt - start_dt
    ).total_seconds()

    return {

        "session_id":
            session_id,

        "execution_id":
            execution_id,

        "process":
            process,

        "case_id":
            case_id,

        "start_time":
            start_dt.isoformat(),

        "end_time":
            end_dt.isoformat(),

        "duration_seconds":
            round(
                max(duration_seconds, 0),
                3
            ),

        "previous_process":
            previous_process,

        "next_process":
            next_process,

        "raw_event_count":
            len(window_events),

        "raw_event_types":
            json.dumps(
                dict(event_types),
                ensure_ascii=False
            ),

        "raw_sources":
            json.dumps(
                dict(sources),
                ensure_ascii=False
            ),

        "raw_layers":
            json.dumps(
                dict(layers),
                ensure_ascii=False
            ),

        "raw_apps":
            json.dumps(
                dict(applications),
                ensure_ascii=False
            )
    }


# ============================================================
# EXTRACT GT PROCESS WINDOWS
# ============================================================

def extract_gt_intervals(
    session_id,
    gt_events,
    raw_events
):

    intervals = []

    active_process = None
    active_case = ""
    active_start = None

    previous_process = ""

    execution_id = 0

    # --------------------------------------------------------
    # CLOSE CURRENT WINDOW
    # --------------------------------------------------------

    def close_interval(
        end_time,
        next_process=""
    ):

        nonlocal execution_id
        nonlocal active_process
        nonlocal active_case
        nonlocal active_start
        nonlocal previous_process

        if active_process is None:
            return

        if active_start is None:
            return

        if end_time is None:
            return

        if end_time < active_start:
            return

        execution_id += 1

        row = build_execution_row(
            session_id=session_id,
            execution_id=execution_id,
            process=active_process,
            case_id=active_case,
            start_dt=active_start,
            end_dt=end_time,
            raw_events=raw_events,
            previous_process=previous_process,
            next_process=next_process
        )

        intervals.append(row)

        previous_process = active_process

    # ========================================================
    # WALK THROUGH GT EVENTS
    # ========================================================

    for gt in gt_events:

        event_type = gt.get("event")

        process = get_process(gt)

        case_id = get_case_id(gt)

        timestamp = parse_timestamp(
            gt.get("ts_utc")
        )

        if timestamp is None:
            continue

        # ----------------------------------------------------
        # PROCESS STARTED
        # ----------------------------------------------------

        if event_type == "process_started":

            # No active process yet

            if active_process is None:

                active_process = process
                active_case = case_id
                active_start = timestamp

                continue

            # ------------------------------------------------
            # Duplicate process_started
            # ------------------------------------------------

            if (
                process == active_process
                and case_id == active_case
            ):

                continue

            # ------------------------------------------------
            # Same process, different case
            # ------------------------------------------------

            if process == active_process:

                close_interval(
                    timestamp,
                    next_process=process
                )

                active_process = process
                active_case = case_id
                active_start = timestamp

                continue

            # ------------------------------------------------
            # Different process without switch
            # ------------------------------------------------

            close_interval(
                timestamp,
                next_process=process
            )

            active_process = process
            active_case = case_id
            active_start = timestamp

            continue

        # ----------------------------------------------------
        # PROCESS SWITCHED OUT
        # ----------------------------------------------------

        if event_type == "process_switched_out":

            source = safe_value(
                gt.get("from")
            )

            target = safe_value(
                gt.get("to")
            )

            if active_process is not None:

                close_interval(
                    timestamp,
                    next_process=target
                )

            active_process = target
            active_case = ""
            active_start = timestamp

            continue

        # ----------------------------------------------------
        # SUSPENDED
        #
        # Do not close the process.
        # ----------------------------------------------------

        if event_type == "process_suspended":

            continue

        # ----------------------------------------------------
        # RESUMED
        # ----------------------------------------------------

        if event_type == "process_resumed":

            continue

        # ----------------------------------------------------
        # SESSION ENDED
        # ----------------------------------------------------

        if event_type == "session_ended":

            if active_process is not None:

                close_interval(
                    timestamp,
                    next_process=""
                )

                active_process = None
                active_case = ""
                active_start = None

            continue

    # ========================================================
    # CLOSE FINAL ACTIVE PROCESS
    # ========================================================

    if (
        active_process is not None
        and active_start is not None
    ):

        final_timestamp = None

        for gt in reversed(gt_events):

            candidate = parse_timestamp(
                gt.get("ts_utc")
            )

            if candidate is not None:

                final_timestamp = candidate
                break

        if (
            final_timestamp is not None
            and final_timestamp >= active_start
        ):

            close_interval(
                final_timestamp,
                next_process=""
            )

    return intervals


# ============================================================
# CSV SAVER
# ============================================================

def save_csv(
    path,
    rows
):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print(
        "DAY 1 - GT EXECUTION -> RAW EVENT MAPPING"
    )
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find sessions
    # --------------------------------------------------------

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

    all_execution_rows = []

    # ========================================================
    # PROCESS SESSIONS
    # ========================================================

    for index, session_dir in enumerate(
        session_dirs,
        start=1
    ):

        session_id = session_dir.name

        gt_path = (
            session_dir / "gt.jsonl"
        )

        gt_events = load_jsonl(
            gt_path
        )

        if not gt_events:

            print(
                f"[WARNING] No GT events: "
                f"{session_id}"
            )

            continue

        raw_events = build_raw_events(
            session_dir
        )

        intervals = extract_gt_intervals(
            session_id,
            gt_events,
            raw_events
        )

        all_execution_rows.extend(
            intervals
        )

        print(
            f"[{index:02}/{len(session_dirs)}] "
            f"{session_id} | "
            f"GT={len(gt_events)} | "
            f"RAW={len(raw_events)} | "
            f"windows={len(intervals)}"
        )

    # ========================================================
    # AGGREGATION
    # ========================================================

    process_event_types = Counter()
    process_sources = Counter()
    process_layers = Counter()
    process_apps = Counter()
    process_raw_event_count = Counter()

    for row in all_execution_rows:

        process = row["process"]

        process_raw_event_count[
            process
        ] += row["raw_event_count"]

        # Event types

        event_types = json.loads(
            row["raw_event_types"]
        )

        for event_type, count in (
            event_types.items()
        ):

            process_event_types[
                (process, event_type)
            ] += count

        # Sources

        sources = json.loads(
            row["raw_sources"]
        )

        for source, count in (
            sources.items()
        ):

            process_sources[
                (process, source)
            ] += count

        # Layers

        layers = json.loads(
            row["raw_layers"]
        )

        for layer, count in (
            layers.items()
        ):

            process_layers[
                (process, layer)
            ] += count

        # Applications

        apps = json.loads(
            row["raw_apps"]
        )

        for app, count in (
            apps.items()
        ):

            process_apps[
                (process, app)
            ] += count

    # ========================================================
    # EXECUTION WINDOW TABLE
    # ========================================================

    save_csv(
        EXECUTION_TABLE,
        all_execution_rows
    )

    # ========================================================
    # EVENT TYPE TABLE
    # ========================================================

    event_type_rows = []

    for (
        process,
        event_type
    ), count in sorted(
        process_event_types.items(),
        key=lambda x: (
            x[0][0],
            -x[1]
        )
    ):

        event_type_rows.append({

            "process":
                process,

            "raw_event_type":
                event_type,

            "count":
                count
        })

    save_csv(
        EVENT_TYPE_TABLE,
        event_type_rows
    )

    # ========================================================
    # SOURCE TABLE
    # ========================================================

    source_rows = []

    for (
        process,
        source
    ), count in sorted(
        process_sources.items(),
        key=lambda x: (
            x[0][0],
            -x[1]
        )
    ):

        source_rows.append({

            "process":
                process,

            "raw_source":
                source,

            "count":
                count
        })

    save_csv(
        SOURCE_TABLE,
        source_rows
    )

    # ========================================================
    # LAYER TABLE
    # ========================================================

    layer_rows = []

    for (
        process,
        layer
    ), count in sorted(
        process_layers.items(),
        key=lambda x: (
            x[0][0],
            -x[1]
        )
    ):

        layer_rows.append({

            "process":
                process,

            "raw_layer":
                layer,

            "count":
                count
        })

    save_csv(
        LAYER_TABLE,
        layer_rows
    )

    # ========================================================
    # APPLICATION TABLE
    # ========================================================

    app_rows = []

    for (
        process,
        app
    ), count in sorted(
        process_apps.items(),
        key=lambda x: (
            x[0][0],
            -x[1]
        )
    ):

        app_rows.append({

            "process":
                process,

            "application":
                app,

            "count":
                count
        })

    save_csv(
        APP_TABLE,
        app_rows
    )

    # ========================================================
    # MAPPING QUALITY
    # ========================================================

    total_windows = len(
        all_execution_rows
    )

    windows_with_raw_events = sum(
        1
        for row in all_execution_rows
        if row["raw_event_count"] > 0
    )

    windows_without_raw_events = (
        total_windows
        - windows_with_raw_events
    )

    total_raw_events_mapped = sum(
        row["raw_event_count"]
        for row in all_execution_rows
    )

    if total_windows > 0:

        coverage_ratio = (
            windows_with_raw_events
            / total_windows
        )

        average_raw_events = (
            total_raw_events_mapped
            / total_windows
        )

    else:

        coverage_ratio = 0
        average_raw_events = 0

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {

        "dataset":
            "Dataset A",

        "sessions":
            len(session_dirs),

        "gt_execution_windows":
            total_windows,

        "windows_with_raw_events":
            windows_with_raw_events,

        "windows_without_raw_events":
            windows_without_raw_events,

        "window_raw_event_coverage":
            round(
                coverage_ratio,
                4
            ),

        "total_mapped_raw_events":
            total_raw_events_mapped,

        "average_raw_events_per_window":
            round(
                average_raw_events,
                3
            ),

        "raw_events_by_process":
            dict(
                process_raw_event_count
            )
    }

    with MAPPING_JSON.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # PRINT MAPPING SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("MAPPING SUMMARY")
    print("=" * 75)

    print(
        f"Sessions                : "
        f"{len(session_dirs)}"
    )

    print(
        f"GT execution windows    : "
        f"{total_windows}"
    )

    print(
        f"Windows with raw events : "
        f"{windows_with_raw_events}"
    )

    print(
        f"Windows without raw     : "
        f"{windows_without_raw_events}"
    )

    print(
        f"Raw event coverage      : "
        f"{coverage_ratio * 100:.2f}%"
    )

    print(
        f"Mapped raw events       : "
        f"{total_raw_events_mapped}"
    )

    print(
        f"Avg raw events/window   : "
        f"{average_raw_events:.2f}"
    )

    # ========================================================
    # TOP RAW EVENT TYPES
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "TOP RAW EVENT TYPES PER PROCESS"
    )
    print("=" * 75)

    processes = sorted(
        process_raw_event_count.keys()
    )

    for process in processes:

        counts = Counter()

        for (
            p,
            event_type
        ), count in process_event_types.items():

            if p == process:

                counts[event_type] += count

        print(
            f"\nProcess {process}:"
        )

        for event_type, count in (
            counts.most_common(8)
        ):

            print(
                f"  "
                f"{event_type:<35}"
                f"{count}"
            )

    # ========================================================
    # TOP APPLICATIONS
    # ========================================================

    print("\n" + "=" * 75)
    print(
        "TOP APPLICATIONS PER PROCESS"
    )
    print("=" * 75)

    for process in processes:

        counts = Counter()

        for (
            p,
            app
        ), count in process_apps.items():

            if p == process:

                counts[app] += count

        print(
            f"\nProcess {process}:"
        )

        for app, count in (
            counts.most_common(5)
        ):

            print(
                f"  "
                f"{app:<35}"
                f"{count}"
            )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print("\n" + "=" * 75)
    print("OUTPUT FILES")
    print("=" * 75)

    print(
        f"Execution windows : "
        f"{EXECUTION_TABLE}"
    )

    print(
        f"Event types       : "
        f"{EVENT_TYPE_TABLE}"
    )

    print(
        f"Sources           : "
        f"{SOURCE_TABLE}"
    )

    print(
        f"Layers            : "
        f"{LAYER_TABLE}"
    )

    print(
        f"Applications      : "
        f"{APP_TABLE}"
    )

    print(
        f"Summary JSON      : "
        f"{MAPPING_JSON}"
    )

    print("\nDone.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()