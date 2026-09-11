import json
import csv
from pathlib import Path
from collections import Counter
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(r"Dataset A\dataset_a")

OUTPUT_DIR = DATASET_ROOT / "day1_session_analysis"

SESSION_PROFILE_CSV = OUTPUT_DIR / "session_profiles.csv"
SESSION_PROFILE_JSON = OUTPUT_DIR / "session_profiles.json"
PROCESS_PRESENCE_CSV = OUTPUT_DIR / "session_process_presence.csv"


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    """Load JSONL file safely."""
    records = []

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARNING] Invalid JSON in {path}, line {line_no}")

    return records


def parse_timestamp(ts):
    """Parse ISO timestamp into datetime."""
    if not ts:
        return None

    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def get_process_code(event):
    """
    Get process code from a GT event.

    Primary source:
        current_process

    Fallback:
        process_code
    """
    return (
        event.get("current_process")
        or event.get("process_code")
        or ""
    )


def get_case_id(event):
    """Return case_id if available."""
    return event.get("case_id")


# ============================================================
# PROFILE ONE SESSION
# ============================================================

def profile_session(session_dir):

    session_id = session_dir.name
    gt_path = session_dir / "gt.jsonl"

    events = load_jsonl(gt_path)

    if not events:
        return None

    # --------------------------------------------------------
    # Basic event counts
    # --------------------------------------------------------

    event_types = Counter(
        event.get("event", "")
        for event in events
    )

    total_events = len(events)

    process_switched_out = event_types["process_switched_out"]
    process_started = event_types["process_started"]
    process_suspended = event_types["process_suspended"]
    process_resumed = event_types["process_resumed"]
    task_started = event_types["task_started"]
    clipboard_copy = event_types["clipboard_copy"]
    clipboard_paste = event_types["clipboard_paste"]

    # --------------------------------------------------------
    # Process information
    # --------------------------------------------------------

    process_codes = []

    for event in events:
        process = get_process_code(event)

        if process:
            process_codes.append(process)

    process_counter = Counter(process_codes)

    distinct_processes = sorted(process_counter.keys())

    # --------------------------------------------------------
    # Process starts
    #
    # IMPORTANT:
    # process_started events are NOT automatically treated as
    # unique executions because the schema allows duplicate
    # process_started events.
    # --------------------------------------------------------

    process_start_pairs = []

    for event in events:

        if event.get("event") != "process_started":
            continue

        process = get_process_code(event)
        case_id = get_case_id(event)

        if process:
            process_start_pairs.append(
                (
                    process,
                    case_id
                )
            )

    raw_process_starts = len(process_start_pairs)

    unique_process_case_pairs = set(process_start_pairs)

    unique_process_executions = len(unique_process_case_pairs)

    # --------------------------------------------------------
    # Case IDs
    # --------------------------------------------------------

    case_ids = {
        event.get("case_id")
        for event in events
        if event.get("case_id") not in (None, "")
    }

    distinct_cases = len(case_ids)

    # --------------------------------------------------------
    # Transitions
    # --------------------------------------------------------

    transitions = []

    for event in events:

        if event.get("event") != "process_switched_out":
            continue

        source = event.get("from")
        target = event.get("to")

        if source and target:
            transitions.append(
                (source, target)
            )

    transition_count = len(transitions)

    unique_transitions = len(set(transitions))

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    timestamps = []

    for event in events:
        dt = parse_timestamp(event.get("ts_utc"))

        if dt:
            timestamps.append(dt)

    duration_seconds = 0

    if len(timestamps) >= 2:
        start_time = min(timestamps)
        end_time = max(timestamps)

        duration_seconds = (
            end_time - start_time
        ).total_seconds()

    else:
        start_time = None
        end_time = None

    # --------------------------------------------------------
    # Process switch density
    # --------------------------------------------------------

    switches_per_minute = 0

    if duration_seconds > 0:
        switches_per_minute = (
            transition_count /
            (duration_seconds / 60)
        )

    # --------------------------------------------------------
    # Process diversity
    # --------------------------------------------------------

    process_diversity_ratio = 0

    if len(process_codes) > 0:
        process_diversity_ratio = (
            len(set(process_codes)) /
            len(process_codes)
        )

    # --------------------------------------------------------
    # Most frequent process
    # --------------------------------------------------------

    most_frequent_process = ""

    if process_counter:
        most_frequent_process = (
            process_counter.most_common(1)[0][0]
        )

    # --------------------------------------------------------
    # Serialize process frequencies
    # --------------------------------------------------------

    process_frequency_json = json.dumps(
        dict(process_counter),
        ensure_ascii=False
    )

    # --------------------------------------------------------
    # Serialize unique processes
    # --------------------------------------------------------

    process_list = ",".join(distinct_processes)

    # --------------------------------------------------------
    # Return profile
    # --------------------------------------------------------

    return {
        "session_id": session_id,

        "total_gt_events": total_events,

        "process_switched_out": process_switched_out,
        "process_started": process_started,
        "process_suspended": process_suspended,
        "process_resumed": process_resumed,
        "task_started": task_started,
        "clipboard_copy": clipboard_copy,
        "clipboard_paste": clipboard_paste,

        "distinct_processes": len(distinct_processes),
        "processes": process_list,

        "distinct_cases": distinct_cases,

        "raw_process_starts": raw_process_starts,
        "unique_process_executions": unique_process_executions,

        "transition_count": transition_count,
        "unique_transitions": unique_transitions,

        "duration_seconds": round(duration_seconds, 2),
        "duration_minutes": round(duration_seconds / 60, 2),

        "switches_per_minute": round(
            switches_per_minute,
            4
        ),

        "process_diversity_ratio": round(
            process_diversity_ratio,
            4
        ),

        "most_frequent_process": most_frequent_process,

        "process_frequency": process_frequency_json,

        "start_time": (
            start_time.isoformat()
            if start_time else ""
        ),

        "end_time": (
            end_time.isoformat()
            if end_time else ""
        )
    }


# ============================================================
# PROCESS PRESENCE MATRIX
# ============================================================

def create_process_presence_matrix(profiles):

    all_processes = set()

    for profile in profiles:

        processes = profile["processes"].split(",")

        for process in processes:
            if process:
                all_processes.add(process)

    all_processes = sorted(all_processes)

    rows = []

    for profile in profiles:

        processes = set(
            p for p in profile["processes"].split(",")
            if p
        )

        row = {
            "session_id": profile["session_id"]
        }

        for process in all_processes:
            row[process] = 1 if process in processes else 0

        rows.append(row)

    return rows, all_processes


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(path, rows):

    if not rows:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DAY 1 - SESSION PROFILE ANALYSIS")
    print("=" * 70)

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

    print(f"\nSessions found: {len(session_dirs)}")

    profiles = []

    for session_dir in session_dirs:

        profile = profile_session(session_dir)

        if profile is not None:
            profiles.append(profile)

    print(f"Sessions profiled: {len(profiles)}")

    # --------------------------------------------------------
    # Save session profile CSV
    # --------------------------------------------------------

    save_csv(
        SESSION_PROFILE_CSV,
        profiles
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with SESSION_PROFILE_JSON.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            profiles,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Process presence matrix
    # --------------------------------------------------------

    presence_rows, processes = (
        create_process_presence_matrix(profiles)
    )

    save_csv(
        PROCESS_PRESENCE_CSV,
        presence_rows
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    if not profiles:
        print("\nNo profiles generated.")
        return

    total_events = sum(
        p["total_gt_events"]
        for p in profiles
    )

    total_switches = sum(
        p["transition_count"]
        for p in profiles
    )

    total_cases = sum(
        p["distinct_cases"]
        for p in profiles
    )

    total_duration = sum(
        p["duration_seconds"]
        for p in profiles
    )

    # --------------------------------------------------------
    # Average statistics
    # --------------------------------------------------------

    avg_duration = (
        total_duration /
        len(profiles)
    )

    avg_switches = (
        total_switches /
        len(profiles)
    )

    avg_processes = sum(
        p["distinct_processes"]
        for p in profiles
    ) / len(profiles)

    avg_cases = sum(
        p["distinct_cases"]
        for p in profiles
    ) / len(profiles)

    # --------------------------------------------------------
    # Most active sessions
    # --------------------------------------------------------

    most_switching = sorted(
        profiles,
        key=lambda x: x["transition_count"],
        reverse=True
    )[:10]

    # --------------------------------------------------------
    # Longest sessions
    # --------------------------------------------------------

    longest_sessions = sorted(
        profiles,
        key=lambda x: x["duration_seconds"],
        reverse=True
    )[:10]

    # --------------------------------------------------------
    # Most diverse sessions
    # --------------------------------------------------------

    most_diverse = sorted(
        profiles,
        key=lambda x: x["distinct_processes"],
        reverse=True
    )[:10]

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Sessions                  : {len(profiles)}")
    print(f"Total GT events           : {total_events}")
    print(f"Total transitions         : {total_switches}")
    print(f"Total session cases       : {total_cases}")

    print(
        f"Average session duration  : "
        f"{avg_duration / 60:.2f} minutes"
    )

    print(
        f"Average process diversity : "
        f"{avg_processes:.2f} processes/session"
    )

    print(
        f"Average cases/session     : "
        f"{avg_cases:.2f}"
    )

    print(
        f"Average transitions       : "
        f"{avg_switches:.2f}/session"
    )

    print("\nProcesses discovered:")

    for process in processes:
        count = sum(
            1
            for p in profiles
            if process in p["processes"].split(",")
        )

        print(
            f"  {process}: "
            f"{count}/{len(profiles)} sessions"
        )

    # ========================================================
    # TOP SWITCHING SESSIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP 10 SESSIONS BY PROCESS SWITCHES")
    print("=" * 70)

    for i, profile in enumerate(
        most_switching,
        start=1
    ):

        print(
            f"{i:2}. "
            f"{profile['session_id']} | "
            f"switches={profile['transition_count']:3} | "
            f"processes={profile['distinct_processes']:2} | "
            f"cases={profile['distinct_cases']:2}"
        )

    # ========================================================
    # LONGEST SESSIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP 10 LONGEST SESSIONS")
    print("=" * 70)

    for i, profile in enumerate(
        longest_sessions,
        start=1
    ):

        print(
            f"{i:2}. "
            f"{profile['session_id']} | "
            f"duration="
            f"{profile['duration_minutes']:.2f} min | "
            f"switches={profile['transition_count']:3}"
        )

    # ========================================================
    # MOST DIVERSE SESSIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("TOP 10 MOST PROCESS-DIVERSE SESSIONS")
    print("=" * 70)

    for i, profile in enumerate(
        most_diverse,
        start=1
    ):

        print(
            f"{i:2}. "
            f"{profile['session_id']} | "
            f"processes={profile['distinct_processes']:2} | "
            f"switches={profile['transition_count']:3}"
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(f"Session profiles : {SESSION_PROFILE_CSV}")
    print(f"JSON             : {SESSION_PROFILE_JSON}")
    print(f"Presence matrix  : {PROCESS_PRESENCE_CSV}")

    print("\nDone.")


if __name__ == "__main__":
    main()