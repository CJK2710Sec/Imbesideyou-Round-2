from pathlib import Path
import json
import csv
from collections import Counter, defaultdict
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

# Change this if your Dataset A is located somewhere else.
DATASET_ROOT = Path(r"Dataset A\dataset_a")

OUTPUT_DIR = Path(r"Outputs\Day 2\day2_gt_boundary_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# BASIC HELPERS
# ============================================================

def load_jsonl(path):
    """
    Load a JSONL file into a list of dictionaries.
    """
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNING] Invalid JSON in {path}, line {line_number}: {e}")

    return records


def parse_timestamp(ts):
    """
    Parse ISO UTC timestamp.

    Example:
    2026-07-01T07:57:12.251773+00:00
    """
    if ts is None:
        return None

    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def timestamp_to_ms(ts):
    """
    Convert ISO timestamp to Unix milliseconds.
    """
    dt = parse_timestamp(ts)

    if dt is None:
        return None

    return int(dt.timestamp() * 1000)


# ============================================================
# GT TRANSITION EXTRACTION
# ============================================================

def extract_gt_events(gt_records):
    """
    Extract all relevant GT process events.

    We keep:
        process_started
        process_switched_out
        process_suspended
        process_resumed
        task_started

    This allows us to understand the GT structure before deciding
    which events should be treated as actual boundaries.
    """

    extracted = []

    for index, record in enumerate(gt_records):

        event_type = record.get("event")

        timestamp = (
            record.get("ts_utc")
            or record.get("timestamp")
            or record.get("timestamp_iso")
        )

        event = {
            "gt_index": index,
            "event": event_type,
            "timestamp": timestamp,
            "timestamp_ms": timestamp_to_ms(timestamp),
            "current_process": record.get("current_process"),
            "process_variant": record.get("process_variant"),
        }

        # Process start
        if event_type == "process_started":
            event["process_code"] = record.get("process_code")
            event["process_name"] = record.get("process_name")
            event["case_id"] = record.get("case_id")

        # Process switch
        elif event_type == "process_switched_out":
            event["from_process"] = record.get("from")
            event["to_process"] = record.get("to")

        # Suspension
        elif event_type == "process_suspended":
            event["from_process"] = record.get("from")
            event["to_process"] = record.get("to")
            event["split_id"] = record.get("split_id")

        # Resumption
        elif event_type == "process_resumed":
            event["process_code"] = record.get("process_code")
            event["split_id"] = record.get("split_id")
            event["phase"] = record.get("phase")

        # Task
        elif event_type == "task_started":
            event["task_id"] = record.get("task_id")
            event["action"] = record.get("action")
            event["entity"] = record.get("entity")

        extracted.append(event)

    return extracted


# ============================================================
# TRANSITION CONSISTENCY CHECK
# ============================================================

def check_transition_consistency(events):
    """
    Examine process_switched_out events for obvious inconsistencies.

    IMPORTANT:
    This function does NOT attempt to "fix" the GT.

    It only reports suspicious patterns.

    This follows the DATA_SCHEMA warning that GT pairing should
    be verified before relying on it.
    """

    switches = [
        e for e in events
        if e["event"] == "process_switched_out"
    ]

    starts = [
        e for e in events
        if e["event"] == "process_started"
    ]

    suspensions = [
        e for e in events
        if e["event"] == "process_suspended"
    ]

    resumptions = [
        e for e in events
        if e["event"] == "process_resumed"
    ]

    issues = []

    # --------------------------------------------------------
    # Check duplicate consecutive process_started
    # --------------------------------------------------------

    for i in range(1, len(starts)):

        previous = starts[i - 1]
        current = starts[i]

        if (
            previous.get("process_code")
            == current.get("process_code")
            and previous.get("case_id")
            == current.get("case_id")
        ):
            issues.append({
                "type": "duplicate_process_started",
                "gt_index_1": previous["gt_index"],
                "gt_index_2": current["gt_index"],
                "process_code": current.get("process_code"),
                "case_id": current.get("case_id"),
            })

    # --------------------------------------------------------
    # Check switch records for missing from/to
    # --------------------------------------------------------

    for switch in switches:

        if not switch.get("from_process"):
            issues.append({
                "type": "switch_missing_from",
                "gt_index": switch["gt_index"],
                "timestamp": switch["timestamp"],
            })

        if not switch.get("to_process"):
            issues.append({
                "type": "switch_missing_to",
                "gt_index": switch["gt_index"],
                "timestamp": switch["timestamp"],
            })

    # --------------------------------------------------------
    # Check chronological ordering
    # --------------------------------------------------------

    timestamps = [
        e["timestamp_ms"]
        for e in events
        if e["timestamp_ms"] is not None
    ]

    for i in range(1, len(timestamps)):

        if timestamps[i] < timestamps[i - 1]:

            issues.append({
                "type": "non_monotonic_timestamp",
                "position": i,
            })

    return {
        "num_switches": len(switches),
        "num_starts": len(starts),
        "num_suspensions": len(suspensions),
        "num_resumptions": len(resumptions),
        "issues": issues,
    }


# ============================================================
# EXTRACT ACTUAL SWITCH BOUNDARIES
# ============================================================

def extract_switch_boundaries(events):
    """
    Extract process_switched_out events.

    At this stage, these are our candidate 'true boundaries'.

    We DO NOT yet claim that every switch should become a
    segmentation boundary in the final algorithm.
    """

    boundaries = []

    for event in events:

        if event["event"] != "process_switched_out":
            continue

        boundaries.append({
            "gt_index": event["gt_index"],
            "timestamp": event["timestamp"],
            "timestamp_ms": event["timestamp_ms"],
            "from_process": event.get("from_process"),
            "to_process": event.get("to_process"),
            "current_process": event.get("current_process"),
            "process_variant": event.get("process_variant"),
        })

    return boundaries


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(path, rows):

    if not rows:
        return

    fieldnames = sorted({
        key
        for row in rows
        for key in row.keys()
    })

    with open(path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN SESSION ANALYSIS
# ============================================================

def analyze_session(session_dir):

    gt_path = session_dir / "gt.jsonl"

    if not gt_path.exists():
        return None

    gt_records = load_jsonl(gt_path)

    events = extract_gt_events(gt_records)

    consistency = check_transition_consistency(events)

    boundaries = extract_switch_boundaries(events)

    session_result = {
        "session_id": session_dir.name,

        "num_gt_records": len(gt_records),

        "num_process_starts": consistency["num_starts"],
        "num_process_switches": consistency["num_switches"],

        "num_process_suspensions":
            consistency["num_suspensions"],

        "num_process_resumptions":
            consistency["num_resumptions"],

        "num_gt_issues":
            len(consistency["issues"]),
    }

    return session_result, boundaries, consistency["issues"]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DAY 2 — GT BOUNDARY ANALYSIS")
    print("=" * 70)

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"Dataset root does not exist:\n{DATASET_ROOT}"
        )

    session_dirs = sorted([
        p for p in DATASET_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("ses_")
    ])

    print(f"\nSessions found: {len(session_dirs)}")

    all_session_results = []
    all_boundaries = []
    all_issues = []

    # --------------------------------------------------------
    # Analyze every session
    # --------------------------------------------------------

    for session_dir in session_dirs:

        result = analyze_session(session_dir)

        if result is None:
            print(
                f"[WARNING] No gt.jsonl found: "
                f"{session_dir.name}"
            )
            continue

        session_result, boundaries, issues = result

        all_session_results.append(session_result)

        all_boundaries.extend([
            {
                "session_id": session_dir.name,
                **boundary
            }
            for boundary in boundaries
        ])

        all_issues.extend([
            {
                "session_id": session_dir.name,
                **issue
            }
            for issue in issues
        ])

    # ========================================================
    # GLOBAL SUMMARY
    # ========================================================

    event_counts = Counter()

    for session_dir in session_dirs:

        gt_path = session_dir / "gt.jsonl"

        if not gt_path.exists():
            continue

        records = load_jsonl(gt_path)

        for record in records:
            event_counts[record.get("event")] += 1

    summary = {

        "dataset_root": str(DATASET_ROOT),

        "sessions_found": len(session_dirs),

        "sessions_analyzed":
            len(all_session_results),

        "total_gt_boundaries":
            len(all_boundaries),

        "gt_event_counts":
            dict(event_counts),

        "total_gt_issues":
            len(all_issues),

        "issue_counts":
            dict(
                Counter(
                    issue["type"]
                    for issue in all_issues
                )
            ),

        "note":
            "process_switched_out events are treated as "
            "candidate boundary events for Day 2 analysis. "
            "This script does not assume they are the final "
            "segmentation boundaries."
    }

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "-" * 70)
    print("GLOBAL GT EVENT COUNTS")
    print("-" * 70)

    for event_type, count in event_counts.most_common():

        print(
            f"{event_type:30s} {count:8d}"
        )

    print("\n" + "-" * 70)
    print("BOUNDARY SUMMARY")
    print("-" * 70)

    print(
        f"Candidate process switches: "
        f"{len(all_boundaries)}"
    )

    print(
        f"GT consistency issues: "
        f"{len(all_issues)}"
    )

    print("\n" + "-" * 70)
    print("ISSUE TYPES")
    print("-" * 70)

    issue_counts = Counter(
        issue["type"]
        for issue in all_issues
    )

    if issue_counts:

        for issue_type, count in issue_counts.most_common():

            print(
                f"{issue_type:35s} {count:8d}"
            )

    else:

        print("No consistency issues detected.")

    # ========================================================
    # SAVE OUTPUTS
    # ========================================================

    save_csv(
        OUTPUT_DIR / "session_gt_summary.csv",
        all_session_results
    )

    save_csv(
        OUTPUT_DIR / "gt_switch_boundaries.csv",
        all_boundaries
    )

    save_csv(
        OUTPUT_DIR / "gt_consistency_issues.csv",
        all_issues
    )

    with open(
        OUTPUT_DIR / "day2_gt_boundary_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)

    print(
        OUTPUT_DIR /
        "session_gt_summary.csv"
    )

    print(
        OUTPUT_DIR /
        "gt_switch_boundaries.csv"
    )

    print(
        OUTPUT_DIR /
        "gt_consistency_issues.csv"
    )

    print(
        OUTPUT_DIR /
        "day2_gt_boundary_summary.json"
    )

    print("\nDay 2 Phase 1 complete.")


if __name__ == "__main__":
    main()