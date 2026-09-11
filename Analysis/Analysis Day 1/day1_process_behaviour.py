import json
import csv
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(r"Dataset A\dataset_a")

INPUT_DIR = DATASET_ROOT / "day1_session_analysis"
SESSION_PROFILE_JSON = INPUT_DIR / "session_profiles.json"

OUTPUT_DIR = DATASET_ROOT / "day1_process_analysis"

PROCESS_STATS_CSV = OUTPUT_DIR / "process_statistics.csv"
PREDECESSOR_CSV = OUTPUT_DIR / "process_predecessors.csv"
SUCCESSOR_CSV = OUTPUT_DIR / "process_successors.csv"
CO_OCCURRENCE_CSV = OUTPUT_DIR / "process_cooccurrence.csv"
PROCESS_SUMMARY_JSON = OUTPUT_DIR / "process_behavior_summary.json"


# ============================================================
# HELPERS
# ============================================================

def load_jsonl(path):
    records = []

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(
                    f"[WARNING] Invalid JSON: "
                    f"{path} line {line_no}"
                )

    return records


def save_csv(path, rows):

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
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    print("=" * 75)
    print("DAY 1 - CROSS-SESSION PROCESS BEHAVIOR ANALYSIS")
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find sessions
    # --------------------------------------------------------

    session_dirs = sorted(
        p for p in DATASET_ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith("ses_")
    )

    print(f"\nSessions found: {len(session_dirs)}")

    # --------------------------------------------------------
    # Global structures
    # --------------------------------------------------------

    process_sessions = defaultdict(set)

    process_started_count = Counter()
    process_case_pairs = defaultdict(set)

    process_incoming = Counter()
    process_outgoing = Counter()

    predecessor_counts = Counter()
    successor_counts = Counter()

    process_suspend_count = Counter()
    process_resume_count = Counter()

    process_event_count = Counter()

    # For co-occurrence
    session_process_sets = []

    total_transitions = 0

    # ========================================================
    # PROCESS EACH SESSION
    # ========================================================

    for session_dir in session_dirs:

        gt_path = session_dir / "gt.jsonl"

        events = load_jsonl(gt_path)

        if not events:
            continue

        session_processes = set()

        for event in events:

            event_type = event.get("event")

            process = (
                event.get("current_process")
                or event.get("process_code")
                or ""
            )

            case_id = event.get("case_id")

            # ------------------------------------------------
            # Any event associated with a process
            # ------------------------------------------------

            if process:

                process_sessions[process].add(
                    session_dir.name
                )

                session_processes.add(process)

                process_event_count[process] += 1

            # ------------------------------------------------
            # Process started
            # ------------------------------------------------

            if event_type == "process_started":

                if process:

                    process_started_count[process] += 1

                    process_case_pairs[process].add(
                        (process, case_id)
                    )

            # ------------------------------------------------
            # Process suspended
            # ------------------------------------------------

            elif event_type == "process_suspended":

                if process:
                    process_suspend_count[process] += 1

            # ------------------------------------------------
            # Process resumed
            # ------------------------------------------------

            elif event_type == "process_resumed":

                if process:
                    process_resume_count[process] += 1

            # ------------------------------------------------
            # Process transition
            # ------------------------------------------------

            elif event_type == "process_switched_out":

                source = event.get("from")
                target = event.get("to")

                if source and target:

                    total_transitions += 1

                    process_outgoing[source] += 1
                    process_incoming[target] += 1

                    predecessor_counts[
                        (target, source)
                    ] += 1

                    successor_counts[
                        (source, target)
                    ] += 1

        if session_processes:
            session_process_sets.append(
                session_processes
            )

    # ========================================================
    # PROCESS LIST
    # ========================================================

    processes = sorted(
        process_sessions.keys()
    )

    print(f"Processes found: {len(processes)}")
    print(f"Total transitions: {total_transitions}")

    # ========================================================
    # PROCESS STATISTICS
    # ========================================================

    process_stats = []

    total_sessions = len(session_dirs)

    for process in processes:

        sessions = len(
            process_sessions[process]
        )

        starts = process_started_count[process]

        unique_executions = len(
            process_case_pairs[process]
        )

        incoming = process_incoming[process]
        outgoing = process_outgoing[process]

        suspend = process_suspend_count[process]
        resume = process_resume_count[process]

        presence_ratio = (
            sessions / total_sessions
            if total_sessions
            else 0
        )

        avg_starts_per_session = (
            starts / sessions
            if sessions
            else 0
        )

        avg_executions_per_session = (
            unique_executions / sessions
            if sessions
            else 0
        )

        process_stats.append({

            "process": process,

            "sessions_present": sessions,

            "session_presence_ratio":
                round(presence_ratio, 4),

            "process_started_events":
                starts,

            "unique_process_case_executions":
                unique_executions,

            "avg_starts_per_session":
                round(
                    avg_starts_per_session,
                    3
                ),

            "avg_executions_per_session":
                round(
                    avg_executions_per_session,
                    3
                ),

            "incoming_transitions":
                incoming,

            "outgoing_transitions":
                outgoing,

            "total_transitions":
                incoming + outgoing,

            "suspended_events":
                suspend,

            "resumed_events":
                resume,

            "gt_events_associated":
                process_event_count[process]
        })

    save_csv(
        PROCESS_STATS_CSV,
        process_stats
    )

    # ========================================================
    # PREDECESSORS
    # ========================================================

    predecessor_rows = []

    for (target, source), count in sorted(
        predecessor_counts.items(),
        key=lambda x: (-x[1], x[0])
    ):

        predecessor_rows.append({

            "process": target,

            "predecessor": source,

            "transition_count": count
        })

    save_csv(
        PREDECESSOR_CSV,
        predecessor_rows
    )

    # ========================================================
    # SUCCESSORS
    # ========================================================

    successor_rows = []

    for (source, target), count in sorted(
        successor_counts.items(),
        key=lambda x: (-x[1], x[0])
    ):

        successor_rows.append({

            "process": source,

            "successor": target,

            "transition_count": count
        })

    save_csv(
        SUCCESSOR_CSV,
        successor_rows
    )

    # ========================================================
    # PROCESS CO-OCCURRENCE
    # ========================================================

    cooccurrence = Counter()

    for process_set in session_process_sets:

        process_list = sorted(process_set)

        for i in range(len(process_list)):

            for j in range(i + 1, len(process_list)):

                pair = (
                    process_list[i],
                    process_list[j]
                )

                cooccurrence[pair] += 1

    cooccurrence_rows = []

    for (p1, p2), count in sorted(
        cooccurrence.items(),
        key=lambda x: (-x[1], x[0])
    ):

        cooccurrence_rows.append({

            "process_1": p1,

            "process_2": p2,

            "sessions_together": count,

            "cooccurrence_ratio":
                round(
                    count / total_sessions,
                    4
                )
        })

    save_csv(
        CO_OCCURRENCE_CSV,
        cooccurrence_rows
    )

    # ========================================================
    # SUMMARY JSON
    # ========================================================

    summary = {

        "dataset": "Dataset A",

        "sessions": total_sessions,

        "processes": processes,

        "total_transitions":
            total_transitions,

        "process_statistics":
            process_stats,

        "top_predecessors": [
            {
                "process": target,
                "predecessor": source,
                "count": count
            }
            for (target, source), count
            in predecessor_counts.most_common(30)
        ],

        "top_successors": [
            {
                "process": source,
                "successor": target,
                "count": count
            }
            for (source, target), count
            in successor_counts.most_common(30)
        ],

        "top_cooccurring_pairs": [
            {
                "process_1": p1,
                "process_2": p2,
                "sessions_together": count
            }
            for (p1, p2), count
            in cooccurrence.most_common(30)
        ]
    }

    with PROCESS_SUMMARY_JSON.open(
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
    # PRINT PROCESS STATISTICS
    # ========================================================

    print("\n" + "=" * 75)
    print("PROCESS STATISTICS")
    print("=" * 75)

    print(
        f"{'P':<5}"
        f"{'Sessions':>10}"
        f"{'Starts':>10}"
        f"{'Exec':>10}"
        f"{'In':>8}"
        f"{'Out':>8}"
        f"{'Suspend':>10}"
        f"{'Resume':>10}"
    )

    print("-" * 75)

    for row in process_stats:

        print(
            f"{row['process']:<5}"
            f"{row['sessions_present']:>10}"
            f"{row['process_started_events']:>10}"
            f"{row['unique_process_case_executions']:>10}"
            f"{row['incoming_transitions']:>8}"
            f"{row['outgoing_transitions']:>8}"
            f"{row['suspended_events']:>10}"
            f"{row['resumed_events']:>10}"
        )

    # ========================================================
    # TOP PREDECESSORS
    # ========================================================

    print("\n" + "=" * 75)
    print("TOP 20 PROCESS TRANSITIONS")
    print("=" * 75)

    for i, ((source, target), count) in enumerate(
        successor_counts.most_common(20),
        start=1
    ):

        print(
            f"{i:2}. "
            f"{source} -> {target} : {count}"
        )

    # ========================================================
    # TOP CO-OCCURRENCE
    # ========================================================

    print("\n" + "=" * 75)
    print("TOP 20 PROCESS CO-OCCURRENCES")
    print("=" * 75)

    for i, ((p1, p2), count) in enumerate(
        cooccurrence.most_common(20),
        start=1
    ):

        print(
            f"{i:2}. "
            f"{p1} + {p2} : "
            f"{count}/{total_sessions} sessions"
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    print("\n" + "=" * 75)
    print("OUTPUT FILES")
    print("=" * 75)

    print(
        f"Process statistics : "
        f"{PROCESS_STATS_CSV}"
    )

    print(
        f"Predecessors       : "
        f"{PREDECESSOR_CSV}"
    )

    print(
        f"Successors         : "
        f"{SUCCESSOR_CSV}"
    )

    print(
        f"Co-occurrence      : "
        f"{CO_OCCURRENCE_CSV}"
    )

    print(
        f"Summary JSON        : "
        f"{PROCESS_SUMMARY_JSON}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()