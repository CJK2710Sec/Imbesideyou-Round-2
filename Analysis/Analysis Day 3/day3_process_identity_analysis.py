import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import statistics


# ============================================================
# Utilities
# ============================================================

def load_jsonl(path):
    """Load a JSONL file safely."""
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] Invalid JSON: {path}:{line_no}")

    return records


def parse_time(value):
    """Convert timestamp string to datetime."""
    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except Exception:
        return None


def get_timestamp(record):
    """
    Extract timestamp from different possible event/GT formats.
    """
    for key in [
        "timestamp",
        "time",
        "ts",
        "start",
        "end",
        "event_time"
    ]:
        if key in record:
            return record[key]

    return None


def safe_name(value):
    """Convert arbitrary value into a printable string."""
    if value is None:
        return "UNKNOWN"

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


# ============================================================
# GT interpretation
# ============================================================

def find_gt_files(dataset_root):
    """
    Find every gt.jsonl file inside Dataset A.
    """
    return sorted(dataset_root.rglob("gt.jsonl"))


def get_process_name(record):
    """
    Try the common GT fields used by the dataset.

    We deliberately keep this flexible because GT records can
    represent different process lifecycle events.
    """

    possible_fields = [
        "current_process",
        "process",
        "process_name",
        "process_code",
        "process_family"
    ]

    for field in possible_fields:
        value = record.get(field)

        if value is not None and value != "":
            return safe_name(value)

    return "UNKNOWN_PROCESS"


def get_variant(record):
    """
    Extract process variant if available.
    """
    possible_fields = [
        "process_variant",
        "variant"
    ]

    for field in possible_fields:
        value = record.get(field)

        if value is not None and value != "":
            return safe_name(value)

    return "UNKNOWN_VARIANT"


def get_event_type(record):
    """
    GT lifecycle event type.
    """
    for field in ["event", "event_type", "type", "action"]:
        if field in record:
            return safe_name(record[field])

    return "UNKNOWN_EVENT"


# ============================================================
# GT process statistics
# ============================================================

def analyze_gt_file(gt_path):
    """
    Analyze one session's GT file.
    """

    records = load_jsonl(gt_path)

    session_dir = gt_path.parent
    session_id = session_dir.name

    process_stats = defaultdict(lambda: {
        "events": 0,
        "started": 0,
        "suspended": 0,
        "resumed": 0,
        "switched_out": 0,
        "variants": Counter(),
        "event_types": Counter(),
        "timestamps": []
    })

    all_events = []

    for record in records:

        process = get_process_name(record)
        variant = get_variant(record)
        event_type = get_event_type(record)

        timestamp = get_timestamp(record)

        stats = process_stats[process]

        stats["events"] += 1
        stats["event_types"][event_type] += 1

        if variant != "UNKNOWN_VARIANT":
            stats["variants"][variant] += 1

        if timestamp:
            parsed = parse_time(timestamp)

            if parsed:
                stats["timestamps"].append(parsed)

        event_lower = event_type.lower()

        if "process_started" in event_lower:
            stats["started"] += 1

        elif "process_suspended" in event_lower:
            stats["suspended"] += 1

        elif "process_resumed" in event_lower:
            stats["resumed"] += 1

        elif "process_switched_out" in event_lower:
            stats["switched_out"] += 1

        all_events.append({
            "timestamp": timestamp,
            "process": process,
            "variant": variant,
            "event_type": event_type
        })

    return {
        "session_id": session_id,
        "gt_file": str(gt_path),
        "records": records,
        "process_stats": process_stats,
        "all_events": all_events
    }


# ============================================================
# Process execution reconstruction
# ============================================================

def reconstruct_executions(records):
    """
    Reconstruct approximate process executions from GT lifecycle
    events.

    This is deliberately conservative.

    We do NOT assume that process_switched_out means the process
    permanently ended because Day 1 established that suspension
    and interleaving exist.
    """

    executions = []

    active = {}

    execution_counter = defaultdict(int)

    for record in records:

        process = get_process_name(record)
        variant = get_variant(record)
        event_type = get_event_type(record)
        timestamp = get_timestamp(record)

        event_lower = event_type.lower()

        # ----------------------------------------------------
        # Process started
        # ----------------------------------------------------

        if "process_started" in event_lower:

            execution_counter[process] += 1

            execution_id = (
                f"{process}#{execution_counter[process]}"
            )

            active[process] = {
                "execution_id": execution_id,
                "process": process,
                "variant": variant,
                "start": timestamp,
                "end": None,
                "suspended": False,
                "suspension_count": 0,
                "resume_count": 0,
                "events": 1
            }

        # ----------------------------------------------------
        # Process resumed
        # ----------------------------------------------------

        elif "process_resumed" in event_lower:

            if process in active:

                active[process]["suspended"] = False
                active[process]["resume_count"] += 1
                active[process]["events"] += 1

        # ----------------------------------------------------
        # Process suspended
        # ----------------------------------------------------

        elif "process_suspended" in event_lower:

            if process in active:

                active[process]["suspended"] = True
                active[process]["suspension_count"] += 1
                active[process]["events"] += 1

        # ----------------------------------------------------
        # Process switched out
        # ----------------------------------------------------

        elif "process_switched_out" in event_lower:

            if process in active:

                active[process]["end"] = timestamp
                active[process]["events"] += 1

                executions.append(active[process])

                del active[process]

        else:

            if process in active:
                active[process]["events"] += 1

    # --------------------------------------------------------
    # Close remaining active processes
    # --------------------------------------------------------

    for process, execution in active.items():

        execution["end"] = None
        executions.append(execution)

    return executions


# ============================================================
# Execution statistics
# ============================================================

def execution_duration(execution):

    start = parse_time(execution.get("start"))
    end = parse_time(execution.get("end"))

    if start is None or end is None:
        return None

    return (end - start).total_seconds()


def summarize_executions(all_executions):

    summary = defaultdict(list)

    for execution in all_executions:

        process = execution["process"]

        duration = execution_duration(execution)

        if duration is not None:
            summary[process].append(duration)

    result = {}

    for process, durations in summary.items():

        result[process] = {
            "execution_count": len(durations),
            "mean_duration_sec": statistics.mean(durations),
            "median_duration_sec": statistics.median(durations),
            "min_duration_sec": min(durations),
            "max_duration_sec": max(durations)
        }

    return result


# ============================================================
# Interleaving analysis
# ============================================================

def analyze_interleaving(all_events):

    transitions = Counter()

    previous_process = None

    for event in all_events:

        process = event["process"]

        if process == "UNKNOWN_PROCESS":
            continue

        if previous_process is not None and process != previous_process:

            transitions[
                (previous_process, process)
            ] += 1

        previous_process = process

    return transitions


def detect_return_patterns(all_events):

    """
    Detect patterns such as:

        A -> B -> A

    which are important because they can indicate interruption
    / interleaving rather than two independent executions.
    """

    patterns = Counter()

    processes = [
        event["process"]
        for event in all_events
        if event["process"] != "UNKNOWN_PROCESS"
    ]

    for i in range(len(processes) - 2):

        a = processes[i]
        b = processes[i + 1]
        c = processes[i + 2]

        if a == c and a != b:

            patterns[(a, b, a)] += 1

    return patterns


# ============================================================
# Report generation
# ============================================================

def write_json(data, path):

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            default=str
        )


def write_report(
    output_path,
    session_results,
    global_process_stats,
    execution_summary,
    transitions,
    return_patterns
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("# Day 3 — Phase 6.1 Process Identity Analysis\n\n")

        # ----------------------------------------------------
        # Dataset overview
        # ----------------------------------------------------

        f.write("## 1. Dataset Overview\n\n")

        f.write(
            f"- Sessions analyzed: {len(session_results)}\n"
        )

        f.write(
            f"- Process families discovered: "
            f"{len(global_process_stats)}\n"
        )

        f.write(
            f"- Reconstructed executions: "
            f"{sum(global_process_stats[p]['started'] for p in global_process_stats)}\n"
        )

        f.write("\n")

        # ----------------------------------------------------
        # Process families
        # ----------------------------------------------------

        f.write("## 2. Process Families\n\n")

        f.write(
            "| Process | GT events | Started | Suspended | "
            "Resumed | Switched out | Variants |\n"
        )

        f.write(
            "|---|---:|---:|---:|---:|---:|---:|\n"
        )

        for process in sorted(global_process_stats):

            stats = global_process_stats[process]

            variants = len(stats["variants"])

            f.write(
                f"| `{process}` | "
                f"{stats['events']} | "
                f"{stats['started']} | "
                f"{stats['suspended']} | "
                f"{stats['resumed']} | "
                f"{stats['switched_out']} | "
                f"{variants} |\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Execution duration
        # ----------------------------------------------------

        f.write("## 3. Execution Duration\n\n")

        f.write(
            "| Process | Executions | Mean (s) | "
            "Median (s) | Min (s) | Max (s) |\n"
        )

        f.write(
            "|---|---:|---:|---:|---:|---:|\n"
        )

        for process in sorted(execution_summary):

            s = execution_summary[process]

            f.write(
                f"| `{process}` | "
                f"{s['execution_count']} | "
                f"{s['mean_duration_sec']:.2f} | "
                f"{s['median_duration_sec']:.2f} | "
                f"{s['min_duration_sec']:.2f} | "
                f"{s['max_duration_sec']:.2f} |\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Interleaving
        # ----------------------------------------------------

        f.write("## 4. Process Transition / Interleaving\n\n")

        f.write(
            "| From | To | Count |\n"
        )

        f.write(
            "|---|---|---:|\n"
        )

        for (a, b), count in transitions.most_common():

            f.write(
                f"| `{a}` | `{b}` | {count} |\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # A-B-A patterns
        # ----------------------------------------------------

        f.write("## 5. Return / Interleaving Patterns\n\n")

        f.write(
            "Patterns of the form `A → B → A` are listed below. "
            "These are evidence for possible process interruption "
            "or interleaving and should not automatically be treated "
            "as separate A executions.\n\n"
        )

        f.write(
            "| Pattern | Count |\n"
        )

        f.write(
            "|---|---:|\n"
        )

        for pattern, count in return_patterns.most_common():

            pattern_string = " → ".join(
                f"`{x}`" for x in pattern
            )

            f.write(
                f"| {pattern_string} | {count} |\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Conclusions
        # ----------------------------------------------------

        f.write("## 6. Phase 6.1 Conclusions\n\n")

        f.write(
            "The analysis establishes the Dataset A process "
            "identity landscape before process clustering or "
            "final segment construction.\n\n"
        )

        f.write(
            "The next analysis should determine which raw-event "
            "features are stable within a process family and "
            "which features distinguish different process "
            "families.\n"
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Day 3 Phase 6.1 - Dataset A process identity analysis"
        )
    )

    parser.add_argument(
        "--dataset-a",
        required=True,
        help="Path to dataset_a"
    )

    parser.add_argument(
        "--output",
        default="Outputs/Day 3",
        help="Output directory"
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_a)
    output_root = Path(args.output)

    if not dataset_root.exists():

        raise FileNotFoundError(
            f"Dataset A not found: {dataset_root}"
        )

    print("=" * 60)
    print("DAY 3 — PHASE 6.1")
    print("Process Identity Analysis")
    print("=" * 60)

    # --------------------------------------------------------
    # Find GT files
    # --------------------------------------------------------

    gt_files = find_gt_files(dataset_root)

    print(f"\nFound {len(gt_files)} GT files.")

    if not gt_files:

        raise RuntimeError(
            "No gt.jsonl files found in Dataset A."
        )

    # --------------------------------------------------------
    # Analyze sessions
    # --------------------------------------------------------

    session_results = []

    global_process_stats = defaultdict(lambda: {
        "events": 0,
        "started": 0,
        "suspended": 0,
        "resumed": 0,
        "switched_out": 0,
        "variants": Counter(),
        "event_types": Counter(),
        "timestamps": []
    })

    all_executions = []
    all_events = []

    for index, gt_path in enumerate(gt_files, 1):

        print(
            f"[{index}/{len(gt_files)}] "
            f"{gt_path.parent.name}"
        )

        result = analyze_gt_file(gt_path)

        session_results.append(result)

        # Global process statistics
        for process, stats in result["process_stats"].items():

            global_stats = global_process_stats[process]

            global_stats["events"] += stats["events"]
            global_stats["started"] += stats["started"]
            global_stats["suspended"] += stats["suspended"]
            global_stats["resumed"] += stats["resumed"]
            global_stats["switched_out"] += stats["switched_out"]

            global_stats["variants"].update(
                stats["variants"]
            )

            global_stats["event_types"].update(
                stats["event_types"]
            )

            global_stats["timestamps"].extend(
                stats["timestamps"]
            )

        # Reconstruct executions
        executions = reconstruct_executions(
            result["records"]
        )

        all_executions.extend(executions)
        all_events.extend(result["all_events"])

    # --------------------------------------------------------
    # Analyze executions
    # --------------------------------------------------------

    execution_summary = summarize_executions(
        all_executions
    )

    # --------------------------------------------------------
    # Analyze interleaving
    # --------------------------------------------------------

    transitions = analyze_interleaving(
        all_events
    )

    return_patterns = detect_return_patterns(
        all_events
    )

    # --------------------------------------------------------
    # Prepare JSON-safe process stats
    # --------------------------------------------------------

    process_stats_json = {}

    for process, stats in global_process_stats.items():

        process_stats_json[process] = {
            "events": stats["events"],
            "started": stats["started"],
            "suspended": stats["suspended"],
            "resumed": stats["resumed"],
            "switched_out": stats["switched_out"],
            "variants": dict(stats["variants"]),
            "event_types": dict(stats["event_types"])
        }

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    write_json(
        process_stats_json,
        output_root / "process_identity_statistics.json"
    )

    write_json(
        all_executions,
        output_root / "reconstructed_executions.json"
    )

    transition_json = {
        f"{a} -> {b}": count
        for (a, b), count in transitions.items()
    }

    write_json(
        transition_json,
        output_root / "process_transitions.json"
    )

    return_json = {
        f"{a} -> {b} -> {c}": count
        for (a, b, c), count in return_patterns.items()
    }

    write_json(
        return_json,
        output_root / "return_patterns.json"
    )

    # --------------------------------------------------------
    # Markdown report
    # --------------------------------------------------------

    write_report(
        output_root / "day3_phase6_1_report.md",
        session_results,
        global_process_stats,
        execution_summary,
        transitions,
        return_patterns
    )

    # --------------------------------------------------------
    # Final console summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PHASE 6.1 COMPLETE")
    print("=" * 60)

    print(
        f"Sessions analyzed       : {len(session_results)}"
    )

    print(
        f"Process families        : "
        f"{len(global_process_stats)}"
    )

    print(
        f"Executions reconstructed : "
        f"{len(all_executions)}"
    )

    print(
        f"Process transitions      : "
        f"{len(transitions)}"
    )

    print(
        f"A-B-A return patterns    : "
        f"{len(return_patterns)}"
    )

    print(
        f"\nOutputs written to: "
        f"{output_root}"
    )


if __name__ == "__main__":
    main()