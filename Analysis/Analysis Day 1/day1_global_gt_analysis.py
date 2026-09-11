import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

# CHANGE THIS TO YOUR DATASET ROOT
DATASET_ROOT = Path(r"Dataset A\dataset_a")

OUTPUT_DIR = DATASET_ROOT / "day1_global_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def parse_time(value):
    if not isinstance(value, str):
        return None

    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except Exception:
        return None


def get_timestamp(event):
    for key in [
        "timestamp",
        "time",
        "ts",
        "datetime",
        "event_time",
        "created_at",
    ]:
        if key in event:
            t = parse_time(event[key])
            if t:
                return t

    return None


def get_event_type(event):
    for key in [
        "event_type",
        "event",
        "type",
        "action",
        "name",
    ]:
        value = event.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return "UNKNOWN"


def get_process(event):
    """
    Handles both possible GT structures:

        "process": "A"

    and:

        "process": {
            "code": "A",
            "name": "...",
            ...
        }
    """

    process = event.get("process")

    if isinstance(process, str):
        return process, "", ""

    if isinstance(process, dict):
        code = (
            process.get("code")
            or process.get("id")
            or process.get("process_code")
            or ""
        )

        name = (
            process.get("name")
            or process.get("process_name")
            or process.get("label")
            or ""
        )

        case = (
            process.get("case")
            or process.get("case_id")
            or ""
        )

        return str(code), str(name), str(case)

    code = (
        event.get("process_code")
        or event.get("code")
        or ""
    )

    name = (
        event.get("process_name")
        or event.get("name")
        or ""
    )

    case = (
        event.get("case")
        or event.get("case_id")
        or ""
    )

    return str(code), str(name), str(case)


def read_jsonl(path):
    events = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[WARNING] Invalid JSON "
                    f"{path} line {line_number}"
                )
                continue

            if isinstance(event, dict):
                events.append(event)

    return events


def find_gt(session):
    """
    Find gt.jsonl inside a session.
    """

    direct = session / "gt.jsonl"

    if direct.exists():
        return direct

    matches = list(session.rglob("gt.jsonl"))

    if matches:
        return matches[0]

    return None


# ============================================================
# FIND SESSIONS
# ============================================================

sessions = sorted(
    p
    for p in DATASET_ROOT.rglob("ses_*")
    if p.is_dir()
)

print("=" * 75)
print("DAY 1 - GLOBAL GROUND TRUTH ANALYSIS")
print("=" * 75)

print(f"Dataset root : {DATASET_ROOT.resolve()}")
print(f"Sessions     : {len(sessions)}")

if not sessions:
    print("\nERROR: No ses_* directories found.")
    exit()


# ============================================================
# GLOBAL DATA STRUCTURES
# ============================================================

global_event_types = Counter()

global_process_frequency = Counter()

global_process_names = {}

global_transitions = Counter()

global_process_start_count = Counter()
global_process_switch_count = Counter()
global_process_resume_count = Counter()
global_process_suspend_count = Counter()

global_cases = defaultdict(set)

process_session_count = Counter()

session_processes = {}

session_event_counts = {}

process_transition_examples = defaultdict(list)

all_gt_events = []


# ============================================================
# PROCESS EACH SESSION
# ============================================================

for index, session in enumerate(sessions, 1):

    print(
        f"\n[{index}/{len(sessions)}] "
        f"{session.name}"
    )

    gt_file = find_gt(session)

    if gt_file is None:
        print("  WARNING: gt.jsonl not found")
        continue

    events = read_jsonl(gt_file)

    print(f"  GT events: {len(events)}")

    session_event_counts[session.name] = len(events)

    previous_process = None

    processes_in_session = set()

    for event in events:

        event_type = get_event_type(event)

        global_event_types[event_type] += 1

        code, name, case = get_process(event)

        # ----------------------------------------------------
        # Process statistics
        # ----------------------------------------------------

        if code:

            processes_in_session.add(code)

            global_process_frequency[code] += 1

            if name:
                global_process_names[code] = name

            if case:
                global_cases[code].add(case)

        # ----------------------------------------------------
        # Event-type specific statistics
        # ----------------------------------------------------

        if event_type == "process_started":

            if code:
                global_process_start_count[code] += 1

        elif event_type == "process_switched_out":

            if code:
                global_process_switch_count[code] += 1

            # ------------------------------------------------
            # Transition
            # ------------------------------------------------

            destination = (
                event.get("to")
                or event.get("to_process")
                or event.get("target_process")
            )

            if isinstance(destination, dict):
                destination = (
                    destination.get("code")
                    or destination.get("id")
                )

            if destination:

                destination = str(destination)

                if code:

                    transition = (
                        code,
                        destination
                    )

                    global_transitions[transition] += 1

                    if len(
                        process_transition_examples[transition]
                    ) < 5:

                        process_transition_examples[
                            transition
                        ].append(
                            {
                                "session": session.name,
                                "timestamp": event.get(
                                    "timestamp",
                                    event.get("time", "")
                                ),
                            }
                        )

        elif event_type == "process_resumed":

            if code:
                global_process_resume_count[code] += 1

        elif event_type == "process_suspended":

            if code:
                global_process_suspend_count[code] += 1

        # ----------------------------------------------------
        # Process occurrence
        # ----------------------------------------------------

        if code:

            if previous_process != code:

                previous_process = code

    session_processes[session.name] = sorted(
        processes_in_session
    )

    for process in processes_in_session:
        process_session_count[process] += 1

    all_gt_events.extend(events)


# ============================================================
# PRINT GLOBAL EVENT DISTRIBUTION
# ============================================================

print("\n")
print("=" * 75)
print("1. GLOBAL GT EVENT DISTRIBUTION")
print("=" * 75)

for event_type, count in global_event_types.most_common():

    percentage = (
        count / sum(global_event_types.values())
    ) * 100

    print(
        f"{event_type:35s}"
        f"{count:8d}"
        f"  ({percentage:6.2f}%)"
    )


# ============================================================
# PROCESS FREQUENCY
# ============================================================

print("\n")
print("=" * 75)
print("2. PROCESS FREQUENCY")
print("=" * 75)

print(
    f"{'CODE':<8}"
    f"{'NAME':<35}"
    f"{'EVENTS':>10}"
    f"{'SESSIONS':>10}"
    f"{'CASES':>10}"
)

print("-" * 75)

for process, count in global_process_frequency.most_common():

    name = global_process_names.get(
        process,
        ""
    )

    sessions_count = process_session_count[
        process
    ]

    case_count = len(
        global_cases[process]
    )

    print(
        f"{process:<8}"
        f"{name[:34]:<35}"
        f"{count:>10}"
        f"{sessions_count:>10}"
        f"{case_count:>10}"
    )


# ============================================================
# PROCESS LIFECYCLE
# ============================================================

print("\n")
print("=" * 75)
print("3. PROCESS LIFECYCLE STATISTICS")
print("=" * 75)

print(
    f"{'CODE':<8}"
    f"{'START':>10}"
    f"{'SWITCH':>10}"
    f"{'RESUME':>10}"
    f"{'SUSPEND':>10}"
)

print("-" * 50)

all_process_codes = sorted(
    global_process_frequency.keys()
)

for code in all_process_codes:

    print(
        f"{code:<8}"
        f"{global_process_start_count[code]:>10}"
        f"{global_process_switch_count[code]:>10}"
        f"{global_process_resume_count[code]:>10}"
        f"{global_process_suspend_count[code]:>10}"
    )


# ============================================================
# PROCESS TRANSITIONS
# ============================================================

print("\n")
print("=" * 75)
print("4. GLOBAL PROCESS TRANSITIONS")
print("=" * 75)

print(
    f"{'FROM':<8}"
    f"{'TO':<8}"
    f"{'COUNT':>10}"
)

print("-" * 30)

for (source, destination), count in (
    global_transitions.most_common()
):

    print(
        f"{source:<8}"
        f"{destination:<8}"
        f"{count:>10}"
    )


# ============================================================
# TRANSITION EXAMPLES
# ============================================================

print("\n")
print("=" * 75)
print("5. TRANSITION EXAMPLES")
print("=" * 75)

for (
    source,
    destination
), examples in global_transitions.items():

    key = (
        source,
        destination
    )

    print(
        f"\n{source} -> {destination}"
        f"  count={global_transitions[key]}"
    )

    for example in process_transition_examples[key]:

        print(
            f"    {example['session']} "
            f"{example['timestamp']}"
        )


# ============================================================
# PROCESS CO-OCCURRENCE
# ============================================================

print("\n")
print("=" * 75)
print("6. PROCESS CO-OCCURRENCE")
print("=" * 75)

pair_counts = Counter()

for session, processes in session_processes.items():

    for i in range(len(processes)):

        for j in range(i + 1, len(processes)):

            pair = (
                processes[i],
                processes[j]
            )

            pair_counts[pair] += 1


for (a, b), count in pair_counts.most_common():

    print(
        f"{a} + {b}: "
        f"{count} sessions"
    )


# ============================================================
# SESSION PROCESS COVERAGE
# ============================================================

print("\n")
print("=" * 75)
print("7. SESSION PROCESS COVERAGE")
print("=" * 75)

for process, count in process_session_count.most_common():

    percentage = (
        count / len(sessions)
    ) * 100

    print(
        f"{process:<8}"
        f"{count:>5}/{len(sessions)} sessions"
        f"  ({percentage:6.2f}%)"
    )


# ============================================================
# SESSIONS WITH MOST PROCESSES
# ============================================================

print("\n")
print("=" * 75)
print("8. SESSIONS WITH MOST DISTINCT PROCESSES")
print("=" * 75)

session_ranking = sorted(
    session_processes.items(),
    key=lambda x: len(x[1]),
    reverse=True
)

for session, processes in session_ranking[:20]:

    print(
        f"{len(processes):>3} processes | "
        f"{session}"
    )

    print(
        "    "
        + ", ".join(processes)
    )


# ============================================================
# SAVE JSON REPORT
# ============================================================

report = {

    "dataset": {
        "root": str(DATASET_ROOT.resolve()),
        "session_count": len(sessions),
    },

    "gt_event_types": dict(
        global_event_types
    ),

    "process_frequency": dict(
        global_process_frequency
    ),

    "process_names": global_process_names,

    "process_session_count": dict(
        process_session_count
    ),

    "process_cases": {
        process: sorted(cases)
        for process, cases in global_cases.items()
    },

    "process_lifecycle": {

        process: {
            "started": global_process_start_count[process],
            "switched_out": global_process_switch_count[process],
            "resumed": global_process_resume_count[process],
            "suspended": global_process_suspend_count[process],
        }

        for process in all_process_codes
    },

    "transitions": {

        f"{source}->{destination}": count

        for (source, destination), count
        in global_transitions.items()
    },

    "session_processes": session_processes,

    "session_event_counts": session_event_counts,

    "process_cooccurrence": {

        f"{a}+{b}": count

        for (a, b), count
        in pair_counts.items()
    }
}


json_output = (
    OUTPUT_DIR /
    "global_gt_analysis.json"
)

with json_output.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SAVE TRANSITIONS CSV
# ============================================================

transition_csv = (
    OUTPUT_DIR /
    "process_transitions.csv"
)

with transition_csv.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    import csv

    writer = csv.writer(f)

    writer.writerow([
        "from_process",
        "to_process",
        "count"
    ])

    for (
        source,
        destination
    ), count in global_transitions.most_common():

        writer.writerow([
            source,
            destination,
            count
        ])


# ============================================================
# SAVE PROCESS SUMMARY CSV
# ============================================================

process_csv = (
    OUTPUT_DIR /
    "process_summary.csv"
)

with process_csv.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    import csv

    writer = csv.writer(f)

    writer.writerow([
        "process_code",
        "process_name",
        "event_count",
        "session_count",
        "case_count",
        "started",
        "switched_out",
        "resumed",
        "suspended",
    ])

    for process in all_process_codes:

        writer.writerow([
            process,
            global_process_names.get(
                process,
                ""
            ),
            global_process_frequency[process],
            process_session_count[process],
            len(global_cases[process]),
            global_process_start_count[process],
            global_process_switch_count[process],
            global_process_resume_count[process],
            global_process_suspend_count[process],
        ])


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)

print(
    f"Sessions analyzed : {len(sessions)}"
)

print(
    f"GT events analyzed: "
    f"{len(all_gt_events)}"
)

print(
    f"Distinct processes: "
    f"{len(global_process_frequency)}"
)

print(
    f"Transitions found : "
    f"{len(global_transitions)}"
)

print("\nReports:")

print(
    f"  {json_output}"
)

print(
    f"  {transition_csv}"
)

print(
    f"  {process_csv}"
)

print("=" * 75)