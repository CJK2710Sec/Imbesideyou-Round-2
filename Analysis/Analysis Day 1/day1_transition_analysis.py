import json
import csv
from pathlib import Path
from collections import Counter, defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path(r"Dataset A\dataset_a")

OUTPUT_DIR = DATASET_ROOT / "day1_transition_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ JSONL
# ============================================================

def read_jsonl(path):

    events = []

    with path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            try:
                event = json.loads(line)

                if isinstance(event, dict):
                    events.append(event)

            except json.JSONDecodeError:

                print(
                    f"WARNING: Invalid JSON "
                    f"{path} line {line_number}"
                )

    return events


# ============================================================
# FIND SESSIONS
# ============================================================

sessions = sorted(
    p
    for p in DATASET_ROOT.rglob("ses_*")
    if p.is_dir()
)


print("=" * 75)
print("DAY 1 - GLOBAL PROCESS TRANSITION ANALYSIS")
print("=" * 75)

print(
    f"Dataset root : {DATASET_ROOT.resolve()}"
)

print(
    f"Sessions     : {len(sessions)}"
)


# ============================================================
# GLOBAL STATISTICS
# ============================================================

transition_counts = Counter()

transition_examples = defaultdict(list)

process_names = {}

session_transition_counts = Counter()

total_switch_events = 0

unknown_transition_events = 0


# ============================================================
# PROCESS EVERY SESSION
# ============================================================

for session_number, session in enumerate(
    sessions,
    1
):

    gt_file = session / "gt.jsonl"

    if not gt_file.exists():

        print(
            f"[{session_number}/{len(sessions)}] "
            f"{session.name} -> NO GT"
        )

        continue


    events = read_jsonl(gt_file)

    session_transitions = 0


    for event in events:

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Your schema uses "event", NOT "event_type"
        # ----------------------------------------------------

        event_type = event.get("event")


        if event_type != "process_switched_out":
            continue


        total_switch_events += 1


        # ----------------------------------------------------
        # Your schema explicitly defines:
        #
        # from
        # to
        # ----------------------------------------------------

        source = event.get("from")

        destination = event.get("to")


        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if not source or not destination:

            unknown_transition_events += 1

            continue


        source = str(source)
        destination = str(destination)


        transition = (
            source,
            destination
        )


        transition_counts[transition] += 1

        session_transitions += 1


        # ----------------------------------------------------
        # Save a few examples
        # ----------------------------------------------------

        if len(
            transition_examples[transition]
        ) < 5:

            transition_examples[
                transition
            ].append({

                "session":
                    session.name,

                "timestamp":
                    event.get(
                        "ts_utc",
                        ""
                    ),

                "from":
                    source,

                "to":
                    destination,

                "process_variant":
                    event.get(
                        "process_variant"
                    ),

            })


    session_transition_counts[
        session.name
    ] = session_transitions


    print(
        f"[{session_number}/{len(sessions)}] "
        f"{session.name} -> "
        f"{session_transitions} transitions"
    )


# ============================================================
# GET PROCESS NAMES
# ============================================================

for session in sessions:

    gt_file = session / "gt.jsonl"

    if not gt_file.exists():
        continue

    events = read_jsonl(gt_file)

    for event in events:

        process_code = event.get(
            "process_code"
        )

        process_name = event.get(
            "process_name"
        )

        if process_code and process_name:

            process_names[
                str(process_code)
            ] = str(process_name)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("1. TRANSITION SUMMARY")
print("=" * 75)

print(
    f"process_switched_out events : "
    f"{total_switch_events}"
)

print(
    f"Valid transitions           : "
    f"{sum(transition_counts.values())}"
)

print(
    f"Unique transitions          : "
    f"{len(transition_counts)}"
)

print(
    f"Missing from/to             : "
    f"{unknown_transition_events}"
)


# ============================================================
# GLOBAL TRANSITION TABLE
# ============================================================

print("\n")
print("=" * 75)
print("2. GLOBAL PROCESS TRANSITIONS")
print("=" * 75)

print(
    f"{'FROM':<8}"
    f"{'TO':<8}"
    f"{'COUNT':>10}"
)

print("-" * 30)


for (
    source,
    destination
), count in transition_counts.most_common():

    print(
        f"{source:<8}"
        f"{destination:<8}"
        f"{count:>10}"
    )


# ============================================================
# TRANSITION TABLE WITH PROCESS NAMES
# ============================================================

print("\n")
print("=" * 75)
print("3. TRANSITIONS WITH PROCESS NAMES")
print("=" * 75)


for (
    source,
    destination
), count in transition_counts.most_common():

    source_name = process_names.get(
        source,
        ""
    )

    destination_name = process_names.get(
        destination,
        ""
    )

    print(
        f"{source} ({source_name})"
        f"  ->  "
        f"{destination} ({destination_name})"
        f"   [{count}]"
    )


# ============================================================
# OUTGOING TRANSITIONS
# ============================================================

print("\n")
print("=" * 75)
print("4. OUTGOING TRANSITIONS BY PROCESS")
print("=" * 75)


outgoing = defaultdict(Counter)


for (
    source,
    destination
), count in transition_counts.items():

    outgoing[source][
        destination
    ] += count


for source in sorted(outgoing):

    print(
        f"\n{source} "
        f"({process_names.get(source, '')})"
    )

    for destination, count in (
        outgoing[source].most_common()
    ):

        print(
            f"    -> "
            f"{destination:<5}"
            f"{count:>6}"
        )


# ============================================================
# INCOMING TRANSITIONS
# ============================================================

print("\n")
print("=" * 75)
print("5. INCOMING TRANSITIONS BY PROCESS")
print("=" * 75)


incoming = defaultdict(Counter)


for (
    source,
    destination
), count in transition_counts.items():

    incoming[destination][
        source
    ] += count


for destination in sorted(incoming):

    print(
        f"\n{destination} "
        f"({process_names.get(destination, '')})"
    )

    for source, count in (
        incoming[destination].most_common()
    ):

        print(
            f"    <- "
            f"{source:<5}"
            f"{count:>6}"
        )


# ============================================================
# TOP TRANSITIONS
# ============================================================

print("\n")
print("=" * 75)
print("6. TOP 20 TRANSITIONS")
print("=" * 75)


for rank, (
    (source, destination),
    count
) in enumerate(
    transition_counts.most_common(20),
    1
):

    print(
        f"{rank:>2}. "
        f"{source} -> {destination}"
        f"   {count}"
    )


# ============================================================
# SESSION TRANSITIONS
# ============================================================

print("\n")
print("=" * 75)
print("7. SESSIONS WITH MOST TRANSITIONS")
print("=" * 75)


for session, count in (
    session_transition_counts
    .most_common(20)
):

    print(
        f"{count:>4} transitions | "
        f"{session}"
    )


# ============================================================
# SAVE JSON
# ============================================================

json_report = {

    "session_count":
        len(sessions),

    "process_switched_out_events":
        total_switch_events,

    "valid_transitions":
        sum(transition_counts.values()),

    "unique_transitions":
        len(transition_counts),

    "missing_from_to":
        unknown_transition_events,

    "process_names":
        process_names,

    "transitions": {

        f"{source}->{destination}":
            count

        for (
            source,
            destination
        ), count
        in transition_counts.items()
    },

    "transition_examples": {

        f"{source}->{destination}":
            examples

        for (
            source,
            destination
        ), examples
        in transition_examples.items()
    },

    "session_transition_counts":
        dict(session_transition_counts)
}


json_path = (
    OUTPUT_DIR /
    "transition_analysis.json"
)


with json_path.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        json_report,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    OUTPUT_DIR /
    "process_transitions.csv"
)


with csv_path.open(
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "from_process",
        "from_process_name",
        "to_process",
        "to_process_name",
        "count"
    ])


    for (
        source,
        destination
    ), count in transition_counts.most_common():

        writer.writerow([
            source,
            process_names.get(
                source,
                ""
            ),
            destination,
            process_names.get(
                destination,
                ""
            ),
            count
        ])


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 75)
print("TRANSITION ANALYSIS COMPLETE")
print("=" * 75)

print(
    f"Sessions analyzed : {len(sessions)}"
)

print(
    f"Switch events     : {total_switch_events}"
)

print(
    f"Transitions       : "
    f"{sum(transition_counts.values())}"
)

print(
    f"Unique transitions: "
    f"{len(transition_counts)}"
)

print(
    f"Missing from/to   : "
    f"{unknown_transition_events}"
)

print("\nOutput:")

print(
    f"  {json_path}"
)

print(
    f"  {csv_path}"
)

print("=" * 75)