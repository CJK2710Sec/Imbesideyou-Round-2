import csv
from pathlib import Path
from collections import Counter, defaultdict
import json
import re


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(r"Dataset A\dataset_a")

INPUT_DIR = DATASET_ROOT / "day1_transition_context"
INPUT_FILE = INPUT_DIR / "transition_boundary_events.csv"

OUTPUT_DIR = DATASET_ROOT / "day1_transition_signals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 15


# ============================================================
# HELPERS
# ============================================================

def safe_int(value):
    try:
        return int(value)
    except Exception:
        return 0


def clean_text(value, max_len=250):
    if value is None:
        return ""

    value = str(value).strip()

    if not value:
        return ""

    value = re.sub(r"\s+", " ", value)

    if len(value) > max_len:
        value = value[:max_len] + "..."

    return value


def read_csv(path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def top_counter(counter, n=TOP_N):
    return [
        {
            "value": value,
            "count": count
        }
        for value, count in counter.most_common(n)
    ]


def add_counter(counter_dict, key, value):
    value = clean_text(value)

    if value:
        counter_dict[key][value] += 1


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("DAY 1 — TRANSITION SIGNAL ANALYSIS")
print("=" * 70)

print(f"Input: {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

rows = read_csv(INPUT_FILE)

print(f"Boundary raw-event rows: {len(rows)}")


# ============================================================
# STRUCTURES
# ============================================================

# transition -> phase -> signal -> counter
signals = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(Counter)
    )
)

# transition -> number of raw-event rows
transition_rows = Counter()

# transition -> number of sessions
transition_sessions = defaultdict(set)

# overall counters
overall = defaultdict(Counter)

# transition -> phase -> sessions
phase_sessions = defaultdict(
    lambda: defaultdict(set)
)


# ============================================================
# SIGNAL EXTRACTION
# ============================================================

SIGNAL_FIELDS = [
    "event_type",
    "layer",
    "source",
    "app_name",
    "process_name",
    "window_title",
    "browser_domain",
    "browser_path",
    "browser_element",
]


for row in rows:

    from_process = clean_text(row.get("from_process"))
    to_process = clean_text(row.get("to_process"))

    if not from_process or not to_process:
        continue

    transition = f"{from_process}->{to_process}"

    phase = clean_text(row.get("phase"))
    session_id = clean_text(row.get("session_id"))

    transition_rows[transition] += 1

    if session_id:
        transition_sessions[transition].add(session_id)

    if phase and session_id:
        phase_sessions[
            transition
        ][phase].add(session_id)

    # --------------------------------------------------------
    # Collect signals
    # --------------------------------------------------------

    for field in SIGNAL_FIELDS:

        value = clean_text(row.get(field))

        if not value:
            continue

        signals[
            transition
        ][phase][field][value] += 1

        overall[field][value] += 1


# ============================================================
# TRANSITION SUMMARY
# ============================================================

transition_summary = []

for transition in sorted(transition_rows):

    from_process, to_process = transition.split("->", 1)

    transition_summary.append({
        "transition": transition,
        "from_process": from_process,
        "to_process": to_process,

        "raw_event_rows":
            transition_rows[transition],

        "unique_sessions":
            len(transition_sessions[transition]),

        "pre_5_2_sessions":
            len(phase_sessions[transition]["pre_5_2"]),

        "pre_2_0_sessions":
            len(phase_sessions[transition]["pre_2_0"]),

        "post_0_2_sessions":
            len(phase_sessions[transition]["post_0_2"]),

        "post_2_5_sessions":
            len(phase_sessions[transition]["post_2_5"]),
    })


summary_file = OUTPUT_DIR / "transition_signal_summary.csv"

summary_fields = [
    "transition",
    "from_process",
    "to_process",
    "raw_event_rows",
    "unique_sessions",
    "pre_5_2_sessions",
    "pre_2_0_sessions",
    "post_0_2_sessions",
    "post_2_5_sessions",
]


with summary_file.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=summary_fields
    )

    writer.writeheader()
    writer.writerows(transition_summary)


# ============================================================
# DETAILED SIGNAL TABLE
# ============================================================

detail_rows = []

for transition in sorted(signals):

    for phase in [
        "pre_5_2",
        "pre_2_0",
        "post_0_2",
        "post_2_5"
    ]:

        for signal_type in SIGNAL_FIELDS:

            counter = signals[
                transition
            ][phase][signal_type]

            for value, count in counter.most_common(TOP_N):

                detail_rows.append({
                    "transition": transition,
                    "phase": phase,
                    "signal_type": signal_type,
                    "signal_value": value,
                    "count": count,
                    "sessions": len(
                        phase_sessions[
                            transition
                        ][phase]
                    ),
                })


detail_file = OUTPUT_DIR / "transition_signal_details.csv"

detail_fields = [
    "transition",
    "phase",
    "signal_type",
    "signal_value",
    "count",
    "sessions",
]


with detail_file.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=detail_fields
    )

    writer.writeheader()
    writer.writerows(detail_rows)


# ============================================================
# JSON REPORT
# ============================================================

report = {
    "dataset": "Dataset A",
    "transition_count":
        len(transition_rows),

    "boundary_raw_event_rows":
        len(rows),

    "signals": {}
}


for transition in sorted(signals):

    report["signals"][transition] = {}

    for phase in [
        "pre_5_2",
        "pre_2_0",
        "post_0_2",
        "post_2_5"
    ]:

        report["signals"][transition][phase] = {}

        for signal_type in SIGNAL_FIELDS:

            counter = signals[
                transition
            ][phase][signal_type]

            report[
                "signals"
            ][transition][phase][signal_type] = (
                top_counter(counter)
            )


json_file = OUTPUT_DIR / "transition_signal_report.json"

with json_file.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# OVERALL SIGNAL DISTRIBUTION
# ============================================================

overall_rows = []

for signal_type in SIGNAL_FIELDS:

    counter = overall[signal_type]

    for value, count in counter.most_common(TOP_N * 3):

        overall_rows.append({
            "signal_type": signal_type,
            "signal_value": value,
            "count": count,
        })


overall_file = OUTPUT_DIR / "overall_signal_distribution.csv"

with overall_file.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "signal_type",
            "signal_value",
            "count"
        ]
    )

    writer.writeheader()
    writer.writerows(overall_rows)


# ============================================================
# TRANSITION × EVENT TYPE MATRIX
# ============================================================

event_matrix = defaultdict(Counter)

for row in rows:

    from_process = clean_text(row.get("from_process"))
    to_process = clean_text(row.get("to_process"))
    event_type = clean_text(row.get("event_type"))

    if not from_process or not to_process or not event_type:
        continue

    transition = f"{from_process}->{to_process}"

    event_matrix[transition][event_type] += 1


matrix_rows = []

for transition in sorted(event_matrix):

    row = {
        "transition": transition
    }

    for event_type in sorted(
        set(
            event
            for counter in event_matrix.values()
            for event in counter
        )
    ):
        row[event_type] = event_matrix[
            transition
        ].get(event_type, 0)

    matrix_rows.append(row)


all_event_types = sorted(
    set(
        event
        for counter in event_matrix.values()
        for event in counter
    )
)

matrix_file = OUTPUT_DIR / "transition_event_type_matrix.csv"

with matrix_file.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    fieldnames = [
        "transition"
    ] + all_event_types

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in matrix_rows:
        writer.writerow(row)


# ============================================================
# CONSOLE REPORT
# ============================================================

print()
print("=" * 70)
print("SIGNAL ANALYSIS COMPLETE")
print("=" * 70)

print(f"Unique transitions : {len(transition_rows)}")
print(f"Boundary rows      : {len(rows)}")

print()
print("Top transitions by raw-event evidence:")
print("-" * 70)

for transition, count in transition_rows.most_common(20):

    sessions = len(
        transition_sessions[transition]
    )

    print(
        f"{transition:10s} | "
        f"events={count:5d} | "
        f"sessions={sessions:2d}"
    )


# ============================================================
# SHOW EXAMPLES
# ============================================================

print()
print("=" * 70)
print("EXAMPLE SIGNALS")
print("=" * 70)

example_transitions = [
    "A->E",
    "D->F",
    "E->M",
    "M->A",
    "G->J",
]


for transition in example_transitions:

    if transition not in signals:
        continue

    print()
    print(f"### {transition}")

    for phase in [
        "pre_5_2",
        "pre_2_0",
        "post_0_2",
        "post_2_5"
    ]:

        print(f"\n[{phase}]")

        for signal_type in [
            "event_type",
            "app_name",
            "window_title",
            "browser_domain",
            "browser_path",
            "browser_element",
        ]:

            counter = signals[
                transition
            ][phase][signal_type]

            if not counter:
                continue

            top = counter.most_common(5)

            print(f"  {signal_type}:")

            for value, count in top:

                print(
                    f"    {count:4d} | {value}"
                )


print()
print("Output files:")
print(f"  {summary_file}")
print(f"  {detail_file}")
print(f"  {json_file}")
print(f"  {overall_file}")
print(f"  {matrix_file}")

print("=" * 70)