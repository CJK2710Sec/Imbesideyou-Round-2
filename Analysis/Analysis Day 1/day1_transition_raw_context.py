import json
import csv
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path(r"Dataset A\dataset_a")

OUTPUT_DIR = DATASET_ROOT / "day1_transition_context"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Raw-event windows around GT transition timestamp
PHASES = [
    ("pre_5_2", -5000, -2000),
    ("pre_2_0", -2000, 0),
    ("post_0_2", 0, 2000),
    ("post_2_5", 2000, 5000),
]


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
                print(f"[WARN] Invalid JSON: {path} line {line_no}")

    return records


def safe_value(value, max_len=1000):
    """
    Convert nested dict/list/scalar values into a stable string.
    Prevents 'unhashable type: dict' problems.
    """

    if value is None:
        return ""

    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True
            )
        except Exception:
            text = str(value)
    else:
        text = str(value)

    text = text.replace("\n", " ").replace("\r", " ")

    if len(text) > max_len:
        text = text[:max_len] + "..."

    return text


def get_nested(obj, *keys):
    """
    Safely retrieve nested dictionary values.
    """

    current = obj

    for key in keys:
        if not isinstance(current, dict):
            return None

        current = current.get(key)

    return current


def parse_gt_timestamp(ts):
    """
    Convert GT ts_utc into epoch milliseconds.
    Handles ISO timestamps with Z.
    """

    if ts is None:
        return None

    try:
        value = str(ts)

        if value.endswith("Z"):
            value = value[:-1] + "+00:00"

        dt = datetime.fromisoformat(value)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return int(dt.timestamp() * 1000)

    except Exception:
        return None


def normalize_app_name(raw_event):
    """
    Extract application name without relying on process IDs.
    """

    context = raw_event.get("context", {})
    metadata = raw_event.get("metadata", {})
    payload = raw_event.get("payload", {})

    candidates = [
        get_nested(context, "active_app", "name"),
        get_nested(context, "active_app", "app_name"),
        get_nested(context, "application", "name"),
        get_nested(context, "application", "app_name"),

        get_nested(metadata, "app_name"),
        get_nested(metadata, "application"),
        get_nested(metadata, "process_name"),

        get_nested(payload, "app_name"),
        get_nested(payload, "application"),
    ]

    for value in candidates:
        if value:
            return safe_value(value)

    return ""


def normalize_process_name(raw_event):
    """
    Extract process name, deliberately ignoring process ID.
    """

    context = raw_event.get("context", {})
    metadata = raw_event.get("metadata", {})
    payload = raw_event.get("payload", {})

    candidates = [
        get_nested(context, "active_app", "process_name"),
        get_nested(context, "application", "process_name"),
        get_nested(metadata, "process_name"),
        get_nested(payload, "process_name"),
    ]

    for value in candidates:
        if value:
            return safe_value(value)

    return ""


def normalize_window_title(raw_event):
    context = raw_event.get("context", {})
    metadata = raw_event.get("metadata", {})
    payload = raw_event.get("payload", {})

    candidates = [
        get_nested(context, "active_app", "window_title"),
        get_nested(context, "active_window", "title"),
        get_nested(context, "window", "title"),

        get_nested(metadata, "window_title"),
        get_nested(metadata, "title"),

        get_nested(payload, "window_title"),
        get_nested(payload, "title"),
    ]

    for value in candidates:
        if value:
            return safe_value(value)

    return ""


def extract_browser_info(raw_event):
    """
    Try several possible locations because browser information
    can vary between event types.
    """

    context = raw_event.get("context", {})
    payload = raw_event.get("payload", {})
    metadata = raw_event.get("metadata", {})

    url_candidates = [
        get_nested(context, "browser", "url"),
        get_nested(context, "browser", "current_url"),
        get_nested(context, "active_browser", "url"),

        get_nested(payload, "url"),
        get_nested(payload, "browser_url"),

        get_nested(metadata, "url"),
    ]

    url = ""

    for value in url_candidates:
        if value:
            url = safe_value(value)
            break

    domain = ""
    path = ""

    if url:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path

        except Exception:
            pass

    return url, domain, path


def extract_browser_element(raw_event):
    payload = raw_event.get("payload", {})

    candidates = [
        payload.get("element"),
        payload.get("element_attributes"),
        payload.get("target"),
        payload.get("target_element"),
        payload.get("attributes"),
    ]

    for value in candidates:
        if value:
            return safe_value(value)

    # Also inspect common direct browser fields
    browser_fields = {}

    for key in [
        "id",
        "class",
        "tag",
        "tag_name",
        "role",
        "name",
        "text",
        "aria_label",
        "href",
    ]:
        if key in payload:
            browser_fields[key] = payload[key]

    if browser_fields:
        return safe_value(browser_fields)

    return ""


def extract_text(raw_event):
    """
    Extract useful textual context where available.
    """

    context = raw_event.get("context", {})
    payload = raw_event.get("payload", {})
    metadata = raw_event.get("metadata", {})

    candidates = [
        get_nested(context, "extracted_text"),
        get_nested(context, "text"),
        get_nested(context, "ocr_text"),

        get_nested(payload, "text"),
        get_nested(payload, "value"),
        get_nested(payload, "input_text"),

        get_nested(metadata, "extracted_text"),
        get_nested(metadata, "text"),
    ]

    values = []

    for value in candidates:
        if value:
            text = safe_value(value)

            if text and text not in values:
                values.append(text)

    return " | ".join(values)


def normalize_raw_event(raw_event):
    timestamp_ms = raw_event.get("timestamp_ms")

    try:
        timestamp_ms = int(timestamp_ms)
    except Exception:
        timestamp_ms = None

    url, domain, path = extract_browser_info(raw_event)

    return {
        "event_id": safe_value(raw_event.get("event_id")),
        "event_type": safe_value(raw_event.get("event_type")),
        "timestamp_ms": timestamp_ms,
        "timestamp_iso": safe_value(raw_event.get("timestamp_iso")),

        "layer": safe_value(raw_event.get("layer")),
        "source": safe_value(raw_event.get("source")),

        "app_name": normalize_app_name(raw_event),
        "process_name": normalize_process_name(raw_event),
        "window_title": normalize_window_title(raw_event),

        "browser_url": url,
        "browser_domain": domain,
        "browser_path": path,

        "browser_element": extract_browser_element(raw_event),
        "extracted_text": extract_text(raw_event),
    }


# ============================================================
# LOAD SESSION RAW EVENTS
# ============================================================

def load_session_raw_events(session_dir):
    """
    Load every chunk/events.jsonl belonging to a session.
    """

    raw_events = []

    chunk_dirs = sorted(session_dir.glob("chunk_*"))

    for chunk_dir in chunk_dirs:

        events_file = chunk_dir / "events.jsonl"

        if not events_file.exists():
            continue

        events = load_jsonl(events_file)

        for event in events:
            raw_events.append(event)

    raw_events.sort(
        key=lambda x: x.get("timestamp_ms", 0)
        if isinstance(x.get("timestamp_ms"), (int, float))
        else 0
    )

    return raw_events


# ============================================================
# FIND RAW EVENTS AROUND TRANSITION
# ============================================================

def get_boundary_events(raw_events, boundary_ms):
    rows = []

    for raw_event in raw_events:

        timestamp_ms = raw_event.get("timestamp_ms")

        if timestamp_ms is None:
            continue

        try:
            timestamp_ms = int(timestamp_ms)
        except Exception:
            continue

        delta = timestamp_ms - boundary_ms

        phase = None

        for phase_name, start_offset, end_offset in PHASES:

            if start_offset <= delta < end_offset:
                phase = phase_name
                break

        if phase is None:
            continue

        normalized = normalize_raw_event(raw_event)

        normalized["delta_ms"] = delta
        normalized["phase"] = phase

        rows.append(normalized)

    return rows


# ============================================================
# MAIN
# ============================================================

all_boundary_rows = []
summary_counter = Counter()
transition_counter = Counter()

session_count = 0
transition_count = 0
boundary_with_raw_events = 0
boundary_without_raw_events = 0


session_dirs = sorted(
    p for p in DATASET_ROOT.iterdir()
    if p.is_dir() and p.name.startswith("ses_")
)

print("=" * 70)
print("DAY 1 — GT TRANSITION / RAW EVENT CONTEXT ANALYSIS")
print("=" * 70)

print(f"Dataset root : {DATASET_ROOT}")
print(f"Sessions     : {len(session_dirs)}")
print()


for session_dir in session_dirs:

    session_count += 1

    session_id = session_dir.name

    gt_file = session_dir / "gt.jsonl"

    if not gt_file.exists():
        print(f"[WARN] No gt.jsonl: {session_id}")
        continue

    gt_events = load_jsonl(gt_file)

    raw_events = load_session_raw_events(session_dir)

    print(
        f"[{session_count:02d}/{len(session_dirs)}] "
        f"{session_id} | "
        f"GT={len(gt_events)} RAW={len(raw_events)}"
    )

    # --------------------------------------------------------
    # Find explicit GT transitions
    # --------------------------------------------------------

    session_transitions = 0

    for gt in gt_events:

        if gt.get("event") != "process_switched_out":
            continue

        from_process = gt.get("from")
        to_process = gt.get("to")
        ts_utc = gt.get("ts_utc")

        if not from_process or not to_process or not ts_utc:
            continue

        boundary_ms = parse_gt_timestamp(ts_utc)

        if boundary_ms is None:
            continue

        session_transitions += 1
        transition_count += 1

        transition_key = f"{from_process}->{to_process}"

        transition_counter[transition_key] += 1

        boundary_rows = get_boundary_events(
            raw_events,
            boundary_ms
        )

        if boundary_rows:
            boundary_with_raw_events += 1
        else:
            boundary_without_raw_events += 1

        # ----------------------------------------------------
        # Attach GT transition metadata to every raw event
        # ----------------------------------------------------

        for row in boundary_rows:

            row["session_id"] = session_id
            row["from_process"] = safe_value(from_process)
            row["to_process"] = safe_value(to_process)
            row["gt_ts_utc"] = safe_value(ts_utc)
            row["gt_case_id"] = safe_value(gt.get("case_id"))
            row["gt_split_id"] = safe_value(gt.get("split_id"))

            all_boundary_rows.append(row)

            # Aggregation
            summary_counter[
                (
                    transition_key,
                    row["phase"],
                    row["event_type"]
                )
            ] += 1

    print(f"    transitions={session_transitions}")


# ============================================================
# SAVE DETAILED BOUNDARY EVENTS
# ============================================================

detail_file = OUTPUT_DIR / "transition_boundary_events.csv"

detail_fields = [
    "session_id",
    "from_process",
    "to_process",
    "gt_ts_utc",
    "gt_case_id",
    "gt_split_id",

    "phase",
    "delta_ms",

    "event_id",
    "event_type",
    "timestamp_ms",
    "timestamp_iso",

    "layer",
    "source",

    "app_name",
    "process_name",
    "window_title",

    "browser_url",
    "browser_domain",
    "browser_path",

    "browser_element",
    "extracted_text",
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

    for row in all_boundary_rows:
        writer.writerow({
            field: row.get(field, "")
            for field in detail_fields
        })


# ============================================================
# SAVE AGGREGATED SUMMARY
# ============================================================

summary_file = OUTPUT_DIR / "transition_boundary_summary.csv"

summary_rows = []

for (
    transition_key,
    phase,
    event_type
), count in summary_counter.items():

    from_process, to_process = transition_key.split("->", 1)

    summary_rows.append({
        "from_process": from_process,
        "to_process": to_process,
        "phase": phase,
        "event_type": event_type,
        "count": count,
    })


summary_rows.sort(
    key=lambda x: (
        x["from_process"],
        x["to_process"],
        x["phase"],
        -x["count"]
    )
)


summary_fields = [
    "from_process",
    "to_process",
    "phase",
    "event_type",
    "count",
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

    writer.writerows(summary_rows)


# ============================================================
# SAVE TRANSITION SUMMARY
# ============================================================

transition_file = OUTPUT_DIR / "transition_counts.csv"

with transition_file.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "from_process",
        "to_process",
        "count"
    ])

    for transition, count in transition_counter.most_common():

        from_process, to_process = transition.split("->", 1)

        writer.writerow([
            from_process,
            to_process,
            count
        ])


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {
    "sessions": session_count,
    "gt_process_switched_out_events": transition_count,

    "boundaries_with_raw_events": boundary_with_raw_events,
    "boundaries_without_raw_events": boundary_without_raw_events,

    "boundary_raw_coverage_pct": (
        round(
            boundary_with_raw_events /
            transition_count * 100,
            2
        )
        if transition_count
        else 0
    ),

    "raw_events_within_boundary_windows":
        len(all_boundary_rows),

    "phases": {
        "pre_5_2": [-5000, -2000],
        "pre_2_0": [-2000, 0],
        "post_0_2": [0, 2000],
        "post_2_5": [2000, 5000],
    },

    "unique_transitions":
        len(transition_counter),

    "output_files": [
        str(detail_file),
        str(summary_file),
        str(transition_file),
    ],
}


metadata_file = OUTPUT_DIR / "transition_boundary_metadata.json"

with metadata_file.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"Sessions                         : {session_count}")
print(f"GT transitions                   : {transition_count}")
print(f"Unique transitions               : {len(transition_counter)}")
print(
    f"Boundaries with raw events      : "
    f"{boundary_with_raw_events}"
)
print(
    f"Boundaries without raw events   : "
    f"{boundary_without_raw_events}"
)
print(
    f"Boundary raw coverage            : "
    f"{metadata['boundary_raw_coverage_pct']:.2f}%"
)
print(
    f"Raw events around boundaries    : "
    f"{len(all_boundary_rows)}"
)

print()
print("Top transitions:")
print("-" * 50)

for transition, count in transition_counter.most_common(20):
    print(f"{transition:10s} : {count}")

print()
print("Output files:")
print(f"  {detail_file}")
print(f"  {summary_file}")
print(f"  {transition_file}")
print(f"  {metadata_file}")

print("=" * 70)