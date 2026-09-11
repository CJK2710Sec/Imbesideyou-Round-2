from pathlib import Path
import json
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path("Dataset A\dataset_a")

SESSION_NAME = "ses_20260630-124826-CHAITANYA0BCF"

SESSION_PATH = DATASET_ROOT / SESSION_NAME


# Number of raw events to show around each GT process boundary.
CONTEXT_EVENTS = 8


# ============================================================
# HELPERS
# ============================================================

def parse_timestamp(value):
    """Convert ISO timestamp to datetime."""

    if not value:
        return None

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except Exception:
        return None


def load_jsonl(path):
    """Load JSONL safely."""

    records = []

    with path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as e:
                print(
                    f"WARNING: Invalid JSON in {path} "
                    f"line {line_number}: {e}"
                )

    return records


def get_event_time(event):
    """Get timestamp from an event."""

    value = event.get("timestamp_iso")

    if value:
        return parse_timestamp(value)

    return None


def describe_event(event):
    """Create a compact human-readable event description."""

    timestamp = (
        event.get("timestamp_iso")
        or event.get("timestamp_ms")
        or "UNKNOWN"
    )

    event_type = event.get("event_type", "UNKNOWN")
    layer = event.get("layer", "UNKNOWN")

    context = event.get("context")

    if not isinstance(context, dict):
        context = {}

    active_app = context.get("active_app")

    if not isinstance(active_app, dict):
        active_app = {}

    app = (
        active_app.get("app_name")
        or active_app.get("process_name")
        or "?"
    )

    window = active_app.get("window_title") or ""

    browser = context.get("active_browser_tab")

    if not isinstance(browser, dict):
        browser = {}

    browser_title = browser.get("title") or ""
    browser_url = browser.get("url") or ""

    payload = event.get("payload")

    if not isinstance(payload, dict):
        payload = {}

    extra = ""

    # Keyboard
    if event_type == "keystroke":
        key = payload.get("key")
        character = payload.get("character")

        if key:
            extra = f"key={key}"

        if character:
            extra += f" char={character}"

    # Browser click
    elif event_type == "browser_click":
        element = payload.get("element")

        if isinstance(element, dict):
            attributes = element.get("attributes")

            if isinstance(attributes, dict):
                element_id = attributes.get("id")
                placeholder = attributes.get("placeholder")
                class_name = attributes.get("class")

                parts = []

                if element_id:
                    parts.append(f"id={element_id}")

                if placeholder:
                    parts.append(
                        f"placeholder={placeholder}"
                    )

                if class_name:
                    parts.append(
                        f"class={class_name}"
                    )

                extra = " ".join(parts)

    # Browser navigation
    elif event_type == "browser_navigation":

        extra = f"url={browser_url}"

    # Clipboard
    elif event_type == "clipboard_change":

        transition = payload.get("transition")

        if transition:
            extra = f"transition={transition}"

    # Screenshot
    elif event_type == "screenshot_smart":

        file_reference = payload.get("file_reference")

        if isinstance(file_reference, dict):
            filename = file_reference.get("filename")

            if filename:
                extra = f"screenshot={filename}"

    return (
        f"{timestamp} | "
        f"{layer:6} | "
        f"{event_type:25} | "
        f"app={app:20} | "
        f"window={window[:35]:35} | "
        f"{extra}"
    )


# ============================================================
# LOAD ALL RAW EVENTS FROM SESSION
# ============================================================

def load_session_events():

    all_events = []

    chunks = sorted(
        SESSION_PATH.glob("chunk_*")
    )

    print("=" * 110)
    print("SESSION")
    print("=" * 110)

    print(f"Session : {SESSION_NAME}")
    print(f"Chunks  : {len(chunks)}")

    for chunk in chunks:

        events_path = chunk / "events.jsonl"

        if not events_path.exists():
            continue

        events = load_jsonl(events_path)

        print(
            f"  {chunk.name:50} "
            f"{len(events):8,} events"
        )

        all_events.extend(events)

    # Sort across chunks by timestamp.
    all_events.sort(
        key=lambda event: (
            get_event_time(event)
            or datetime.min
        )
    )

    print(f"\nTotal raw events: {len(all_events):,}")

    return all_events


# ============================================================
# LOAD GROUND TRUTH PROCESS BOUNDARIES
# ============================================================

def load_gt():

    gt_path = SESSION_PATH / "gt.jsonl"

    if not gt_path.exists():

        raise FileNotFoundError(
            f"Ground truth not found: {gt_path}"
        )

    records = load_jsonl(gt_path)

    process_events = []

    for record in records:

        if record.get("event") in {
            "process_started",
            "process_switched_out",
            "process_suspended",
            "process_resumed",
            "session_ended",
        }:

            timestamp = parse_timestamp(
                record.get("ts_utc")
            )

            if timestamp:

                process_events.append(
                    (timestamp, record)
                )

    process_events.sort(
        key=lambda x: x[0]
    )

    return process_events


# ============================================================
# FIND RAW EVENTS AROUND GT TIMESTAMP
# ============================================================

def print_raw_context(
    raw_events,
    target_time,
    context_events=8
):

    # Find closest raw event.
    closest_index = min(
        range(len(raw_events)),
        key=lambda i: abs(
            (
                get_event_time(raw_events[i])
                - target_time
            ).total_seconds()
        )
        if get_event_time(raw_events[i])
        else float("inf")
    )

    start = max(
        0,
        closest_index - context_events
    )

    end = min(
        len(raw_events),
        closest_index + context_events + 1
    )

    print("\nRAW EVENT CONTEXT:")

    for i in range(start, end):

        event = raw_events[i]

        marker = (
            "  >>> "
            if i == closest_index
            else "      "
        )

        print(
            marker + describe_event(event)
        )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():

    if not SESSION_PATH.exists():

        print("ERROR:")
        print(
            f"Session not found:\n{SESSION_PATH}"
        )

        print("\nCurrent directory:")
        print(Path.cwd())

        return

    raw_events = load_session_events()

    gt_events = load_gt()

    print("\n")
    print("=" * 110)
    print("GROUND TRUTH → RAW EVENT CORRELATION")
    print("=" * 110)

    print(
        "\nFor every GT process boundary, "
        "we show nearby raw events."
    )

    for timestamp, record in gt_events:

        print("\n")
        print("#" * 110)

        print(
            f"GT TIME   : {record.get('ts_utc')}"
        )

        print(
            f"GT EVENT  : {record.get('event')}"
        )

        print(
            f"PROCESS   : {record.get('current_process')}"
        )

        print(
            f"VARIANT   : {record.get('process_variant')}"
        )

        if record.get("process_code"):
            print(
                f"CODE      : {record.get('process_code')}"
            )

        if record.get("process_name"):
            print(
                f"NAME      : {record.get('process_name')}"
            )

        if record.get("case_id"):
            print(
                f"CASE      : {record.get('case_id')}"
            )

        if record.get("from"):
            print(
                f"FROM      : {record.get('from')}"
            )

        if record.get("to"):
            print(
                f"TO        : {record.get('to')}"
            )

        if record.get("split_id"):
            print(
                f"SPLIT     : {record.get('split_id')}"
            )

        target_time = timestamp

        print_raw_context(
            raw_events,
            target_time,
            CONTEXT_EVENTS
        )

    print("\n")
    print("=" * 110)
    print("END")
    print("=" * 110)


if __name__ == "__main__":
    main()