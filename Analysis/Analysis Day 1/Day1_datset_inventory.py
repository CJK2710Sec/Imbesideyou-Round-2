#!/usr/bin/env python3
"""
Day 1 - Dataset Inventory

Purpose
-------
Build a complete inventory of the dataset before doing any process-signature
extraction or ML work.

The script:
1. Finds every session directory (ses_*)
2. Finds every chunk directory (chunk_*)
3. Counts raw JSONL events
4. Counts event types
5. Checks for gt.jsonl
6. Counts GT process types / process codes
7. Calculates session duration when timestamps are available
8. Counts screenshots
9. Writes CSV + JSON reports

IMPORTANT
---------
- A chunk is NOT treated as a session.
- Ground truth is used only for inventory/statistics at this stage.
- The script does not infer process signatures.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


# ============================================================
# CONFIGURATION
# ============================================================

# Change this to your dataset root.
# Example:
# DATASET_ROOT = Path(r"C:\Users\Sahil\Desktop\ProcMine\dataset_a")
DATASET_ROOT = Path(r"Dataset A\dataset_a")

# Output directory will be created inside DATASET_ROOT.
OUTPUT_DIR = DATASET_ROOT / "day1_inventory"


# ============================================================
# HELPERS
# ============================================================

SESSION_PATTERN = re.compile(r"^ses_")
CHUNK_PATTERN = re.compile(r"^chunk_")


def parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp safely."""
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip()

    try:
        # Handles timestamps ending in Z as UTC.
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def get_timestamp(event: dict[str, Any]) -> Optional[datetime]:
    """Try common timestamp field names."""
    for key in (
        "timestamp",
        "time",
        "ts",
        "datetime",
        "created_at",
        "event_time",
    ):
        dt = parse_timestamp(event.get(key))
        if dt is not None:
            return dt

    return None


def get_event_type(event: dict[str, Any]) -> str:
    """Try common event-type field names."""
    for key in (
        "event_type",
        "event",
        "type",
        "action",
        "name",
    ):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return "<unknown>"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield valid JSON objects from a JSONL file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"[WARNING] Invalid JSON: {path} "
                        f"(line {line_no})"
                    )
                    continue

                if isinstance(obj, dict):
                    yield obj
                else:
                    print(
                        f"[WARNING] JSON object expected: {path} "
                        f"(line {line_no})"
                    )

    except OSError as exc:
        print(f"[WARNING] Could not read {path}: {exc}")


def find_session_dirs(root: Path) -> list[Path]:
    """Find all session directories recursively."""
    sessions = []

    for path in root.rglob("*"):
        if path.is_dir() and SESSION_PATTERN.match(path.name):
            sessions.append(path)

    return sorted(set(sessions))


def find_chunk_dirs(session_dir: Path) -> list[Path]:
    """Find chunk directories belonging to one session."""
    return sorted(
        p for p in session_dir.iterdir()
        if p.is_dir() and CHUNK_PATTERN.match(p.name)
    )


def find_gt_file(session_dir: Path) -> Optional[Path]:
    """Find gt.jsonl directly under the session first, then recursively."""
    direct = session_dir / "gt.jsonl"
    if direct.is_file():
        return direct

    matches = sorted(session_dir.rglob("gt.jsonl"))
    return matches[0] if matches else None


def find_raw_jsonl_files(session_dir: Path, gt_file: Optional[Path]) -> list[Path]:
    """
    Find JSONL files belonging to raw event data.

    gt.jsonl is excluded.
    """
    files = []

    for path in session_dir.rglob("*.jsonl"):
        if gt_file is not None and path.resolve() == gt_file.resolve():
            continue
        files.append(path)

    return sorted(files)


def count_raw_events(
    raw_files: list[Path],
) -> tuple[int, Counter, int, Optional[datetime], Optional[datetime]]:
    """Count raw events, event types, screenshots and timestamps."""
    total = 0
    event_types = Counter()
    screenshots = 0
    min_time = None
    max_time = None

    for path in raw_files:
        for event in iter_jsonl(path):
            total += 1

            event_type = get_event_type(event)
            event_types[event_type] += 1

            if "screenshot" in event_type.lower():
                screenshots += 1

            dt = get_timestamp(event)

            if dt is not None:
                if min_time is None or dt < min_time:
                    min_time = dt

                if max_time is None or dt > max_time:
                    max_time = dt

    return total, event_types, screenshots, min_time, max_time


def extract_process_info(event: dict[str, Any]) -> tuple[str, str, str]:
    """Extract process code/name/variant from a GT event."""
    process = event.get("process")

    if isinstance(process, dict):
        code = (
            process.get("code")
            or process.get("id")
            or process.get("process_code")
            or ""
        )
        name = (
            process.get("name")
            or process.get("label")
            or process.get("process_name")
            or ""
        )
        variant = process.get("variant") or ""
        return str(code), str(name), str(variant)

    code = (
        event.get("process_code")
        or event.get("code")
        or event.get("process_id")
        or ""
    )

    name = (
        event.get("process_name")
        or event.get("name")
        or ""
    )

    variant = event.get("variant") or ""

    # Sometimes process is simply a string such as "A".
    if isinstance(process, str):
        if not code:
            code = process

    return str(code), str(name), str(variant)


def count_gt_events(
    gt_file: Optional[Path],
) -> tuple[int, Counter, Counter, Counter, Optional[datetime], Optional[datetime]]:
    """
    Count GT event types and process information.

    Returns:
        total_gt_events
        gt_event_types
        process_codes
        process_names
        min_time
        max_time
    """
    if gt_file is None:
        return 0, Counter(), Counter(), Counter(), None, None

    total = 0
    event_types = Counter()
    process_codes = Counter()
    process_names = Counter()

    min_time = None
    max_time = None

    for event in iter_jsonl(gt_file):
        total += 1

        event_type = get_event_type(event)
        event_types[event_type] += 1

        code, name, _variant = extract_process_info(event)

        if code:
            process_codes[code] += 1

        if name:
            process_names[name] += 1

        dt = get_timestamp(event)

        if dt is not None:
            if min_time is None or dt < min_time:
                min_time = dt

            if max_time is None or dt > max_time:
                max_time = dt

    return (
        total,
        event_types,
        process_codes,
        process_names,
        min_time,
        max_time,
    )


def format_dt(dt: Optional[datetime]) -> str:
    return dt.isoformat() if dt else ""


def duration_seconds(
    start: Optional[datetime],
    end: Optional[datetime],
) -> Optional[float]:
    if start is None or end is None:
        return None

    return round((end - start).total_seconds(), 3)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Counter):
        return dict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ============================================================
# MAIN INVENTORY
# ============================================================

def build_inventory(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sessions = find_session_dirs(root)

    if not sessions:
        print()
        print("[ERROR] No session directories found.")
        print("Expected directories whose names start with: ses_")
        print(f"Dataset root: {root.resolve()}")
        print()
        return [], {}

    print("=" * 70)
    print("DAY 1 - DATASET INVENTORY")
    print("=" * 70)
    print(f"Dataset root : {root.resolve()}")
    print(f"Sessions found: {len(sessions)}")
    print("=" * 70)

    rows = []

    overall_raw_events = 0
    overall_gt_events = 0
    overall_screenshots = 0

    overall_raw_event_types = Counter()
    overall_gt_event_types = Counter()
    overall_process_codes = Counter()
    overall_process_names = Counter()

    sessions_with_gt = 0
    sessions_without_gt = 0

    for index, session_dir in enumerate(sessions, start=1):
        print()
        print(f"[{index}/{len(sessions)}] {session_dir.name}")

        chunks = find_chunk_dirs(session_dir)
        gt_file = find_gt_file(session_dir)
        raw_files = find_raw_jsonl_files(session_dir, gt_file)

        (
            raw_count,
            raw_types,
            screenshot_count,
            raw_start,
            raw_end,
        ) = count_raw_events(raw_files)

        (
            gt_count,
            gt_types,
            process_codes,
            process_names,
            gt_start,
            gt_end,
        ) = count_gt_events(gt_file)

        # Prefer raw timestamps for session duration.
        session_start = raw_start or gt_start
        session_end = raw_end or gt_end

        duration = duration_seconds(session_start, session_end)

        has_gt = gt_file is not None

        if has_gt:
            sessions_with_gt += 1
        else:
            sessions_without_gt += 1

        overall_raw_events += raw_count
        overall_gt_events += gt_count
        overall_screenshots += screenshot_count

        overall_raw_event_types.update(raw_types)
        overall_gt_event_types.update(gt_types)
        overall_process_codes.update(process_codes)
        overall_process_names.update(process_names)

        row = {
            "session": session_dir.name,
            "session_path": str(session_dir.resolve()),
            "chunk_count": len(chunks),
            "raw_jsonl_count": len(raw_files),
            "raw_event_count": raw_count,
            "screenshot_event_count": screenshot_count,
            "gt_exists": has_gt,
            "gt_path": str(gt_file.resolve()) if gt_file else "",
            "gt_event_count": gt_count,
            "session_start": format_dt(session_start),
            "session_end": format_dt(session_end),
            "duration_seconds": duration,
            "raw_event_types": dict(raw_types),
            "gt_event_types": dict(gt_types),
            "gt_process_codes": dict(process_codes),
            "gt_process_names": dict(process_names),
        }

        rows.append(row)

        print(f"  chunks       : {len(chunks)}")
        print(f"  raw JSONL    : {len(raw_files)}")
        print(f"  raw events   : {raw_count}")
        print(f"  screenshots  : {screenshot_count}")
        print(f"  GT exists   : {'YES' if has_gt else 'NO'}")
        print(f"  GT events    : {gt_count}")

        if duration is not None:
            print(f"  duration     : {duration:.3f} sec")

    summary = {
        "dataset_root": str(root.resolve()),
        "session_count": len(sessions),
        "sessions_with_gt": sessions_with_gt,
        "sessions_without_gt": sessions_without_gt,
        "total_raw_events": overall_raw_events,
        "total_gt_events": overall_gt_events,
        "total_screenshot_events": overall_screenshots,
        "overall_raw_event_types": dict(
            overall_raw_event_types.most_common()
        ),
        "overall_gt_event_types": dict(
            overall_gt_event_types.most_common()
        ),
        "overall_gt_process_codes": dict(
            overall_process_codes.most_common()
        ),
        "overall_gt_process_names": dict(
            overall_process_names.most_common()
        ),
    }

    return rows, summary


# ============================================================
# REPORT WRITERS
# ============================================================

def write_session_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fieldnames = [
        "session",
        "session_path",
        "chunk_count",
        "raw_jsonl_count",
        "raw_event_count",
        "screenshot_event_count",
        "gt_exists",
        "gt_path",
        "gt_event_count",
        "session_start",
        "session_end",
        "duration_seconds",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in fieldnames}
            )


def write_json_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    report = {
        "summary": summary,
        "sessions": rows,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )


def write_text_report(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("DAY 1 - DATASET INVENTORY REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Dataset root       : {summary['dataset_root']}\n")
        f.write(f"Sessions           : {summary['session_count']}\n")
        f.write(f"Sessions with GT   : {summary['sessions_with_gt']}\n")
        f.write(f"Sessions without GT: {summary['sessions_without_gt']}\n")
        f.write(f"Total raw events   : {summary['total_raw_events']}\n")
        f.write(f"Total GT events    : {summary['total_gt_events']}\n")
        f.write(
            f"Total screenshots : "
            f"{summary['total_screenshot_events']}\n"
        )

        f.write("\n")
        f.write("-" * 70 + "\n")
        f.write("OVERALL RAW EVENT TYPES\n")
        f.write("-" * 70 + "\n")

        for event_type, count in summary["overall_raw_event_types"].items():
            f.write(f"{event_type:35s} {count:10d}\n")

        f.write("\n")
        f.write("-" * 70 + "\n")
        f.write("OVERALL GT EVENT TYPES\n")
        f.write("-" * 70 + "\n")

        for event_type, count in summary["overall_gt_event_types"].items():
            f.write(f"{event_type:35s} {count:10d}\n")

        f.write("\n")
        f.write("-" * 70 + "\n")
        f.write("OVERALL GT PROCESS CODES\n")
        f.write("-" * 70 + "\n")

        for code, count in summary["overall_gt_process_codes"].items():
            f.write(f"{code:35s} {count:10d}\n")

        f.write("\n")
        f.write("-" * 70 + "\n")
        f.write("OVERALL GT PROCESS NAMES\n")
        f.write("-" * 70 + "\n")

        for name, count in summary["overall_gt_process_names"].items():
            f.write(f"{name:35s} {count:10d}\n")

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("SESSION SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        for row in rows:
            f.write(f"Session: {row['session']}\n")
            f.write(f"  chunks        : {row['chunk_count']}\n")
            f.write(f"  raw JSONL     : {row['raw_jsonl_count']}\n")
            f.write(f"  raw events    : {row['raw_event_count']}\n")
            f.write(
                f"  screenshots   : "
                f"{row['screenshot_event_count']}\n"
            )
            f.write(f"  GT exists     : {row['gt_exists']}\n")
            f.write(f"  GT events     : {row['gt_event_count']}\n")
            f.write(
                f"  duration (s)  : "
                f"{row['duration_seconds']}\n"
            )
            f.write("\n")


# ============================================================
# CONSOLE SUMMARY
# ============================================================

def print_final_summary(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    print()
    print("=" * 70)
    print("INVENTORY COMPLETE")
    print("=" * 70)

    print(f"Sessions found       : {summary['session_count']}")
    print(f"Sessions with GT     : {summary['sessions_with_gt']}")
    print(f"Sessions without GT  : {summary['sessions_without_gt']}")
    print(f"Total raw events     : {summary['total_raw_events']}")
    print(f"Total GT events      : {summary['total_gt_events']}")
    print(
        f"Total screenshots    : "
        f"{summary['total_screenshot_events']}"
    )

    print()
    print("Top raw event types:")
    for event_type, count in list(
        summary["overall_raw_event_types"].items()
    )[:20]:
        print(f"  {event_type:35s} {count}")

    print()
    print("GT event types:")
    for event_type, count in summary["overall_gt_event_types"].items():
        print(f"  {event_type:35s} {count}")

    print()
    print("GT process codes:")
    for code, count in summary["overall_gt_process_codes"].items():
        print(f"  {code:35s} {count}")

    print()
    print("Reports written to:")
    print(f"  {OUTPUT_DIR / 'session_inventory.csv'}")
    print(f"  {OUTPUT_DIR / 'dataset_inventory.json'}")
    print(f"  {OUTPUT_DIR / 'dataset_inventory.txt'}")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    root = DATASET_ROOT.resolve()

    if not root.exists():
        print(f"[ERROR] Dataset root does not exist: {root}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, summary = build_inventory(root)

    if not rows:
        return

    write_session_csv(
        rows,
        OUTPUT_DIR / "session_inventory.csv",
    )

    write_json_report(
        rows,
        summary,
        OUTPUT_DIR / "dataset_inventory.json",
    )

    write_text_report(
        rows,
        summary,
        OUTPUT_DIR / "dataset_inventory.txt",
    )

    print_final_summary(rows, summary)


if __name__ == "__main__":
    main()
