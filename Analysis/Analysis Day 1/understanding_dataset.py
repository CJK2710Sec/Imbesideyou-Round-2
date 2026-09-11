from pathlib import Path
import json

session = Path("Dataset A\dataset_a\ses_20260630-124826-CHAITANYA0BCF")
gt = session / "gt.jsonl"

interesting = {
    "process_started",
    "process_switched_out",
    "process_suspended",
    "process_resumed",
    "session_ended"
}

with gt.open("r", encoding="utf-8") as f:

    for line in f:

        if not line.strip():
            continue

        record = json.loads(line)

        if record.get("event") not in interesting:
            continue

        print(
            record.get("ts_utc"),
            "|",
            record.get("event"),
            "| process:",
            record.get("current_process"),
            "| code:",
            record.get("process_code"),
            "| name:",
            record.get("process_name"),
            "| case:",
            record.get("case_id"),
            "| from:",
            record.get("from"),
            "| to:",
            record.get("to"),
            "| split:",
            record.get("split_id"),
        )