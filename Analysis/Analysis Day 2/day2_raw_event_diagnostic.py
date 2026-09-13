from pathlib import Path
import json


DATASET_ROOT = Path(r"Dataset A\dataset_a")


print("=" * 70)
print("DAY 2 — RAW EVENT SCHEMA DIAGNOSTIC")
print("=" * 70)


# Find first actual events.jsonl
events_files = sorted(
    DATASET_ROOT.rglob("events.jsonl")
)


print(
    f"\nTotal events.jsonl files found: "
    f"{len(events_files)}"
)


if not events_files:
    raise RuntimeError(
        "No events.jsonl files found."
    )


events_file = events_files[0]


print("\nInspecting:")
print(events_file)


print("\n" + "-" * 70)
print("FIRST 5 RAW EVENTS")
print("-" * 70)


with open(
    events_file,
    "r",
    encoding="utf-8"
) as f:

    for i in range(5):

        line = f.readline()

        if not line:
            break

        print(
            f"\nEVENT {i + 1}"
        )

        print(
            line[:5000]
        )


print("\n" + "-" * 70)
print("PARSED EVENT STRUCTURE")
print("-" * 70)


with open(
    events_file,
    "r",
    encoding="utf-8"
) as f:

    line = f.readline()


record = json.loads(line)


print(
    "\nTop-level keys:"
)

for key in record.keys():

    print(
        f"  {key}: "
        f"{type(record[key]).__name__}"
    )


print("\nFull first event:")

print(
    json.dumps(
        record,
        indent=2,
        ensure_ascii=False
    )[:10000]
)


print("\n" + "-" * 70)
print("TIMESTAMP SEARCH")
print("-" * 70)


def find_timestamp_fields(obj, path="root"):

    if isinstance(obj, dict):

        for key, value in obj.items():

            current_path = (
                f"{path}.{key}"
            )

            if (
                "time" in key.lower()
                or key.lower() in {
                    "ts",
                    "timestamp"
                }
            ):

                print(
                    f"{current_path} = "
                    f"{value}"
                )

            find_timestamp_fields(
                value,
                current_path
            )

    elif isinstance(obj, list):

        for i, value in enumerate(obj):

            find_timestamp_fields(
                value,
                f"{path}[{i}]"
            )


find_timestamp_fields(record)


print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)