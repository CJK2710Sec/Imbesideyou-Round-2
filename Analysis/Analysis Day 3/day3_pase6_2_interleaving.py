import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime


# ============================================================
# Helpers
# ============================================================

def parse_ts(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def find_manifest_files(dataset_a):
    """
    Find all gt_manifest.json files recursively.
    """
    return sorted(Path(dataset_a).rglob("gt_manifest.json"))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Extract executions from gt_manifest.json
# ============================================================

def extract_executions(manifest, manifest_path):
    """
    Extract execution-level GT information.

    Each execution represents one logical process execution.
    """

    executions = []

    session_info = manifest.get("session", {})

    session_start = session_info.get("start_ts")
    session_end = session_info.get("end_ts")

    for process in manifest.get("processes", []):

        process_code = process.get("code")
        family_name = process.get("family_name")
        domain = process.get("domain")

        for execution in process.get("executions", []):

            item = {
                "process_code": process_code,
                "family_name": family_name,
                "domain": domain,

                "variant": execution.get("variant"),
                "case_id": execution.get("case_id"),

                "start_ts": execution.get("start_ts"),
                "end_ts": execution.get("end_ts"),

                "start": parse_ts(execution.get("start_ts")),
                "end": parse_ts(execution.get("end_ts")),

                "phase": execution.get("phase"),
                "seq": execution.get("seq"),

                "continues_from_prev":
                    execution.get("continues_from_prev", False),

                "continues_to_next":
                    execution.get("continues_to_next", False),

                "apps": execution.get("apps", []),

                "session_start": session_start,
                "session_end": session_end,

                "source_file": str(manifest_path)
            }

            executions.append(item)

    return executions


# ============================================================
# Sort executions
# ============================================================

def sort_executions(executions):
    """
    Sort all executions by session and start time.
    """

    return sorted(
        executions,
        key=lambda x: (
            x["source_file"],
            x["start"] if x["start"] else datetime.min
        )
    )


# ============================================================
# Process identity statistics
# ============================================================

def process_identity_analysis(executions):

    stats = defaultdict(lambda: {
        "executions": 0,
        "variants": Counter(),
        "domains": Counter(),
        "case_ids": 0,
        "continuations_from_previous": 0,
        "continuations_to_next": 0
    })

    for e in executions:

        code = e["process_code"]

        stats[code]["executions"] += 1

        if e["variant"]:
            stats[code]["variants"][e["variant"]] += 1

        if e["domain"]:
            stats[code]["domains"][e["domain"]] += 1

        if e["case_id"]:
            stats[code]["case_ids"] += 1

        if e["continues_from_prev"]:
            stats[code]["continuations_from_previous"] += 1

        if e["continues_to_next"]:
            stats[code]["continuations_to_next"] += 1

    return stats


# ============================================================
# Detect repeated executions
# ============================================================

def repeated_process_analysis(executions):

    by_session = defaultdict(list)

    for e in executions:
        by_session[e["source_file"]].append(e)

    repeated = []

    for session, items in by_session.items():

        items.sort(
            key=lambda x: x["start"] if x["start"] else datetime.min
        )

        counts = Counter(
            e["process_code"]
            for e in items
        )

        for process_code, count in counts.items():

            if count > 1:

                repeated.append({
                    "session": session,
                    "process_code": process_code,
                    "count": count
                })

    return repeated


# ============================================================
# Detect A -> B -> A patterns
# ============================================================

def detect_aba_patterns(executions):

    by_session = defaultdict(list)

    for e in executions:
        by_session[e["source_file"]].append(e)

    aba_patterns = []

    for session, items in by_session.items():

        items.sort(
            key=lambda x: x["start"] if x["start"] else datetime.min
        )

        for i in range(1, len(items) - 1):

            a = items[i - 1]
            b = items[i]
            c = items[i + 1]

            if (
                a["process_code"]
                and b["process_code"]
                and c["process_code"]
                and a["process_code"] == c["process_code"]
                and a["process_code"] != b["process_code"]
            ):

                aba_patterns.append({
                    "session": session,

                    "process_a": a["process_code"],
                    "process_b": b["process_code"],

                    "a1_case_id": a["case_id"],
                    "b_case_id": b["case_id"],
                    "a2_case_id": c["case_id"],

                    "a1_start": a["start_ts"],
                    "b_start": b["start_ts"],
                    "a2_start": c["start_ts"]
                })

    return aba_patterns


# ============================================================
# Detect process transitions
# ============================================================

def detect_transitions(executions):

    by_session = defaultdict(list)

    for e in executions:
        by_session[e["source_file"]].append(e)

    transitions = Counter()

    for session, items in by_session.items():

        items.sort(
            key=lambda x: x["start"] if x["start"] else datetime.min
        )

        for i in range(len(items) - 1):

            current = items[i]["process_code"]
            nxt = items[i + 1]["process_code"]

            if current and nxt and current != nxt:
                transitions[(current, nxt)] += 1

    return transitions


# ============================================================
# Variant analysis
# ============================================================

def variant_analysis(executions):

    result = defaultdict(Counter)

    for e in executions:

        code = e["process_code"]
        variant = e["variant"] or "UNKNOWN"

        result[code][variant] += 1

    return result


# ============================================================
# Continuation analysis
# ============================================================

def continuation_analysis(executions):

    return {
        "continues_from_previous":
            sum(
                1 for e in executions
                if e["continues_from_prev"]
            ),

        "continues_to_next":
            sum(
                1 for e in executions
                if e["continues_to_next"]
            )
    }


# ============================================================
# Write JSON
# ============================================================

def save_json(path, data):

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            default=str
        )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Day 3 Phase 6.2 - Process identity and interleaving analysis"
    )

    parser.add_argument(
        "--dataset-a",
        required=True,
        help="Path to Dataset A"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output directory"
    )

    args = parser.parse_args()

    dataset_a = Path(args.dataset_a)
    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not dataset_a.exists():

        raise FileNotFoundError(
            f"Dataset A not found: {dataset_a}"
        )

    # --------------------------------------------------------
    # Find manifests
    # --------------------------------------------------------

    manifest_files = find_manifest_files(dataset_a)

    if not manifest_files:

        raise FileNotFoundError(
            f"No gt_manifest.json files found under: {dataset_a}"
        )

    print("=" * 60)
    print("PHASE 6.2 - PROCESS IDENTITY & INTERLEAVING")
    print("=" * 60)

    print(f"Manifest files found : {len(manifest_files)}")

    # --------------------------------------------------------
    # Load executions
    # --------------------------------------------------------

    all_executions = []

    for manifest_path in manifest_files:

        try:
            manifest = load_json(manifest_path)

            executions = extract_executions(
                manifest,
                manifest_path
            )

            all_executions.extend(executions)

        except Exception as e:

            print(
                f"WARNING: failed to read {manifest_path}: {e}"
            )

    all_executions = sort_executions(all_executions)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    process_stats = process_identity_analysis(
        all_executions
    )

    repeated = repeated_process_analysis(
        all_executions
    )

    aba_patterns = detect_aba_patterns(
        all_executions
    )

    transitions = detect_transitions(
        all_executions
    )

    variants = variant_analysis(
        all_executions
    )

    continuations = continuation_analysis(
        all_executions
    )

    # --------------------------------------------------------
    # Save executions
    # --------------------------------------------------------

    execution_output = []

    for e in all_executions:

        execution_output.append({
            k: v
            for k, v in e.items()
            if k not in ("start", "end")
        })

    save_json(
        output_dir / "phase6_2_gt_executions.json",
        execution_output
    )

    # --------------------------------------------------------
    # Save process statistics
    # --------------------------------------------------------

    process_output = {}

    for code, data in process_stats.items():

        process_output[code] = {
            "executions": data["executions"],
            "variants": dict(data["variants"]),
            "domains": dict(data["domains"]),
            "case_ids": data["case_ids"],
            "continues_from_previous":
                data["continuations_from_previous"],
            "continues_to_next":
                data["continuations_to_next"]
        }

    save_json(
        output_dir / "phase6_2_process_identity.json",
        process_output
    )

    # --------------------------------------------------------
    # Save repeated executions
    # --------------------------------------------------------

    save_json(
        output_dir / "phase6_2_repeated_processes.json",
        repeated
    )

    # --------------------------------------------------------
    # Save A-B-A patterns
    # --------------------------------------------------------

    save_json(
        output_dir / "phase6_2_aba_patterns.json",
        aba_patterns
    )

    # --------------------------------------------------------
    # Save transitions
    # --------------------------------------------------------

    transition_output = [
        {
            "from": a,
            "to": b,
            "count": count
        }
        for (a, b), count in
        sorted(
            transitions.items(),
            key=lambda x: -x[1]
        )
    ]

    save_json(
        output_dir / "phase6_2_transitions.json",
        transition_output
    )

    # --------------------------------------------------------
    # Save variants
    # --------------------------------------------------------

    variant_output = {
        code: dict(counter)
        for code, counter in variants.items()
    }

    save_json(
        output_dir / "phase6_2_variants.json",
        variant_output
    )

    # --------------------------------------------------------
    # Save continuation statistics
    # --------------------------------------------------------

    save_json(
        output_dir / "phase6_2_continuations.json",
        continuations
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report_path = (
        output_dir /
        "day3_phase6_2_report.md"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("# Day 3 — Phase 6.2\n\n")
        f.write("## Process Identity and Interleaving Analysis\n\n")

        f.write("## Summary\n\n")

        f.write(
            f"- Manifest files analyzed: "
            f"{len(manifest_files)}\n"
        )

        f.write(
            f"- Ground-truth executions analyzed: "
            f"{len(all_executions)}\n"
        )

        f.write(
            f"- Process families: "
            f"{len(process_stats)}\n"
        )

        f.write(
            f"- Unique process transitions: "
            f"{len(transitions)}\n"
        )

        f.write(
            f"- Repeated-process sessions: "
            f"{len(repeated)}\n"
        )

        f.write(
            f"- A-B-A patterns: "
            f"{len(aba_patterns)}\n"
        )

        f.write(
            f"- Continues from previous chunk: "
            f"{continuations['continues_from_previous']}\n"
        )

        f.write(
            f"- Continues to next chunk: "
            f"{continuations['continues_to_next']}\n"
        )

        f.write("\n## Process Families\n\n")

        f.write(
            "| Process | Executions | Variants |\n"
        )

        f.write(
            "|---|---:|---|\n"
        )

        for code in sorted(process_stats):

            count = process_stats[code]["executions"]

            variant_text = ", ".join(
                f"{v}: {n}"
                for v, n in
                process_stats[code]["variants"].items()
            )

            f.write(
                f"| {code} | {count} | {variant_text} |\n"
            )

        f.write("\n## A-B-A Patterns\n\n")

        if aba_patterns:

            for pattern in aba_patterns[:100]:

                f.write(
                    f"- {pattern['process_a']} → "
                    f"{pattern['process_b']} → "
                    f"{pattern['process_a']} "
                    f"(cases: "
                    f"{pattern['a1_case_id']}, "
                    f"{pattern['b_case_id']}, "
                    f"{pattern['a2_case_id']})\n"
                )

        else:

            f.write(
                "No A-B-A patterns were detected in the "
                "ground-truth execution ordering.\n"
            )

        f.write("\n## Interpretation\n\n")

        f.write(
            "This analysis uses gt_manifest.json execution records "
            "rather than reconstructing executions from event-level "
            "start/switch signals. This provides a direct ground-truth "
            "reference for process identity, repeated executions, "
            "variants, continuation across chunks, and interleaving.\n"
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PHASE 6.2 COMPLETE")
    print("=" * 60)

    print(
        f"GT executions analyzed : {len(all_executions)}"
    )

    print(
        f"Process families        : {len(process_stats)}"
    )

    print(
        f"Process transitions      : {len(transitions)}"
    )

    print(
        f"Repeated-process cases   : {len(repeated)}"
    )

    print(
        f"A-B-A patterns           : {len(aba_patterns)}"
    )

    print(
        f"Cross-chunk continuations: "
        f"{continuations['continues_from_previous']}"
    )

    print()
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()