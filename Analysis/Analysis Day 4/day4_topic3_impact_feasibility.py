"""Topic 3: standalone, conservative impact and feasibility analysis."""
from __future__ import annotations
import csv,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; DATASET=ROOT/"Dataset B"/"dataset_b"; OUT=ROOT/"Outputs"/"Day 4"/"Topic 3"
def load():
    fs=sorted(p for p in DATASET.rglob("events.jsonl") if any(x.name.startswith("ses_") for x in p.parents))
    if not fs: raise FileNotFoundError("Canonical Dataset B event files are missing")
    out=[]
    for p in fs:
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip(): out.append(json.loads(l))
    return out,fs
def marker(e):
    p=e.get("payload") or {}; x=p.get("element") or p.get("target_element") or {}; a=x.get("attributes") or {}; return a.get("id") or x.get("automation_id") or ""
def main():
    OUT.mkdir(parents=True,exist_ok=True); events,fs=load(); sessions={e.get("session_id") for e in events}; bysession={}
    for e in events:
        bysession.setdefault(e.get("session_id"),[]).append(e)
    duration=sum((max(x.get("timestamp_ms",0) for x in xs)-min(x.get("timestamp_ms",0) for x in xs))/1000 for xs in bysession.values() if xs)
    finance=sum("財務会計" in json.dumps({"context":e.get("context"),"payload":e.get("payload")},ensure_ascii=False) or "請求書" in json.dumps({"context":e.get("context"),"payload":e.get("payload")},ensure_ascii=False) for e in events)
    controls=Counter(marker(e) for e in events); interaction=Counter(e.get("event_type") for e in events)
    summary={"observed_workload":{"events":len(events),"sessions":len(sessions),"event_files":len(fs),"session_elapsed_seconds":round(duration,3),"finance_or_invoice_context_events":finance,"pi_note_direct_events":controls["pi-note"],"btn_pi_ok_direct_events":controls["btn-pi-ok"],"high_volume_interactions":{k:interaction[k] for k in ("browser_click","browser_form_input","clipboard_change","keystroke","shortcut")}},"automation_scope":{"automatable_assistance":["supplier/reference lookup after validated integration","supplier and amount comparison","difference flagging","standardized comment draft"],"human_work_remaining":["validate source/reference data","resolve exceptions and ambiguous cases","edit draft where necessary","final approval and registration"],"impact_interpretation":"No time saving is estimated: Dataset B has no ground truth and test-environment waiting time is not production timing."},"feasibility":{"prototype":"high: deterministic field comparison uses explicit inputs","production_integration":"conditional: needs approved source-system read access, field mapping, reference-data ownership, audit logs and human approval","constraints":["no business-process ground truth in Dataset B","BPROC_01 segments are fragmented/interleaved and are not business-execution counts","desktop logs do not establish permissions or business-rule semantics"]}}
    (OUT/"impact_feasibility.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    rows=[{"area":"Observed workload","finding":f"{len(events)} events across {len(sessions)} sessions; elapsed activity {duration:.1f} seconds","interpretation":"activity volume supports investigation, not a time-saving claim"},{"area":"Repetition","finding":f"{controls['pi-note']} direct pi-note and {controls['btn-pi-ok']} direct confirmation interactions","interpretation":"evidence of repeated documentation/confirmation UI activity"},{"area":"Automation opportunity","finding":"compare fields and draft standardized notes","interpretation":"assistance only; no autonomous registration"},{"area":"Fragmentation caveat","finding":"Day 3 found high same-family transitions/interleaving","interpretation":"do not equate BPROC_01 segment count with execution count"}]
    with (OUT/"impact_feasibility_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:w=csv.DictWriter(f,fieldnames=["area","finding","interpretation"]);w.writeheader();w.writerows(rows)
    (OUT/"impact_feasibility.md").write_text("# Topic 3 — Impact and feasibility\n\nThe evidence supports a repeated documentation/confirmation activity pattern and a feasible **assisted** comparison-and-drafting MVP. It does not support a production time-saving estimate or a count of completed invoices. Segment fragmentation, interleaving, missing Dataset B ground truth, and test-environment timing all prevent those claims.\n",encoding="utf-8")
    print(f"Topic 3 complete: {len(events)} events, {len(sessions)} sessions")
if __name__=="__main__":main()
