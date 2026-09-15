"""Topic 1: reconstruct an evidence-supported recurring workflow from Dataset B."""
from __future__ import annotations
import csv, json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATASET = ROOT / "Dataset B" / "dataset_b"
OUT = ROOT / "Outputs" / "Day 4" / "Topic 1"

def files():
    found = sorted(p for p in DATASET.rglob("events.jsonl") if any(x.name.startswith("ses_") for x in p.parents))
    if not found: raise FileNotFoundError(f"Canonical Dataset B events.jsonl files not found below {DATASET}")
    return found
def load():
    rows=[]
    for p in files():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                e=json.loads(line); e["_file"]=p; rows.append(e)
    return sorted(rows,key=lambda e:e.get("timestamp_ms",0))
def marker(e):
    p=e.get("payload") or {}; x=p.get("element") or p.get("target_element") or {}; a=x.get("attributes") or {}
    return a.get("id") or x.get("automation_id") or ""
def iso(e): return e.get("timestamp_iso", "")
def write(name, rows, fields):
    with (OUT/name).open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True); events=load(); apps=Counter(); stages=Counter(); evidence=[]; sessions=defaultdict(int)
    for e in events:
        c=e.get("context") or {}; app=(c.get("active_app") or {}).get("app_name") or "Unknown"; apps[app]+=1
        text=json.dumps({"context":c,"payload":e.get("payload") or {}},ensure_ascii=False).casefold(); m=marker(e)
        stage=""
        if m=="pi-note": stage="comment_input"
        elif m=="btn-pi-ok": stage="confirmation_registration"
        elif e.get("event_type")=="browser_form_input": stage="record_review_or_input"
        elif e.get("event_type")=="clipboard_change": stage="reference_or_comment_transfer"
        elif app in {"Microsoft Word","Notepad","Microsoft Excel"}: stage="supporting_document_lookup"
        elif "財務会計" in text or "請求書" in text: stage="finance_record_context"
        if stage:
            stages[stage]+=1; sessions[e.get("session_id","")]+=1
            evidence.append({"session_id":e.get("session_id",""),"timestamp":iso(e),"stage":stage,"event_type":e.get("event_type",""),"application":app,"ui_id":m,"source_file":str(e["_file"].relative_to(ROOT))})
    write("workflow_stage_evidence.csv",evidence,["session_id","timestamp","stage","event_type","application","ui_id","source_file"])
    write("application_summary.csv",[{"application":k,"events":v} for k,v in apps.most_common()],["application","events"])
    workflow={"evidence_basis":{"events":len(events),"sessions":len({e.get('session_id') for e in events}),"caveat":"Dataset B has no ground truth. BPROC_01 is a behavioral family, not a confirmed Finance process; UI evidence is not assigned to BPROC_01."},"candidate_workflow":[{"stage":"record selection/review","support":"browser form interactions and Finance-record context"},{"stage":"supporting-document/reference lookup","support":"Word, Notepad, Excel, clipboard, and cross-application activity; interpretation is provisional"},{"stage":"review/decision","support":"inferred from record context before documentation; not directly labelled in logs"},{"stage":"comment generation/input","support":"direct pi-note UI id"},{"stage":"confirmation/registration","support":"direct btn-pi-ok UI id"},{"stage":"next-record behavior","support":"repeated cycles in the same session"}],"stage_event_counts":dict(stages),"stable_core":["structured field comparison","comment drafting/input","human-confirmed registration"],"branches_and_exceptions":["missing or conflicting reference data","amount/supplier discrepancy","business-rule judgement","uncertain supporting-document interpretation"]}
    (OUT/"workflow_reconstruction.json").write_text(json.dumps(workflow,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"workflow_reconstruction.md").write_text("# Topic 1 — Workflow reconstruction\n\nThe strongest supported recurring pattern is **record review → supporting-information lookup → review/decision → comment input → confirmation → next record**. Direct UI evidence supports the final two stages; reference lookup and decision are cautious inferences from application/clipboard context. Dataset B does not identify business processes, and BPROC_01 must not be treated as a Finance label.\n",encoding="utf-8")
    print(f"Topic 1 complete: {len(events)} events; outputs: {OUT}")
if __name__=="__main__": main()
