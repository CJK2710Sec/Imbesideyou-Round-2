"""Topic 1 companion: independently inventory UI, browser, and application evidence."""
from __future__ import annotations
import csv,json
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; DATASET=ROOT/"Dataset B"/"dataset_b"; OUT=ROOT/"Outputs"/"Day 4"/"Topic 1"
def load():
    fs=sorted(p for p in DATASET.rglob("events.jsonl") if any(x.name.startswith("ses_") for x in p.parents))
    if not fs: raise FileNotFoundError("Canonical Dataset B event files are missing")
    out=[]
    for p in fs:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip(): e=json.loads(line); e["_file"]=p; out.append(e)
    return sorted(out,key=lambda e:e.get("timestamp_ms",0))
def marker(e):
    p=e.get("payload") or {}; x=p.get("element") or p.get("target_element") or {}; a=x.get("attributes") or {}; return a.get("id") or x.get("automation_id") or ""
def csvout(name,rows,fields):
    with (OUT/name).open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True); events=load(); apps=Counter(); titles=Counter(); urls=Counter(); signals=Counter(); ui=[]; cycles=[]; seq=defaultdict(list)
    for e in events:
        c=e.get("context") or {}; a=c.get("active_app") or {}; b=c.get("active_browser_tab") or {}; m=marker(e); sid=e.get("session_id","")
        apps[a.get("app_name") or "Unknown"]+=1; titles[a.get("window_title") or ""]+=1; urls[b.get("url") or ""]+=1; seq[sid].append(e)
        if e.get("event_type") in {"browser_click","browser_form_input","clipboard_change","keystroke","shortcut"}: signals[e.get("event_type")]+=1
        if m in {"pi-note","btn-pi-ok"}: ui.append({"session_id":sid,"timestamp":e.get("timestamp_iso",""),"event_type":e.get("event_type",""),"ui_id":m,"application":a.get("app_name", ""),"url":b.get("url", "")})
    for sid,rows in seq.items():
        for i,e in enumerate(rows):
            if e.get("event_type")!="browser_click" or marker(e)!="pi-note": continue
            for later in rows[i+1:]:
                if later.get("timestamp_ms",0)-e.get("timestamp_ms",0)>300000: break
                if later.get("event_type")=="browser_click" and marker(later)=="btn-pi-ok":
                    cycles.append({"session_id":sid,"note_at":e.get("timestamp_iso",""),"confirm_at":later.get("timestamp_iso",""),"elapsed_seconds":round((later["timestamp_ms"]-e["timestamp_ms"])/1000,3)});break
    csvout("ui_application_inventory.csv",[{"application":k,"events":v} for k,v in apps.most_common()],["application","events"])
    csvout("ui_window_title_inventory.csv",[{"window_title":k,"events":v} for k,v in titles.most_common() if k],["window_title","events"])
    csvout("ui_url_inventory.csv",[{"url":k,"events":v} for k,v in urls.most_common() if k],["url","events"])
    csvout("ui_signal_summary.csv",[{"signal":k,"events":v} for k,v in signals.most_common()],["signal","events"])
    csvout("ui_control_evidence.csv",ui,["session_id","timestamp","event_type","ui_id","application","url"])
    (OUT/"representative_ui_cycles.json").write_text(json.dumps(cycles,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Topic 1 UI evidence complete: {len(events)} events, {len(cycles)} candidate cycles")
if __name__=="__main__":main()
