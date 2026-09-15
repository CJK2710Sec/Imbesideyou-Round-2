"""Topic 2: standalone deterministic Finance invoice-verification demonstration."""
from __future__ import annotations
import json,re
from dataclasses import asdict,dataclass
from decimal import Decimal,InvalidOperation
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; OUT=ROOT/"Outputs"/"Day 4"/"Topic 2"
@dataclass(frozen=True)
class Invoice: invoice_number:str; supplier:str; amount:object
@dataclass(frozen=True)
class Reference: supplier:str; expected_amount:object; invoice_number:str|None=None
def norm(v): return re.sub(r"\s+","",str(v or "")).casefold()
def money(v):
    try:
        clean=re.sub(r"[^0-9.\-]","",str(v or ""))
        if not clean: raise InvalidOperation
        return Decimal(clean)
    except InvalidOperation as exc: raise ValueError(f"invalid amount: {v!r}") from exc
def verify(invoice:Invoice,reference:Reference):
    actual,expected=money(invoice.amount),money(reference.expected_amount); supplier_match=norm(invoice.supplier)==norm(reference.supplier); number_match=None if not reference.invoice_number else norm(invoice.invoice_number)==norm(reference.invoice_number); difference=actual-expected
    status="APPROVAL_CANDIDATE" if supplier_match and number_match is not False and difference==0 else "HUMAN_REVIEW_HOLD"
    reasons=[]
    if not supplier_match: reasons.append("取引先")
    if difference!=0: reasons.append("金額")
    if number_match is False: reasons.append("請求書番号")
    result={"status":status,"supplier_matches":supplier_match,"amount_matches":difference==0,"invoice_number_matches":number_match,"difference":str(difference),"requires_human_final_approval":True,"comment":"\n".join(["請求書照合（自動下書き）",f"請求書番号：{invoice.invoice_number}",f"取引先：{invoice.supplier}",f"請求額：{actual:,.0f}円",f"参照額：{expected:,.0f}円",f"差額：{difference:,.0f}円",("照合結果：一致。承認候補（最終確認が必要）。" if status=="APPROVAL_CANDIDATE" else f"照合結果：差異あり（{'・'.join(reasons)}）。人による確認・保留。"),"※自動登録は行いません。"])}
    return result
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=[("matching",Invoice("INV-DEMO-001","Demo Supplier","120,000"),Reference("Demo Supplier",120000,"INV-DEMO-001")),("amount_discrepancy",Invoice("INV-DEMO-002","Demo Supplier",125000),Reference("Demo Supplier",120000,"INV-DEMO-002")),("supplier_discrepancy",Invoice("INV-DEMO-003","Other Supplier",120000),Reference("Demo Supplier",120000,"INV-DEMO-003"))]
    results=[{"case":n,"invoice":asdict(i),"reference":asdict(r),"result":verify(i,r)} for n,i,r in cases]
    (OUT/"mvp_demo_results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"mvp_demo_comments.txt").write_text("\n\n".join(f"[{x['case']}]\n{x['result']['comment']}" for x in results),encoding="utf-8")
    print(f"Topic 2 MVP complete: {len(results)} demonstration cases; outputs: {OUT}")
if __name__=="__main__":main()
