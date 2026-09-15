"""Standalone tests for day4_topic2_invoice_verification_mvp.py."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from day4_topic2_invoice_verification_mvp import Invoice,Reference,verify
def check(condition,message):
    if not condition: raise AssertionError(message)
def test_matching_invoice():
    r=verify(Invoice("INV-1","Supplier A","1,200"),Reference("Supplier A",1200,"INV-1"));check(r["status"]=="APPROVAL_CANDIDATE","matching record should be an approval candidate");check(r["requires_human_final_approval"],"final human approval must remain required")
def test_amount_discrepancy():
    r=verify(Invoice("INV-1","Supplier A",1300),Reference("Supplier A",1200,"INV-1"));check(r["status"]=="HUMAN_REVIEW_HOLD","amount discrepancy must hold")
def test_supplier_discrepancy():
    r=verify(Invoice("INV-1","Supplier B",1200),Reference("Supplier A",1200,"INV-1"));check(not r["supplier_matches"],"supplier difference must be detected")
def test_invalid_amount():
    try: verify(Invoice("INV-1","Supplier A","not money"),Reference("Supplier A",1200))
    except ValueError: return
    raise AssertionError("malformed amount must fail closed")
def main():
    for t in (test_matching_invoice,test_amount_discrepancy,test_supplier_discrepancy,test_invalid_amount): t(); print(f"PASS {t.__name__}")
    print("ALL DAY 4 TOPIC 2 MVP TESTS PASSED")
if __name__=="__main__":main()
