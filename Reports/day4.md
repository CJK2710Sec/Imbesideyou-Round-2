# Final Day 4 Report — Workflow Reconstruction, MVP, Impact and Risks

## 1. Executive Summary

Dataset B evidence supports a repeated record-processing pattern: record review, supporting-information lookup, documentation in `pi-note`, and confirmation through `btn-pi-ok`. The selected MVP is a deterministic Finance invoice-verification and comment-drafting assistant. It compares structured fields and creates a draft; a person retains responsibility for exceptions, approval, and registration.

## 2. Day 4 Objective

Translate the Day 3 priority behavioral family into an evidence-supported workflow hypothesis, a working prototype, a conservative feasibility assessment, and a rollout recommendation.

## 3. Dataset B and Evidence Basis

The Topic 1 scripts independently process the canonical Dataset B session/chunk files. They generated evidence from 20,477 events over 15 sessions. Direct UI evidence includes 540 `pi-note` and 247 `btn-pi-ok` interactions, plus 239 candidate documentation-to-confirmation sequences.

These are observations of UI activity—not invoice counts. Dataset B has no ground truth. Day 3's BPROC_01 is a behavioral family and must not be interpreted as a confirmed Finance process or a count of business executions; Day 3 also documented fragmentation and interleaving.

## 4. Topic 1 — Process/Workflow Reconstruction

The strongest supported candidate workflow is:

```text
Record selection/review → supporting-document or reference lookup
→ review/decision → comment input → confirmation/registration → next record
```

`pi-note` and `btn-pi-ok` directly support comment input and confirmation. Reference lookup and review/decision are constrained inferences from Finance context, Word/Notepad/Excel activity, clipboard activity, browser interactions, and temporal sequence. The stable candidate core is structured comparison, documentation drafting, and human-confirmed registration. Missing references, discrepancies, and business judgement remain branches or exceptions.

## 5. Topic 2 — Selected Automation and Working Prototype

The standalone MVP accepts invoice number, supplier, invoice amount, supplier reference, expected amount, and optional reference invoice number. It normalizes and compares fields, calculates the difference, generates a standardized Japanese note, and returns either `APPROVAL_CANDIDATE` or `HUMAN_REVIEW_HOLD`.

`APPROVAL_CANDIDATE` is not automatic approval: final human approval is always required and the prototype performs no registration.

## 6. Topic 3 — Impact and Feasibility

Observed interaction volume supports investigation of the stable verification/documentation core. The MVP can realistically assist with reference lookup after integration, field comparison, discrepancy flagging, and first-draft comments. It cannot yet replace reference-data validation, exception resolution, business judgement, or final registration.

No production time saving is claimed. Dataset B has no ground truth, BPROC_01 segment counts are fragmented/interleaved, and test-environment waiting time is not production timing. A pilot should measure clean-case handling time and exception rate after the reference-data source is validated.

## 7. Topic 4 — Risks and Alternative Approaches

Deterministic Python was selected because structured comparison rules are auditable and testable. RPA/UI automation was deferred because selectors, authentication, permissions, and submission safety are not established. An AI agent was deferred from the decision path because it adds non-determinism where structured rules suffice; it may later assist with unstructured extraction. A workflow platform is a future option after API access, data ownership, and orchestration requirements are known.

The risk register covers inferred workflow meaning, stale reference data, UI fragility, false matches, and overstated value. The mitigations are user validation, owned reference data, fail-closed discrepancies, no automatic registration, audit records, and a measured pilot.

## 8. Remaining Manual Work

Employees validate the source/reference data, resolve ambiguity or discrepancies, adjust drafts when needed, approve, and register. This boundary keeps consequential decisions with the business owner.

## 9. Recommended Implementation and Rollout

1. Validate this workflow hypothesis with Finance users and sample records.
2. Establish an owned, versioned reference-data source and read-only integration.
3. Pilot deterministic comparison plus comment drafting with mandatory human approval.
4. Measure clean-case time, correction rate, exceptions, and user acceptance.
5. Only then evaluate API/workflow integration or limited RPA.

## 10. Limitations and Evidence Caveats

Observed UI activity is not a labelled business process. Finance context does not prove that every detected behavioral segment is Finance. The logs do not disclose all rules, permissions, source-of-truth data, or production latency.

## 11. Final Day 4 Conclusion

The evidence justifies a conservative, human-in-the-loop invoice-verification MVP—not autonomous process automation. The prototype is working, independently testable, and positioned for a controlled validation pilot.
