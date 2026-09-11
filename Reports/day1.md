# Day 1 Report — Dataset Understanding & Process Behavior Analysis

## 1. Objective

The objective of Day 1 was to understand the structure and semantics of the provided datasets before attempting process segmentation.

The main questions investigated were:

1. How are sessions, chunks, raw events, and ground-truth process events structured?
2. How many business processes occur in Dataset A?
3. How frequently do processes switch within a session?
4. Are sessions themselves equivalent to individual business-process executions?
5. Can raw desktop/browser events be aligned reliably with ground-truth process transitions?
6. Are there observable application, browser, UI, or event-level signals around process boundaries?
7. What assumptions should be avoided before developing the segmentation approach?

---

## 2. Dataset and Schema Understanding

Dataset A contains sessions with raw interaction logs and ground-truth process information.

A session represents one continuous recording. The `chunk_*` directories are fixed time buckets and do not represent business-process boundaries.

Raw interaction data is stored in `events.jsonl`. Raw events contain fields such as:

- `event_id`
- `session_id`
- `timestamp_ms`
- `timestamp_iso`
- `event_type`
- `layer`
- `source`
- `context`
- `payload`
- `metadata`

The ground truth is stored in `gt.jsonl`.

Ground-truth events contain information such as:

- process code
- process name
- case ID
- event type
- transition information
- timestamps
- current process
- process variant

The ground truth distinguishes the process family from individual executions/cases.

This distinction is important because the same business process can occur multiple times within a single session.

---

## 3. Initial Dataset Inventory

Dataset A contains:

- 63 sessions
- 15 business processes

The process codes are:

| Code | Process |
|---|---|
| A | 住民税通知確認 |
| B | 給与備考・控除整備 |
| C | 育児・産休申請確認 |
| D | 社保・年金補正対応 |
| E | 入社照合・手当確認 |
| F | 請求書承認 |
| G | 経費精算承認 |
| H | 銀行勘定照合 |
| I | 予算差異分析 |
| J | 支払処理 |
| K | 受注処理 |
| L | 在庫調整 |
| M | 仕入先連絡 |
| N | 出荷追跡 |
| O | 返品処理 |

---

## 4. Session-Level Analysis

Across Dataset A:

- Total GT events: 10,609
- Total process transitions: 1,590
- Distinct process cases: 2,009
- Average session duration: approximately 23.24 minutes
- Average processes present per session: approximately 9.14
- Average cases per session: approximately 31.89
- Average process transitions per session: approximately 25.24

### Important observation

A session is not equivalent to a business-process execution.

A typical session contains many different processes and many individual cases. Therefore, using the session boundary as the process boundary would be incorrect.

The session-level analysis also showed substantial process interleaving.

The session with the highest number of process switches had:

- 39 process switches
- 13 distinct processes
- 44 distinct cases

This demonstrates that multiple business activities can be interleaved within a single continuous recording.

---

## 5. Process-Level Behavior

Process-level analysis showed that all 15 processes participate in multiple sessions and interact with other processes.

Examples of process presence across the 63 sessions include:

- A: 40 sessions
- B: 35 sessions
- C: 43 sessions
- D: 38 sessions
- E: 37 sessions
- F: 41 sessions
- G: 39 sessions
- H: 39 sessions
- I: 39 sessions
- J: 34 sessions
- K: 32 sessions
- L: 37 sessions
- M: 42 sessions
- N: 43 sessions
- O: 37 sessions

No process behaves as a completely isolated workflow.

Several processes also contain suspension/resumption behavior, particularly:

- A
- B
- F
- K
- L

This means that a process can temporarily leave the active state and later resume.

Therefore, simple assumptions such as:

> process starts → continuous events → process ends

cannot be applied universally.

---

## 6. Process Transition Analysis

Ground-truth transition events were extracted from `process_switched_out` events using the `from`, `to`, and `ts_utc` fields.

Results:

- Process-switch events: 1,590
- Valid transitions: 1,590
- Missing `from`/`to`: 0
- Unique directed transitions: 210

The transition graph is therefore highly connected.

### Most frequent transitions

| Transition | Count |
|---|---:|
| M → C | 18 |
| G → J | 17 |
| G → H | 17 |
| F → A | 15 |
| A → I | 15 |
| M → A | 15 |
| C → M | 15 |
| B → A | 13 |
| K → F | 13 |
| A → F | 13 |
| N → M | 13 |
| H → K | 13 |
| M → N | 13 |
| F → C | 13 |
| C → A | 13 |
| A → D | 13 |
| C → B | 12 |
| A → C | 12 |
| A → E | 12 |
| G → L | 12 |

### Interpretation

The process graph is not a simple linear workflow.

There are many possible transitions between processes, and several processes have a large number of outgoing transitions.

For example, A and M each have 14 observed outgoing transitions.

This indicates that process segmentation will likely need to consider context rather than relying on a fixed global sequence.

---

## 7. Process Co-occurrence

Process co-occurrence was also examined to determine whether certain processes consistently appear together.

Some of the strongest observed combinations included:

- A + N
- C + M
- F + M
- F + N
- C + N
- H + I
- A + C
- A + M
- C + H
- G + H
- A + F
- I + L
- G + J
- H + J
- M + O

### Interpretation

Processes frequently occur in the same sessions, but co-occurrence alone is not sufficient to determine process boundaries.

A session can contain multiple processes that are interleaved rather than executed as one combined workflow.

Therefore:

> Process presence ≠ process identity.

---

## 8. Ground Truth → Raw Event Alignment

An initial attempt was made to construct complete execution windows from the ground-truth event stream and map raw events into those windows.

The initial mapper produced:

- 3,409 estimated GT execution windows
- 1,962 windows containing raw events
- 1,447 windows without raw events
- 57.55% raw-event coverage

However, this result was considered unreliable.

The reason was not poor raw-event quality. The execution-window construction was over-splitting because of complications in the GT event structure, including:

- repeated `process_started` events
- multiple cases of the same process
- process switches
- suspension/resumption behavior
- incomplete one-to-one correspondence between process starts and switch events

Therefore, the 57.55% figure was not treated as a true measure of raw-data quality.

The initial execution-window approach was explicitly rejected as a final segmentation ground truth.

---

## 9. Transition-Boundary Alignment

A more reliable alignment strategy was then used.

Instead of attempting to reconstruct complete process executions, each ground-truth `process_switched_out` event was treated as a known process-boundary timestamp.

For every transition, raw events were collected around the boundary using four temporal windows:

- -5s to -2s
- -2s to 0s
- 0s to +2s
- +2s to +5s

Results:

- GT transitions: 1,590
- Boundaries with raw events: 1,570
- Boundaries without raw events: 20
- Boundary raw-event coverage: 98.74%
- Raw events collected around boundaries: 46,506

### Key conclusion

The raw event stream aligns very well with known process-transition boundaries.

The 98.74% boundary coverage provides strong evidence that the raw desktop/browser logs contain sufficient temporal information for further process-segmentation analysis.

---

## 10. Transition-Specific Signals

Raw events around transition boundaries were inspected for application, browser, UI, and event-level signals.

Examples of signals observed around transitions include:

### A → E

Before the transition, frequent events included:

- screenshot events
- keystrokes
- mouse clicks
- browser clicks
- browser form input

The events were strongly associated with the HR system.

Specific UI elements such as:

- `btn-rt-query`
- `btn-rt-confirm`

were observed.

After the transition, browser navigation and screenshot activity appeared in the next process context.

---

### D → F

Before the transition:

- screenshots
- mouse clicks
- keystrokes
- browser clicks
- browser form input

were common.

The HR system was dominant.

The UI element:

- `btn-si-complete`

was observed.

Immediately after the transition, application-switch events became prominent, followed by activity associated with the Finance system.

---

### E → M

Before the transition:

- screenshots
- mouse clicks
- browser form input
- browser clicks
- keystrokes

were frequent.

The HR system was dominant.

UI elements included:

- `btn-ob-complete`
- `btn-ob-flag`

After the transition, application switches and browser navigation increased and the activity moved toward the system associated with process M.

---

### M → A

Before the transition, activity was strongly associated with the order/inventory system.

Frequent event types included:

- screenshots
- mouse clicks
- browser clicks
- browser form input
- keystrokes

The UI element:

- `btn-la-approve`

was observed.

After the transition, the raw context moved toward the HR system associated with process A.

---

### G → J

Before the transition, the Finance system dominated.

Frequent events included:

- screenshots
- keystrokes
- mouse clicks
- browser clicks
- browser form input

The UI element:

- `btn-pi-register`

was observed.

After the transition, browser navigation and application switching were observed.

---

## 11. Important Findings

### Finding 1 — Sessions are too coarse

A session contains multiple processes and multiple cases.

Therefore, session boundaries cannot be used as process-execution boundaries.

### Finding 2 — Processes are heavily interleaved

The transition graph contains 210 unique directed transitions across only 15 processes.

This indicates substantial contextual variation.

### Finding 3 — Cases matter

The same process can appear multiple times within one session.

Therefore, process family identification and individual execution segmentation are separate problems.

### Finding 4 — Suspension/resumption must be handled

Some processes can be suspended and resumed.

A segmentation system must therefore distinguish temporary process interruption from process completion.

### Finding 5 — Raw event alignment is strong

Using known GT transition timestamps gives 98.74% boundary coverage.

This suggests that raw events contain useful information around true process boundaries.

### Finding 6 — Boundary signals are heterogeneous

Process transitions can be associated with:

- application changes
- browser navigation
- window/context changes
- UI elements
- mouse activity
- keyboard activity
- screenshots
- browser form interactions

No single raw event type should therefore be assumed to define a process boundary.

---

## 12. Day 1 Conclusion

Day 1 established that the core challenge is not simply detecting activity.

The main challenge is:

> determining which continuous low-level desktop/browser events belong to the same business-process execution in a highly interleaved environment.

The analysis provides a strong starting point for this problem because:

1. Ground-truth process transitions can be recovered reliably.
2. Raw events align closely with transition timestamps.
3. Process-specific applications and UI signals are visible.
4. Sessions contain substantial process interleaving.
5. Process suspension/resumption introduces additional complexity.
6. A simple session-based or fixed-sequence segmentation strategy would be insufficient.

The next step should therefore focus on determining which raw signals are actually discriminative for identifying process boundaries and process identity.

This will be investigated during Day 2.

---