# Day 1 Work Log

## Objective

Understand Dataset A, recover the structure of the ground truth, analyze process behavior, and determine whether raw desktop/browser events contain usable information around business-process boundaries.

---

## Step 1 — Dataset Understanding

### What was done

Inspected the dataset structure and the provided schema.

Established the distinction between:

- session
- chunk
- raw event
- process
- process execution/case
- ground-truth event

### Important learning

Chunks are fixed time buckets and should not be interpreted as business-process boundaries.

A session can contain multiple processes and multiple cases.

---

## Step 2 — Initial Dataset Inventory

Created an inventory of Dataset A.

Observed:

- 63 sessions
- 15 business processes

The process codes A–O were identified and mapped to their process names.

---

## Step 3 — Ground Truth Inspection

Inspected the GT event structure.

Initially, the parser assumed fields such as:

- `event_type`
- `process`
- `timestamp`

were directly available in the expected format.

This resulted in zero detected transitions.

### What went wrong

The actual GT schema uses:

- `event`
- `from`
- `to`
- `ts_utc`

for the relevant process-switch events.

### Fix

The parser was corrected to use the actual GT schema.

After correction:

- 1,590 `process_switched_out` events were detected.
- All 1,590 had valid `from` and `to` values.

---

## Step 4 — Transition Analysis

Analyzed process-to-process transitions.

Results:

- 1,590 valid transitions
- 210 unique directed transitions

The process graph showed substantial connectivity and interleaving.

This ruled out the assumption that the processes follow one simple global sequence.

---

## Step 5 — Session Behavior Analysis

Analyzed sessions using:

- duration
- process diversity
- case count
- transition count
- process presence

Results:

- average session duration: ~23.24 minutes
- average processes/session: ~9.14
- average cases/session: ~31.89
- average transitions/session: ~25.24

A high-switching session contained:

- 39 switches
- 13 processes
- 44 cases

### Conclusion

A session is much larger than an individual process execution.

---

## Step 6 — Process Behavior Analysis

Analyzed process starts, executions, incoming/outgoing transitions, suspension, and resumption.

Observed that some processes have suspension/resumption behavior.

### Conclusion

A process cannot always be modeled as a single uninterrupted interval.

Suspension and resumption need to be considered in later segmentation logic.

---

## Step 7 — Initial GT-to-Raw Mapping Attempt

Attempted to reconstruct complete process execution windows from GT events and map raw events into them.

The first implementation encountered:

`TypeError: unhashable type: 'dict'`

### Fix

Added robust handling for nested dictionary/list values.

The script then executed successfully.

### Result

The mapper produced:

- 3,409 estimated execution windows
- 1,962 windows with raw events
- 1,447 windows without raw events
- 57.55% raw-event coverage

### Problem discovered

The execution-window count and coverage did not appear trustworthy.

The GT structure contains complexities such as:

- repeated process-start events
- same-process multiple cases
- switches
- suspension/resumption
- imperfect start/switch pairing

Therefore, the execution-window reconstruction was over-splitting.

### Decision

Do not use this mapper as final ground truth.

Keep it as an exploratory analysis because it helped reveal the GT semantics and prevented an incorrect conclusion about raw-data quality.

---

## Step 8 — Transition Boundary Alignment

A more direct approach was tested.

Instead of reconstructing complete process executions, known `process_switched_out` timestamps were used as boundary anchors.

Raw events were collected around each transition.

Temporal windows:

- -5s to -2s
- -2s to 0s
- 0s to +2s
- +2s to +5s

### Result

- 1,590 GT transitions
- 1,570 boundaries with raw events
- 20 boundaries without raw events
- 98.74% boundary coverage
- 46,506 raw events around boundaries

### Conclusion

Raw event alignment is strong.

The earlier 57.55% result was an artifact of the execution-window reconstruction rather than evidence of poor raw-event coverage.

---

## Step 9 — Transition Signal Inspection

Inspected the raw context around selected transitions.

Signals observed included:

- application name
- process/window name
- browser domain
- browser path
- browser navigation
- UI element information
- mouse clicks
- keyboard activity
- screenshots
- form inputs
- application switches

Specific UI elements were observed around some transitions.

Examples:

- `btn-rt-query`
- `btn-rt-confirm`
- `btn-si-complete`
- `btn-ob-complete`
- `btn-ob-flag`
- `btn-la-approve`
- `btn-pi-register`

### Conclusion

There are promising transition-level signals.

However, they have not yet been proven discriminative.

---

## What Did Not Work

### 1. Incorrect GT field assumptions

The first transition parser assumed incorrect field names.

Result:

- zero transitions detected.

This was fixed after inspecting the actual schema.

### 2. Naive complete execution-window reconstruction

The first GT-to-raw mapping approach generated 3,409 estimated windows and only 57.55% raw coverage.

This was not accepted as a valid representation of execution boundaries because the GT event semantics make simple state-machine reconstruction unreliable.

### 3. Treating session as process execution

Session-level analysis showed that this assumption is incorrect.

A single session can contain many processes and cases.

### 4. Treating event types as process signatures

Generic events such as:

- screenshots
- keystrokes
- mouse clicks
- browser clicks
- application switches

occur across many contexts.

Therefore, event type alone is unlikely to be sufficient for process identification.

---

## Key Day 1 Learnings

1. Dataset A contains rich ground truth but the GT semantics must be handled carefully.
2. There are 15 processes and 210 observed directed transitions.
3. Sessions are highly interleaved.
4. Process executions/cases are distinct from process families.
5. Some processes can be suspended and resumed.
6. Raw events align strongly with GT transition boundaries.
7. Application, browser, and UI context provide promising signals.
8. The next problem is determining which signals are genuinely discriminative.

---

## Day 1 Status

**Completed.**

No final segmentation model or process signature was created during Day 1.

The purpose of Day 1 was to establish a reliable understanding of the data and formulate testable hypotheses for Day 2.