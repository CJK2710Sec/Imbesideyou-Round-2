# Day 3 — Phase 6.1 Process Identity Analysis

## 1. Dataset Overview

- Sessions analyzed: 63
- Process families discovered: 16
- Reconstructed executions: 1819

## 2. Process Families

| Process | GT events | Started | Suspended | Resumed | Switched out | Variants |
|---|---:|---:|---:|---:|---:|---:|
| `A` | 941 | 130 | 13 | 29 | 132 | 2 |
| `B` | 640 | 83 | 26 | 43 | 87 | 2 |
| `C` | 782 | 161 | 0 | 0 | 135 | 0 |
| `D` | 437 | 114 | 0 | 0 | 91 | 0 |
| `E` | 649 | 111 | 0 | 0 | 81 | 0 |
| `F` | 996 | 115 | 23 | 54 | 126 | 2 |
| `G` | 725 | 144 | 0 | 0 | 121 | 2 |
| `H` | 859 | 147 | 0 | 0 | 118 | 0 |
| `I` | 572 | 117 | 0 | 0 | 99 | 0 |
| `J` | 596 | 102 | 0 | 0 | 79 | 0 |
| `K` | 746 | 102 | 15 | 25 | 95 | 2 |
| `L` | 696 | 98 | 22 | 39 | 100 | 2 |
| `M` | 761 | 155 | 0 | 0 | 137 | 0 |
| `N` | 496 | 129 | 0 | 0 | 104 | 0 |
| `O` | 650 | 111 | 0 | 0 | 85 | 0 |
| `UNKNOWN_PROCESS` | 63 | 0 | 0 | 0 | 0 | 0 |

## 3. Execution Duration

| Process | Executions | Mean (s) | Median (s) | Min (s) | Max (s) |
|---|---:|---:|---:|---:|---:|

## 4. Process Transition / Interleaving

| From | To | Count |
|---|---|---:|
| `F` | `A` | 21 |
| `M` | `C` | 18 |
| `G` | `H` | 18 |
| `G` | `J` | 17 |
| `M` | `A` | 17 |
| `A` | `I` | 15 |
| `C` | `M` | 15 |
| `N` | `F` | 15 |
| `E` | `F` | 15 |
| `B` | `A` | 14 |
| `A` | `C` | 14 |
| `K` | `F` | 14 |
| `B` | `F` | 14 |
| `F` | `C` | 14 |
| `C` | `F` | 14 |
| `L` | `H` | 14 |
| `A` | `D` | 14 |
| `A` | `E` | 13 |
| `L` | `I` | 13 |
| `A` | `F` | 13 |
| `N` | `M` | 13 |
| `K` | `H` | 13 |
| `H` | `K` | 13 |
| `M` | `N` | 13 |
| `C` | `A` | 13 |
| `K` | `G` | 13 |
| `A` | `G` | 13 |
| `F` | `I` | 13 |
| `F` | `B` | 13 |
| `F` | `M` | 13 |
| `C` | `B` | 12 |
| `E` | `M` | 12 |
| `M` | `G` | 12 |
| `L` | `G` | 12 |
| `G` | `L` | 12 |
| `M` | `F` | 12 |
| `H` | `L` | 12 |
| `F` | `E` | 12 |
| `A` | `B` | 12 |
| `I` | `L` | 12 |
| `C` | `I` | 12 |
| `G` | `K` | 12 |
| `C` | `O` | 12 |
| `H` | `G` | 12 |
| `A` | `M` | 11 |
| `H` | `J` | 11 |
| `O` | `M` | 11 |
| `K` | `L` | 11 |
| `A` | `L` | 11 |
| `I` | `N` | 11 |
| `C` | `N` | 11 |
| `F` | `L` | 11 |
| `G` | `M` | 11 |
| `C` | `G` | 11 |
| `H` | `N` | 11 |
| `F` | `D` | 11 |
| `O` | `E` | 10 |
| `C` | `E` | 10 |
| `I` | `A` | 10 |
| `D` | `F` | 10 |
| `L` | `C` | 10 |
| `D` | `K` | 10 |
| `M` | `L` | 10 |
| `H` | `B` | 10 |
| `I` | `H` | 10 |
| `N` | `B` | 10 |
| `H` | `M` | 10 |
| `I` | `C` | 10 |
| `E` | `A` | 10 |
| `K` | `M` | 10 |
| `B` | `M` | 10 |
| `E` | `C` | 9 |
| `M` | `K` | 9 |
| `M` | `D` | 9 |
| `N` | `D` | 9 |
| `N` | `C` | 9 |
| `L` | `M` | 9 |
| `E` | `N` | 9 |
| `B` | `D` | 9 |
| `D` | `A` | 9 |
| `B` | `C` | 9 |
| `C` | `H` | 9 |
| `N` | `G` | 9 |
| `F` | `K` | 9 |
| `D` | `B` | 9 |
| `B` | `G` | 9 |
| `F` | `O` | 9 |
| `B` | `E` | 9 |
| `D` | `C` | 9 |
| `L` | `F` | 9 |
| `O` | `H` | 9 |
| `K` | `N` | 9 |
| `N` | `A` | 9 |
| `I` | `F` | 9 |
| `A` | `K` | 9 |
| `L` | `A` | 9 |
| `B` | `H` | 8 |
| `H` | `A` | 8 |
| `G` | `A` | 8 |
| `E` | `O` | 8 |
| `F` | `N` | 8 |
| `K` | `C` | 8 |
| `J` | `L` | 8 |
| `H` | `O` | 8 |
| `D` | `O` | 8 |
| `O` | `L` | 8 |
| `D` | `L` | 8 |
| `O` | `K` | 8 |
| `J` | `M` | 8 |
| `M` | `J` | 8 |
| `O` | `C` | 8 |
| `N` | `O` | 8 |
| `O` | `F` | 8 |
| `G` | `B` | 8 |
| `M` | `O` | 8 |
| `M` | `E` | 8 |
| `M` | `I` | 8 |
| `D` | `N` | 8 |
| `J` | `B` | 8 |
| `I` | `B` | 8 |
| `J` | `K` | 8 |
| `H` | `C` | 7 |
| `M` | `H` | 7 |
| `I` | `K` | 7 |
| `K` | `A` | 7 |
| `N` | `H` | 7 |
| `L` | `J` | 7 |
| `G` | `C` | 7 |
| `L` | `K` | 7 |
| `B` | `L` | 7 |
| `L` | `B` | 7 |
| `J` | `F` | 7 |
| `J` | `A` | 7 |
| `N` | `I` | 7 |
| `G` | `F` | 7 |
| `B` | `O` | 7 |
| `I` | `J` | 7 |
| `L` | `E` | 7 |
| `J` | `H` | 7 |
| `F` | `H` | 7 |
| `J` | `G` | 6 |
| `J` | `O` | 6 |
| `L` | `N` | 6 |
| `A` | `O` | 6 |
| `O` | `B` | 6 |
| `H` | `E` | 6 |
| `C` | `K` | 6 |
| `K` | `B` | 6 |
| `D` | `G` | 6 |
| `G` | `I` | 6 |
| `F` | `G` | 6 |
| `L` | `O` | 6 |
| `J` | `I` | 6 |
| `L` | `D` | 6 |
| `K` | `D` | 6 |
| `K` | `I` | 6 |
| `H` | `I` | 6 |
| `J` | `C` | 6 |
| `B` | `J` | 6 |
| `N` | `K` | 5 |
| `E` | `K` | 5 |
| `C` | `D` | 5 |
| `O` | `D` | 5 |
| `C` | `L` | 5 |
| `O` | `I` | 5 |
| `I` | `G` | 5 |
| `G` | `D` | 5 |
| `O` | `N` | 5 |
| `G` | `N` | 5 |
| `A` | `J` | 5 |
| `D` | `M` | 5 |
| `H` | `D` | 5 |
| `I` | `E` | 5 |
| `E` | `J` | 5 |
| `B` | `N` | 5 |
| `E` | `G` | 5 |
| `O` | `J` | 5 |
| `A` | `H` | 5 |
| `A` | `N` | 5 |
| `D` | `E` | 4 |
| `D` | `H` | 4 |
| `E` | `D` | 4 |
| `E` | `H` | 4 |
| `J` | `D` | 4 |
| `F` | `J` | 4 |
| `O` | `A` | 4 |
| `B` | `I` | 4 |
| `D` | `J` | 4 |
| `H` | `F` | 4 |
| `N` | `L` | 4 |
| `C` | `J` | 3 |
| `G` | `E` | 3 |
| `N` | `E` | 3 |
| `G` | `O` | 3 |
| `K` | `O` | 3 |
| `O` | `G` | 3 |
| `E` | `B` | 3 |
| `K` | `J` | 3 |
| `E` | `L` | 3 |
| `I` | `D` | 3 |
| `J` | `N` | 3 |
| `I` | `O` | 3 |
| `I` | `M` | 3 |
| `B` | `K` | 3 |
| `J` | `E` | 2 |
| `K` | `E` | 2 |
| `M` | `B` | 2 |
| `E` | `I` | 2 |
| `D` | `I` | 1 |
| `N` | `J` | 1 |

## 5. Return / Interleaving Patterns

Patterns of the form `A → B → A` are listed below. These are evidence for possible process interruption or interleaving and should not automatically be treated as separate A executions.

| Pattern | Count |
|---|---:|

## 6. Phase 6.1 Conclusions

The analysis establishes the Dataset A process identity landscape before process clustering or final segment construction.

The next analysis should determine which raw-event features are stable within a process family and which features distinguish different process families.
