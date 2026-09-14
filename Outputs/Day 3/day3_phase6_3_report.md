# Day 3 — Phase 6.3

## Process Identity Feature Analysis

- GT executions: 1752
- Raw events: 162768
- Process families: 15

## Most Discriminative Features

| Rank | Feature | Between-process score |
|---:|---|---:|
| 1 | duration_seconds | 0.3026 |
| 2 | app_entropy | 0.2755 |
| 3 | unique_apps | 0.2744 |
| 4 | keystrokes_per_second | 0.2695 |
| 5 | window_count | 0.2502 |
| 6 | clipboard_count | 0.2416 |
| 7 | screenshot_count | 0.2251 |
| 8 | event_count | 0.2140 |
| 9 | clipboard_per_second | 0.2080 |
| 10 | clicks_per_second | 0.2068 |
| 11 | unique_event_types | 0.2040 |
| 12 | events_per_second | 0.2015 |
| 13 | mouse_click_count | 0.2002 |
| 14 | keystroke_count | 0.1731 |
| 15 | browser_count | 0.1705 |
| 16 | unique_layers | 0.1304 |
| 17 | layer_entropy | 0.1192 |
| 18 | event_type_entropy | 0.1188 |

## Mutual Information

| Rank | Feature | MI score |
|---:|---|---:|
| 1 | duration_seconds | 0.5348 |
| 2 | keystroke_count | 0.3195 |
| 3 | keystrokes_per_second | 0.1785 |
| 4 | clicks_per_second | 0.1261 |
| 5 | window_count | 0.1185 |
| 6 | clipboard_count | 0.1080 |
| 7 | clipboard_per_second | 0.0981 |
| 8 | events_per_second | 0.0945 |
| 9 | event_count | 0.0743 |
| 10 | unique_apps | 0.0704 |
| 11 | screenshot_count | 0.0576 |
| 12 | app_entropy | 0.0564 |
| 13 | mouse_click_count | 0.0557 |
| 14 | event_type_entropy | 0.0411 |
| 15 | browser_count | 0.0366 |
| 16 | unique_event_types | 0.0329 |
| 17 | layer_entropy | 0.0151 |
| 18 | unique_layers | 0.0000 |

## Interpretation

The analysis measures whether behavioral features extracted from raw operation logs differ systematically between logical process families. Features with high between-process variance or mutual information are candidates for process identity modeling. These scores are descriptive and are not themselves a final classification accuracy measure.
