# Output Format

The standard reproduction and config-driven scripts write outputs under the configured `output_dir`.

## Files

| File | Format | Description |
|---|---|---|
| `buffer_levels.csv` | CSV | Buffer/output-place trajectory over time for each node |
| `machine_states.csv` | CSV | FSM state trajectory over time for each node |
| `summary.json` | JSON | Run metadata, seed, runtime, total events, and final counts |
| `figure5.png` / `figure.png` | PNG | Scriptable paper-style visualization |

## `buffer_levels.csv`

| Column | Type | Meaning |
|---|---|---|
| `time` | float | Simulation clock in seconds |
| `node_id` | int | Node index in the manufacturing DAG |
| `buffer_level` | float | Output-buffer/place level for the node at that time |

## `machine_states.csv`

| Column | Type | Meaning |
|---|---|---|
| `time` | float | Simulation clock in seconds |
| `node_id` | int | Node index in the manufacturing DAG |
| `state` | string | Human-readable state name (`idle`, `producing`, `blocked`, `failed`) |
| `state_id` | int | Numeric FSM state identifier |

## `summary.json`

Typical keys:

- `seed`
- `config_path`
- `simulation_end_time`
- `total_events`
- `final_production_count_by_node`
- `failure_counts_by_node`
- `output_dir`

## Notes

- The exported trajectories are deterministic under a fixed seed.
- The current simulator assumes a **DAG-only** manufacturing graph; cyclic/rework loops are not supported.
