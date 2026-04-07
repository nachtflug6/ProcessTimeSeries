# Config Reference

Simulation runs are configured with YAML files under `configs/`.

## Top-level fields

| Field | Required | Description |
|---|---|---|
| `name` | no | Human-readable run name |
| `description` | no | Short description of the config |
| `seed` | no | RNG seed for Python, NumPy, and Torch |
| `output_dir` | yes | Directory for exported artifacts |
| `graph` | yes | Manufacturing DAG definition |
| `petri` | yes | Per-node Petri-net capacities, markings, and weights |
| `events` | yes | Event timing distributions |
| `stop_conditions` | yes | At least one of `max_time`, `max_events`, or `max_output` |

## `graph`

```yaml
graph:
  num_nodes: 5
  edges:
    - {source: 0, target: 1, weight: 1}
    - {source: 1, target: 2, weight: 1}
```

- `num_nodes` must be positive.
- The graph must be a **DAG**.
- Edge `weight` is used as the transport / batch weight.

## `petri`

```yaml
petri:
  length_pns: 2
  initial_markings:
    - [0, 0]
  capacities:
    - [1, 5]
  weights:
    - [1, 1]
```

- `length_pns` is currently expected to be `2` for the maintained paper/demo path.
- `initial_markings` must be non-negative.
- `capacities` and `weights` must be strictly positive.

## `events`

```yaml
events:
  defaults:
    cycle_time:
      type: normal
      mean: 180.0
      std_dev: 0.4
    failure_time:
      type: normal
      mean: 18000.0
      std_dev: 0.4
    repair_time:
      type: normal
      mean: 3600.0
      std_dev: 0.4
```

Supported types:
- `normal`
- `uniform`
- `exponential`

Per-node overrides can be added with `events.per_node` using node IDs as keys.

## `stop_conditions`

```yaml
stop_conditions:
  max_time: 28800
```

At least one stop condition is required:
- `max_time`
- `max_events`
- `max_output`

## Examples

- `configs/paper/wsc2024_demo_5_node.yaml`
- `configs/examples/simple_line_3_node.yaml`
- `configs/examples/branching_5_node.yaml`
- `configs/examples/custom_template.yaml`
