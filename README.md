# ProcessTimeSeries

`ProcessTimeSeries` is a reproducible research simulator for **synthetic discrete manufacturing time series**. It models a manufacturing line as a **directed acyclic graph (DAG)** of production nodes, where each node combines a small Petri-net-style material-flow module with an FSM describing machine states such as `idle`, `producing`, `blocked`, and `failed`.

## What this repo reproduces

This repository reproduces the paper-facing **5-node manufacturing demo** and exposes the reusable simulator backbone for defining your own manufacturing graphs. The main supported outputs are buffer-level trajectories, machine-state trajectories, summary metadata, and a scriptable paper-style figure.

## Quickstart install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

Then verify the install with:

```bash
python3 -m pytest -q
```

## One-command paper reproduction

Run the paper’s extracted 5-node demo with:

```bash
python3 scripts/reproduce_wsc2024.py
```

This uses `configs/paper/wsc2024_demo_5_node.yaml` and writes outputs under `experiments/wsc2024/outputs/`.

## Expected outputs

After the reproduction command finishes, the output directory contains:

- `buffer_levels.csv`
- `machine_states.csv`
- `summary.json`
- `figure5.png`

## Run your own manufacturing system

1. Copy `configs/examples/custom_template.yaml`
2. Edit the node count, edges, capacities, and timing distributions
3. Run:

```bash
python3 scripts/run_simulation.py --config configs/examples/custom_template.yaml
```

You can validate a config before running it:

```bash
python3 scripts/validate_config.py --config configs/examples/custom_template.yaml
```

## Repository structure

```text
ProcessTimeSeries/
├── src/process_timeseries/          # clean public API for simulator / export / plotting
├── discrete_manufacturing_sim/      # preserved compatibility backbone
├── configs/                         # paper and example YAML configs
├── scripts/                         # supported CLI and reproduction entry points
├── experiments/wsc2024/             # paper reproduction outputs
├── notebooks/                       # thin tutorial / convenience notebooks
├── docs/                            # paper map, config docs, output schema
└── tests/                           # automated behavioral checks
```

> The core simulator lives under `src/process_timeseries/simulator/`, with the original implementation preserved in `discrete_manufacturing_sim/` for compatibility and traceability.

## Paper-to-code map

- DAG construction and validation → `src/process_timeseries/simulator/dag.py`
- Petri-net-style material flow → `src/process_timeseries/simulator/petri.py`
- FSM machine logic → `src/process_timeseries/simulator/fsm.py`
- Event timing and distributions → `src/process_timeseries/simulator/events.py` and `distributions.py`
- Scheduling and top-level engine → `src/process_timeseries/simulator/scheduler.py` and `engine.py`
- Export and plotting → `src/process_timeseries/io/export.py` and `src/process_timeseries/plotting/figures.py`

For a fuller traceability table, see `docs/paper_map.md`.

## Current limitations

- **DAG-only** manufacturing graphs are supported; cycles and rework loops are intentionally rejected.
- The maintained paper/demo path focuses on the current 2-place Petri-net-style node model rather than a large general Petri-net framework.
- PyTorch is still part of the runtime because the original simulator backbone already uses it.

## Citation

If you use this repository in research, cite the accompanying paper and the metadata in `CITATION.cff`.

## License

This repository is distributed under the `MIT` license. See `LICENSE`.

