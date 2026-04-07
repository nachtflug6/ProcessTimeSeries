# Paper-to-Code Map

This repository is centered on a **DAG-based discrete manufacturing simulator** that generates synthetic multivariate time series from production nodes with Petri-net-style material flow and FSM-based machine behavior.

## Concept traceability

| Paper concept | Code location | Notes |
|---|---|---|
| Manufacturing line as a DAG | `src/process_timeseries/simulator/dag.py` | Builds adjacency matrices, validates DAG structure, rejects cycles explicitly |
| Weighted transport edges | `src/process_timeseries/simulator/dag.py` + `discrete_manufacturing_sim/dg/wdg.py` | Edge weights are preserved in the weighted adjacency matrix |
| Per-node Petri-net-style production module | `src/process_timeseries/simulator/petri.py` + `discrete_manufacturing_sim/multi_linear_petri_net.py` | Preserved implementation for markings, capacities, weights, and transition firing |
| Machine FSM (`idle`, `producing`, `blocked`, `failed`) | `src/process_timeseries/simulator/fsm.py` + `discrete_manufacturing_sim/fsm/production_asset_fsm.py` | Explicit 4-state / 5-event production asset FSM |
| Event types and timing semantics | `src/process_timeseries/simulator/events.py` | Replaces magic indices with named enums |
| Distribution-based cycle/failure/repair timing | `src/process_timeseries/simulator/distributions.py` | Config-driven normal/uniform/exponential timing metadata |
| Next-event scheduling | `src/process_timeseries/simulator/scheduler.py` + `discrete_manufacturing_sim/multi_temporal_event_handler.py` | Minimum-time event selection and runtime advancement |
| Top-level simulation loop | `src/process_timeseries/simulator/engine.py` + `discrete_manufacturing_sim/simulation_handler.py` | Runs the preserved simulator and records trajectories |
| Buffer/state trajectory export | `src/process_timeseries/io/export.py` | Writes CSV/JSON artifacts for reproducible analysis |
| Paper-style multi-panel figure | `src/process_timeseries/plotting/figures.py` | Scriptable version of the notebook figure |
| 5-node demonstrator configuration | `configs/paper/wsc2024_demo_5_node.yaml` | Extracted from the tutorial notebook |
| One-command paper reproduction | `scripts/reproduce_wsc2024.py` | Canonical supported entry point |
| Thin tutorial / convenience layer | `notebooks/tutorials/introduction.ipynb` | No longer the only reproduction path |

## Behavioral reference

The implementation intentionally preserves the existing `discrete_manufacturing_sim` backbone as the behavioral reference while exposing a clearer public API under `src/process_timeseries/`.
