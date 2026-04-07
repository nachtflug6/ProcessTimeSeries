"""Config-driven engine helpers for reproducible simulation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from discrete_manufacturing_sim.generators.multi_distribution_sampler import MultiDistributionSampler
from discrete_manufacturing_sim.multi_linear_petri_net import MultiLinearPetriNet
from discrete_manufacturing_sim.multi_temporal_event_handler import MultiTemporalEventHandler
from discrete_manufacturing_sim.simulation_handler import SimulationHandler

from process_timeseries.simulator.dag import build_graph_from_config
from process_timeseries.simulator.distributions import EVENT_ORDER, build_distribution_data, set_global_seed
from process_timeseries.simulator.fsm import ProductionAssetFSM
from process_timeseries.simulator.state import SimulationResult


def load_config(config_or_path: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Load a YAML config from disk, or normalize an existing mapping."""
    if isinstance(config_or_path, Mapping):
        return dict(config_or_path)

    config_path = Path(config_or_path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file '{config_path}' must deserialize to a mapping")
    return data


def _resolve_config_and_path(config_or_path: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    if isinstance(config_or_path, Mapping):
        return dict(config_or_path), None
    config_path = Path(config_or_path).resolve()
    return load_config(config_path), str(config_path)


def _validate_matrix(name: str, values: list[list[float]], num_nodes: int, length_pns: int) -> None:
    if len(values) != num_nodes:
        raise ValueError(f"petri.{name} must contain exactly {num_nodes} rows")

    for row_index, row in enumerate(values):
        if len(row) != length_pns:
            raise ValueError(f"petri.{name}[{row_index}] must have length {length_pns}")

        if name == "capacities" and any(float(value) <= 0 for value in row):
            raise ValueError("All capacities must be strictly positive")
        if name == "weights" and any(float(value) <= 0 for value in row):
            raise ValueError("All Petri-net weights must be strictly positive")
        if name == "initial_markings" and any(float(value) < 0 for value in row):
            raise ValueError("Initial markings must be non-negative")


def validate_config(config_or_path: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    """Validate the high-level simulation config and return the normalized mapping."""
    config = load_config(config_or_path)

    graph_config = config.get("graph")
    if not isinstance(graph_config, dict):
        raise ValueError("Config must define a 'graph' section")
    if int(graph_config.get("num_nodes", 0)) <= 0:
        raise ValueError("graph.num_nodes must be a positive integer")
    if not graph_config.get("edges"):
        raise ValueError("graph.edges must define at least one directed edge")

    build_graph_from_config(config)

    petri_config = config.get("petri", {})
    length_pns = int(petri_config.get("length_pns", 2))
    if length_pns < 2:
        raise ValueError("petri.length_pns must be at least 2 for the current manufacturing model")

    num_nodes = int(graph_config["num_nodes"])
    for name in ("capacities", "initial_markings", "weights"):
        if name in petri_config:
            _validate_matrix(name, petri_config[name], num_nodes, length_pns)

    stop_conditions = config.get("stop_conditions", {})
    if not any(key in stop_conditions for key in ("max_time", "max_events", "max_output")):
        raise ValueError("At least one stop condition is required: max_time, max_events, or max_output")

    build_distribution_data(config)
    return config


def _matrix_from_config(config: Mapping[str, Any], key: str, default_row: list[float]) -> list[list[float]]:
    graph_config = config.get("graph", {})
    num_nodes = int(graph_config.get("num_nodes", 0))
    petri_config = config.get("petri", {})
    values = petri_config.get(key)
    if values is None:
        return [list(default_row) for _ in range(num_nodes)]
    return [[float(item) for item in row] for row in values]


def build_simulation_handler(config_or_path: str | Path | Mapping[str, Any]) -> SimulationHandler:
    """Construct a simulation handler from a validated YAML config."""
    config = validate_config(config_or_path)
    set_global_seed(config.get("seed"))

    graph = build_graph_from_config(config)
    petri_config = config.get("petri", {})
    length_pns = int(petri_config.get("length_pns", 2))

    petri_net = MultiLinearPetriNet(connectivity_graph=graph, length_pns=length_pns)

    capacities = _matrix_from_config(config, "capacities", [1.0] * length_pns)
    initial_markings = _matrix_from_config(config, "initial_markings", [0.0] * length_pns)
    weights = _matrix_from_config(config, "weights", [1.0] * length_pns)

    for node_id in range(graph.num_nodes):
        petri_net.set_initial_marking(node_id, initial_markings[node_id])
        for place_idx in range(length_pns):
            petri_net.set_capacity(node_id, place_idx, capacities[node_id][place_idx])
            petri_net.set_weight(node_id, place_idx, weights[node_id][place_idx])

    fsm = ProductionAssetFSM()
    distribution_data = build_distribution_data(config)
    sampler = MultiDistributionSampler(graph.num_nodes, len(EVENT_ORDER), distribution_data)
    event_handler = MultiTemporalEventHandler(sampler)

    return SimulationHandler(petri_net, fsm, event_handler)


def run_simulation(config_or_path: str | Path | Mapping[str, Any]) -> SimulationResult:
    """Run a full simulation and collect structured trajectories over time."""
    config, config_path = _resolve_config_and_path(config_or_path)
    validate_config(config)
    sim_handler = build_simulation_handler(config)

    stop_conditions = config.get("stop_conditions", {})
    max_time = float(stop_conditions.get("max_time", np.inf))
    max_events = int(stop_conditions.get("max_events", np.iinfo(np.int32).max))
    max_output = float(stop_conditions.get("max_output", np.inf))

    times: list[float] = []
    buffer_rows: list[np.ndarray] = []
    state_rows: list[np.ndarray] = []
    event_count = 0

    while True:
        current_time = float(sim_handler.multi_temp_event_handler.runtime)
        current_output = float(sim_handler.mlin_pn.markings[-1, -1].item())
        if current_time >= max_time or event_count >= max_events or current_output >= max_output:
            break

        sim_handler.simulate()

        state_matrix = sim_handler.state_matrix.detach().cpu().numpy()
        state_indices = np.argmax(state_matrix, axis=1)
        buffer_snapshot = sim_handler.mlin_pn.markings[:, -1].detach().cpu().numpy().copy()

        times.append(current_time)
        state_rows.append(state_indices.copy())
        buffer_rows.append(buffer_snapshot)
        event_count += 1

    num_nodes = sim_handler.num_nodes
    buffer_levels = np.asarray(buffer_rows, dtype=float) if buffer_rows else np.zeros((0, num_nodes), dtype=float)
    machine_states = np.asarray(state_rows, dtype=int) if state_rows else np.zeros((0, num_nodes), dtype=int)
    adjacency_matrix = sim_handler.mlin_pn.connectivity_graph.adjacency_matrix.detach().cpu().numpy().copy()

    return SimulationResult(
        config_path=config_path,
        seed=config.get("seed"),
        times=times,
        buffer_levels=buffer_levels,
        machine_states=machine_states,
        adjacency_matrix=adjacency_matrix,
        total_events=event_count,
        end_time=float(sim_handler.multi_temp_event_handler.runtime),
        output_dir=config.get("output_dir"),
    )


__all__ = ["build_simulation_handler", "load_config", "run_simulation", "validate_config"]
