"""Simulator-facing public exports.

This package intentionally wraps the existing `discrete_manufacturing_sim`
backbone rather than rewriting it, so the paper demo remains behaviorally
consistent while the repository gets a cleaner public surface.
"""

from process_timeseries.simulator.dag import (
    build_adjacency_matrix,
    build_graph_from_config,
    topological_order,
    validate_dag,
)
from process_timeseries.simulator.engine import (
    build_simulation_handler,
    load_config,
    run_simulation,
    validate_config,
)
from process_timeseries.simulator.events import EventType, MachineState
from process_timeseries.simulator.fsm import ProductionAssetFSM
from process_timeseries.simulator.petri import MultiLinearPetriNet
from process_timeseries.simulator.scheduler import ScheduledEvent, preview_next_event, select_minimum_time_event
from process_timeseries.simulator.state import SimulationResult

__all__ = [
    "build_adjacency_matrix",
    "build_graph_from_config",
    "build_simulation_handler",
    "EventType",
    "MachineState",
    "MultiLinearPetriNet",
    "ProductionAssetFSM",
    "ScheduledEvent",
    "SimulationResult",
    "load_config",
    "preview_next_event",
    "run_simulation",
    "select_minimum_time_event",
    "topological_order",
    "validate_config",
    "validate_dag",
]
