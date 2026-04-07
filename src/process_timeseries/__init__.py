"""Public package interface for the ProcessTimeSeries research simulator."""

from process_timeseries.simulator.engine import (
    build_simulation_handler,
    load_config,
    run_simulation,
    validate_config,
)
from process_timeseries.simulator.state import SimulationResult

__all__ = [
    "build_simulation_handler",
    "load_config",
    "run_simulation",
    "validate_config",
    "SimulationResult",
]
