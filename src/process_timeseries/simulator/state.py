"""Structured result containers for simulation trajectories and summaries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from process_timeseries.simulator.events import MachineState, state_name


@dataclass
class SimulationResult:
    """Structured outputs from a config-driven simulation run."""

    config_path: str | None
    seed: int | None
    times: list[float]
    buffer_levels: np.ndarray
    machine_states: np.ndarray
    adjacency_matrix: np.ndarray
    total_events: int
    end_time: float | None = None
    output_dir: str | None = None

    def to_buffer_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for row_idx, timestamp in enumerate(self.times):
            for node_id in range(self.buffer_levels.shape[1]):
                rows.append(
                    {
                        "time": float(timestamp),
                        "node_id": int(node_id),
                        "buffer_level": float(self.buffer_levels[row_idx, node_id]),
                    }
                )
        return pd.DataFrame(rows)

    def to_state_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for row_idx, timestamp in enumerate(self.times):
            for node_id in range(self.machine_states.shape[1]):
                state_id = int(self.machine_states[row_idx, node_id])
                rows.append(
                    {
                        "time": float(timestamp),
                        "node_id": int(node_id),
                        "state_id": state_id,
                        "state": state_name(state_id),
                    }
                )
        return pd.DataFrame(rows)

    def failure_counts_by_node(self) -> dict[str, int]:
        if len(self.machine_states) <= 1:
            return {str(node_id): 0 for node_id in range(self.machine_states.shape[1])}

        failure_counts: dict[str, int] = {}
        for node_id in range(self.machine_states.shape[1]):
            previous = self.machine_states[:-1, node_id]
            current = self.machine_states[1:, node_id]
            count = int(np.sum((previous != int(MachineState.FAILED)) & (current == int(MachineState.FAILED))))
            failure_counts[str(node_id)] = count
        return failure_counts

    def summary(self) -> dict[str, Any]:
        final_levels = (
            self.buffer_levels[-1].tolist() if self.buffer_levels.size else [0.0] * self.adjacency_matrix.shape[0]
        )
        output_dir = str(Path(self.output_dir)) if self.output_dir else None
        end_time = self.end_time
        if end_time is None:
            end_time = float(self.times[-1]) if self.times else 0.0

        return {
            "seed": self.seed,
            "config_path": self.config_path,
            "simulation_end_time": float(end_time),
            "total_events": int(self.total_events),
            "final_production_count_by_node": {str(i): float(value) for i, value in enumerate(final_levels)},
            "failure_counts_by_node": self.failure_counts_by_node(),
            "output_dir": output_dir,
        }
