from __future__ import annotations

from pathlib import Path

import numpy as np

from process_timeseries.simulator.engine import run_simulation

ROOT = Path(__file__).resolve().parents[1]


def test_tiny_simulation_runs_with_valid_outputs() -> None:
    result = run_simulation(ROOT / "configs" / "examples" / "simple_line_3_node.yaml")

    assert result.total_events > 0
    assert len(result.times) > 0
    assert result.buffer_levels.shape[0] == len(result.times)
    assert result.machine_states.shape[0] == len(result.times)
    assert np.all(np.diff(result.times) >= 0)
    assert np.all(result.buffer_levels >= 0)
    assert set(np.unique(result.machine_states)).issubset({0, 1, 2, 3})
