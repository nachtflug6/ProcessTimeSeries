"""Export simulation trajectories and metadata to reproducible artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from process_timeseries.simulator.state import SimulationResult


def export_simulation_result(result: SimulationResult, output_dir: str | Path) -> dict[str, Path]:
    """Write the standard CSV/JSON outputs for a completed simulation run."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    buffer_path = output_path / "buffer_levels.csv"
    machine_state_path = output_path / "machine_states.csv"
    summary_path = output_path / "summary.json"

    result.to_buffer_dataframe().to_csv(buffer_path, index=False)
    result.to_state_dataframe().to_csv(machine_state_path, index=False)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result.summary(), handle, indent=2)

    return {
        "buffer_levels": buffer_path,
        "machine_states": machine_state_path,
        "summary": summary_path,
    }
