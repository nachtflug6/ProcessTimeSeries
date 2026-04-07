from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from process_timeseries.io.export import export_simulation_result
from process_timeseries.plotting.figures import plot_paper_figure
from process_timeseries.simulator.engine import load_config, run_simulation, validate_config

ROOT = Path(__file__).resolve().parents[1]
PAPER_CONFIG = ROOT / "configs" / "paper" / "wsc2024_demo_5_node.yaml"


def test_paper_config_loads() -> None:
    config = load_config(PAPER_CONFIG)
    validate_config(config)

    assert config["graph"]["num_nodes"] == 5
    assert len(config["graph"]["edges"]) == 4


def test_paper_demo_outputs_are_created(tmp_path: Path) -> None:
    config = load_config(PAPER_CONFIG)
    config["stop_conditions"] = {"max_time": 1200}
    config["output_dir"] = str(tmp_path)

    result = run_simulation(config)
    artifacts = export_simulation_result(result, tmp_path)
    figure_path = tmp_path / "figure5.png"
    plot_paper_figure(result, output_path=figure_path)
    plt.close("all")

    assert result.total_events > 0
    assert result.summary()["simulation_end_time"] > 0
    assert artifacts["buffer_levels"].exists()
    assert artifacts["machine_states"].exists()
    assert artifacts["summary"].exists()
    assert figure_path.exists()
