#!/usr/bin/env python3
"""One-command reproduction path for the paper's 5-node manufacturing demo."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from process_timeseries.io.export import export_simulation_result
from process_timeseries.plotting.figures import plot_paper_figure
from process_timeseries.simulator.engine import load_config, run_simulation


def main() -> None:
    config_path = ROOT / "configs" / "paper" / "wsc2024_demo_5_node.yaml"
    config = load_config(config_path)
    output_dir = ROOT / config.get("output_dir", "experiments/wsc2024/outputs")

    start = time.perf_counter()
    result = run_simulation(config_path)
    artifacts = export_simulation_result(result, output_dir)
    figure_path = output_dir / "figure5.png"
    plot_paper_figure(result, output_path=figure_path)
    plt.close("all")
    elapsed = time.perf_counter() - start

    summary = result.summary()
    print(f"Config used: {config_path}")
    print(f"Seed: {config.get('seed')}")
    print(f"Simulation end time: {summary['simulation_end_time']:.3f} s")
    print(f"Total events: {result.total_events}")
    print(f"Runtime: {elapsed:.2f} s")
    print(f"Output directory: {output_dir}")
    print("Artifacts:")
    print(f"  - {artifacts['buffer_levels']}")
    print(f"  - {artifacts['machine_states']}")
    print(f"  - {artifacts['summary']}")
    print(f"  - {figure_path}")


if __name__ == "__main__":
    main()
