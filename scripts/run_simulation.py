#!/usr/bin/env python3
"""Run a custom simulation from a YAML config."""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DAG-based manufacturing simulation from YAML.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--output-dir", help="Optional override for the output directory.")
    parser.add_argument("--no-plot", action="store_true", help="Skip figure generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else ROOT / config.get("output_dir", "experiments/custom_run/outputs")

    start = time.perf_counter()
    result = run_simulation(config_path)
    artifacts = export_simulation_result(result, output_dir)

    if not args.no_plot:
        figure_path = output_dir / "figure.png"
        plot_paper_figure(result, output_path=figure_path)
        plt.close("all")
    else:
        figure_path = None

    elapsed = time.perf_counter() - start
    print(f"Config used: {config_path}")
    print(f"Seed: {config.get('seed')}")
    print(f"Runtime: {elapsed:.2f} s")
    print(f"Output directory: {output_dir}")
    print(f"Simulation end time: {result.summary()['simulation_end_time']:.3f} s")
    print(f"Total events: {result.total_events}")
    if figure_path is not None:
        print(f"Figure: {figure_path}")
    print(f"Exports: {artifacts['buffer_levels']}, {artifacts['machine_states']}, {artifacts['summary']}")


if __name__ == "__main__":
    main()
