#!/usr/bin/env python3
"""Convenience wrapper for exporting simulation trajectories without plotting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from process_timeseries.io.export import export_simulation_result
from process_timeseries.simulator.engine import load_config, run_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CSV/JSON trajectories from a simulation config.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--output-dir", help="Optional override for output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else ROOT / config.get("output_dir", "experiments/custom_run/outputs")

    result = run_simulation(config_path)
    artifacts = export_simulation_result(result, output_dir)

    print(f"Config used: {config_path}")
    print(f"Seed: {config.get('seed')}")
    print(f"Output directory: {output_dir}")
    print(f"Exports: {artifacts['buffer_levels']}, {artifacts['machine_states']}, {artifacts['summary']}")


if __name__ == "__main__":
    main()
