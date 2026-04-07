#!/usr/bin/env python3
"""Validate a YAML config before running the simulator."""

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

from process_timeseries.simulator.dag import build_adjacency_matrix, topological_order
from process_timeseries.simulator.engine import load_config, validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a manufacturing simulation YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    validate_config(config)

    adjacency = build_adjacency_matrix(config["graph"]["num_nodes"], config["graph"]["edges"])
    order = topological_order(adjacency)

    print("Config validation passed.")
    print(f"Config used: {config_path}")
    print(f"Seed: {config.get('seed')}")
    print(f"Output directory: {config.get('output_dir')}")
    print(f"Nodes: {config['graph']['num_nodes']}")
    print(f"Topological order: {order}")


if __name__ == "__main__":
    main()
