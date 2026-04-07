"""Distribution helpers for config-driven event timings."""

from __future__ import annotations

import random
from typing import Any, Mapping

import numpy as np
import torch

EVENT_ORDER = ("cycle_time", "failure_time", "repair_time")
SUPPORTED_DISTRIBUTIONS = {"uniform", "normal", "exponential"}


def set_global_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and Torch for reproducible simulation runs."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _canonicalize_spec(raw_spec: Mapping[str, Any], event_name: str) -> dict[str, Any]:
    if not raw_spec:
        raise ValueError(f"Missing distribution spec for event '{event_name}'")

    dist_type = str(raw_spec.get("type", "")).strip().lower()
    if not dist_type:
        raise ValueError(f"Distribution for event '{event_name}' must define a 'type'")
    if dist_type not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(f"Unsupported distribution type '{dist_type}' for event '{event_name}'")

    params = dict(raw_spec.get("params", {}))
    if not params:
        params = {key: value for key, value in raw_spec.items() if key != "type"}

    validate_distribution_spec(dist_type, params, event_name)
    return {"type": dist_type, "params": params}


def validate_distribution_spec(dist_type: str, params: Mapping[str, Any], event_name: str) -> None:
    """Validate supported timing distributions and their parameters."""
    if dist_type == "uniform":
        low = float(params["low"])
        high = float(params["high"])
        if low < 0 or high <= low:
            raise ValueError(f"Uniform distribution for '{event_name}' must satisfy 0 <= low < high")
    elif dist_type == "normal":
        mean = float(params["mean"])
        std_dev = float(params["std_dev"])
        if mean < 0:
            raise ValueError(f"Normal distribution mean for '{event_name}' must be non-negative")
        if std_dev <= 0:
            raise ValueError(f"Normal distribution std_dev for '{event_name}' must be positive")
    elif dist_type == "exponential":
        lambda_value = float(params["lambda"])
        if lambda_value <= 0:
            raise ValueError(f"Exponential distribution lambda for '{event_name}' must be positive")


def build_distribution_data(config: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    """Create explicit per-node distribution metadata from YAML config."""
    num_nodes = int(config.get("graph", {}).get("num_nodes", 0))
    events_config = config.get("events", {})
    defaults = events_config.get("defaults", {})
    per_node = events_config.get("per_node", {})
    if isinstance(per_node, list):
        per_node_lookup = {index: value for index, value in enumerate(per_node)}
    else:
        per_node_lookup = per_node

    distribution_data: list[list[dict[str, Any]]] = []
    for node_id in range(num_nodes):
        node_overrides = per_node_lookup.get(node_id, per_node_lookup.get(str(node_id), {}))
        node_data: list[dict[str, Any]] = []

        for event_name in EVENT_ORDER:
            raw_spec = dict(defaults.get(event_name, {}))
            raw_spec.update(node_overrides.get(event_name, {}))
            node_data.append(_canonicalize_spec(raw_spec, event_name))

        distribution_data.append(node_data)

    return distribution_data
