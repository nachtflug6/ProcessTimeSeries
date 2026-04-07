"""High-level helpers for the paper-facing minimal simulation demo.

This module keeps the public demo setup compact by hiding the lower-level
construction details behind a small configuration object and a few helper
functions.
"""

import random
from dataclasses import dataclass

import torch

from discrete_manufacturing_sim.dg.wdg import WeightedDirectedGraph
from discrete_manufacturing_sim.fsm.production_asset_fsm import ProductionAssetFSM
from discrete_manufacturing_sim.generators.multi_distribution_sampler import MultiDistributionSampler
from discrete_manufacturing_sim.multi_linear_petri_net import MultiLinearPetriNet
from discrete_manufacturing_sim.multi_temporal_event_handler import MultiTemporalEventHandler
from discrete_manufacturing_sim.simulation_handler import SimulationHandler


@dataclass(frozen=True)
class DemoConfig:
    """Configuration for the minimal paper/demo simulation."""

    num_nodes: int = 2
    length_pns: int = 2
    num_distributions: int = 3
    mean: float = 10.0
    spread: float = 2.0
    steps: int = 25
    distribution_type: str = "uniform"
    std_dev: float = 0.4
    seed: int | None = 0
    event_means: tuple[float, float, float] | None = None
    capacities: tuple[float, ...] | None = None


def build_linear_adjacency(num_nodes: int) -> torch.Tensor:
    """Create a simple linear production chain adjacency matrix."""
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for i in range(num_nodes - 1):
        adjacency[i, i + 1] = 1
    return adjacency


def build_uniform_distribution_data(config: DemoConfig):
    """Create uniform distribution metadata for each node/event pair."""
    low = config.mean - config.spread / 2
    high = config.mean + config.spread / 2

    return [
        [
            {"type": "uniform", "params": {"low": low, "high": high}}
            for _ in range(config.num_distributions)
        ]
        for _ in range(config.num_nodes)
    ]


def build_distribution_data(config: DemoConfig):
    """Create event timing metadata for the requested demo distribution type."""
    distribution_type = config.distribution_type.lower()
    if distribution_type == "uniform":
        return build_uniform_distribution_data(config)

    if distribution_type == "normal":
        means = config.event_means or (
            config.mean,
            config.mean * 100,
            config.mean * 20,
        )
        if len(means) != config.num_distributions:
            raise ValueError("event_means must match num_distributions for the demo")
        return [
            [
                {
                    "type": "normal",
                    "params": {"mean": float(means[event_idx]), "std_dev": config.std_dev},
                }
                for event_idx in range(config.num_distributions)
            ]
            for _ in range(config.num_nodes)
        ]

    raise ValueError(f"Unsupported demo distribution_type: {config.distribution_type}")


def build_demo(config: DemoConfig | None = None) -> SimulationHandler:
    """Construct the minimal end-to-end simulation used by the demo and paper."""
    config = config or DemoConfig()

    if config.seed is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)

    adjacency_matrix = build_linear_adjacency(config.num_nodes)
    connectivity_graph = WeightedDirectedGraph(adjacency_matrix)
    petri_net = MultiLinearPetriNet(
        length_pns=config.length_pns,
        connectivity_graph=connectivity_graph,
    )

    if config.capacities is not None:
        if len(config.capacities) != config.num_nodes:
            raise ValueError("capacities must provide one output capacity per node")
        for node_idx, output_capacity in enumerate(config.capacities):
            petri_net.set_capacity(node_idx, config.length_pns - 1, output_capacity)

    fsm = ProductionAssetFSM()

    distribution_data = build_distribution_data(config)
    sampler = MultiDistributionSampler(
        config.num_nodes,
        config.num_distributions,
        distribution_data,
    )
    event_handler = MultiTemporalEventHandler(sampler)

    return SimulationHandler(petri_net, fsm, event_handler)


def summarize_simulation(sim_handler: SimulationHandler) -> dict:
    """Return a small structured summary for docs, notebooks, and tests."""
    return {
        "runtime": sim_handler.multi_temp_event_handler.runtime,
        "state_matrix": sim_handler.state_matrix,
        "markings": sim_handler.mlin_pn.markings,
        "num_nodes": sim_handler.num_nodes,
        "num_states": sim_handler.num_states,
    }


def run_demo(
    config: DemoConfig | None = None,
    steps: int | None = None,
    verbose: bool = True,
) -> SimulationHandler:
    """Run the minimal simulation and optionally print a summary."""
    config = config or DemoConfig()
    total_steps = config.steps if steps is None else steps

    sim_handler = build_demo(config)
    for _ in range(total_steps):
        sim_handler.simulate()

    if verbose:
        summary = summarize_simulation(sim_handler)
        print("Simulation finished.")
        print(f"Runtime: {summary['runtime']:.3f}")
        print("State matrix:\n", summary["state_matrix"])
        print("Markings:\n", summary["markings"])

    return sim_handler
