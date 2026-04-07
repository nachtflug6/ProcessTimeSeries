"""Minimal end-to-end simulation example.

Run with:
    python3 examples/minimal_simulation.py
"""

import torch

from discrete_manufacturing_sim.dg.wdg import WeightedDirectedGraph
from discrete_manufacturing_sim.fsm.production_asset_fsm import ProductionAssetFSM
from discrete_manufacturing_sim.generators.dg_generator import DGGenerator
from discrete_manufacturing_sim.generators.distribution_data_generator import DistributionDataGenerator
from discrete_manufacturing_sim.generators.multi_distribution_sampler import MultiDistributionSampler
from discrete_manufacturing_sim.multi_linear_petri_net import MultiLinearPetriNet
from discrete_manufacturing_sim.multi_temporal_event_handler import MultiTemporalEventHandler
from discrete_manufacturing_sim.simulation_handler import SimulationHandler


def build_demo(num_nodes: int = 2, length_pns: int = 2) -> SimulationHandler:
    """Construct a small simulation handler for smoke testing and demos."""
    adjacency_matrix = DGGenerator(num_nodes=num_nodes).generate_linear_dag()
    dist_gen = DistributionDataGenerator(
        torch.ones(num_nodes, 3) * 10,
        distribution_types=["uniform"],
        uniform_params={"spread": 2},
    )

    connectivity_graph = WeightedDirectedGraph(adjacency_matrix)
    multi_lin_pn = MultiLinearPetriNet(
        length_pns=length_pns,
        connectivity_graph=connectivity_graph,
    )
    pfsm = ProductionAssetFSM()
    mteh = MultiTemporalEventHandler(MultiDistributionSampler(dist_gen))
    return SimulationHandler(multi_lin_pn, pfsm, mteh)


def run_demo(steps: int = 25) -> SimulationHandler:
    """Run a small simulation and print a short summary."""
    sim_handler = build_demo()
    for _ in range(steps):
        sim_handler.simulate()

    print("Simulation finished.")
    print("State matrix:\n", sim_handler.state_matrix)
    print("Markings:\n", sim_handler.mlin_pn.markings)
    return sim_handler


if __name__ == "__main__":
    run_demo()
