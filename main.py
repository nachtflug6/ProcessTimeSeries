import torch

from discrete_manufacturing_sim.multi_linear_petri_net import MultiLinearPetriNet
from discrete_manufacturing_sim.dg.wdg import WeightedDirectedGraph
from discrete_manufacturing_sim.fsm.production_asset_fsm import ProductionAssetFSM
from discrete_manufacturing_sim.simulation_handler import SimulationHandler
# from discrete_manufacturing_sim.multi_temporal_event_handler import MultiTemporalEventHandler
from discrete_manufacturing_sim.generators.dg_generator import DGGenerator
from discrete_manufacturing_sim.generators.distribution_data_generator import DistributionDataGenerator
from discrete_manufacturing_sim.generators.multi_distribution_sampler import MultiDistributionSampler
from discrete_manufacturing_sim.multi_temporal_event_handler import MultiTemporalEventHandler


num_nodes = 2
length_pns = 2

adj_gen = DGGenerator(num_nodes=num_nodes)
dist_gen = DistributionDataGenerator(torch.ones(
    num_nodes, 3) * 10, distribution_types=['uniform'])


adjacency_matrix = adj_gen.generate_linear_dag()

print(adjacency_matrix)

# Multi Petri Net
connectivity_graph = WeightedDirectedGraph(adjacency_matrix)
multi_lin_pn = MultiLinearPetriNet(
    length_pns=length_pns, connectivity_graph=connectivity_graph)

# State Machine
pfsm = ProductionAssetFSM()

# Temporal Event Handling
mds = MultiDistributionSampler(dist_gen)
mteh = MultiTemporalEventHandler(mds)


sim_handler = SimulationHandler(multi_lin_pn, pfsm, mteh)

for i in range(1000):
    sim_handler.simulate()
