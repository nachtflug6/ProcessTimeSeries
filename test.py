import networkx as nx
import matplotlib.pyplot as plt
import torch

from discrete_manufacturing_sim.process_sim import ProdGraph
from discrete_manufacturing_sim.helpers import generate_random_dag, generate_random_distributions
from discrete_manufacturing_sim.fsm.production_asset_fsm import FSMProductionAsset

transition_fsm = FSMProductionAsset()


# def create_dists(n):
#     dists = []
#     for _ in range(n):  # Repeat n times
#         dists.append(
#             [
#                 torch.distributions.Uniform(9, 11),
#                 torch.distributions.Uniform(9, 11),
#                 torch.distributions.Uniform(9, 11),
#             ]
#         )
#     return dists


# n = 5

# adjacency_mtx = generate_random_dag(n, 0.2)
# G = nx.from_numpy_matrix(adjacency_mtx, create_using=nx.DiGraph)

# print(adjacency_mtx)

# adjacency_mtx = torch.from_numpy(adjacency_mtx)

# distribution_mtx = create_dists(n)

# capacity_mtx = torch.randint(1, 10, (n, 2))
# capacity_mtx[:, 0] = 1
# capacity_mtx[:, 1] = 5
# capacity_mtx[-1, -1] = 1e10


# graph = ProdGraph(
#     adjacency_mtx, distribution_mtx, capacity_mtx, transition_fsm, batch_size=40
# )
# # graph.update_event_required()

# # # Check if timed event
# # if th.sum(graph.timed_event_required_mtx) > 0:
# #     pass
# # else:
# #     print('nope')

# # graph.update_nodes_ready()
# # print(graph.nodes_ready)


# # Initialize the transition tensor for 5 states and 5 events
# num_states = 5
# num_events = 5
