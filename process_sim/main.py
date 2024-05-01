import torch

from discrete_manufacturing_sim.multi_linear_petri_net import MultiLinearPetriNet
from discrete_manufacturing_sim.dg.wdg import WeightedDirectedGraph


# Functionality Check 1: Create a MultiPetriNet instance with length 2 and connectivity graph
adjacency_matrix = torch.tensor([[0, 1],  # Connection from PN 0 to PN 1
                                 [0, 0]])  # No connections from PN 1
states = [None, None]  # Placeholder states for the nodes
connectivity_graph = WeightedDirectedGraph(adjacency_matrix)
mpn = MultiLinearPetriNet(length_pns=2, connectivity_graph=connectivity_graph)


mpn.update_active_transitions()
mpn.print_state()
mpn.fire_transition(0, 0)

mpn.print_state()
mpn.fire_transition(0, 1)

mpn.print_state()
mpn.fire_transition(0, 0)

mpn.print_state()
mpn.fire_transition(1, 0)
# # Set initial markings
# mpn.set_initial_marking(0, [1, 0])
# mpn.set_initial_marking(1, [0, 1])

# # Set capacities
# mpn.set_capacity(0, 0, 1)
# mpn.set_capacity(1, 0, 1)

# # Set weights
# mpn.set_weight(0, 0, 1)  # Set weight of transition t0 in PN 0
# mpn.set_weight(1, 1, 2)  # Set weight of transition t0 in PN 1

# # Fire transition t0 in PN 0
# mpn.fire_transition(0, 0)

# # Print current state
mpn.print_state()

# # Functionality Check 2: Check active transitions after firing
# active_transitions = mpn.active_transitions
# print("Active Transitions after firing t0 in PN 0:")
# print(active_transitions)
