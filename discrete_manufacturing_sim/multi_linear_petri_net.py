import torch
import pandas as pd

from discrete_manufacturing_sim.dg.wdg import WeightedDirectedGraph


class MultiLinearPetriNet:
    def __init__(self, connectivity_graph: WeightedDirectedGraph, length_pns=2):
        num_pns = connectivity_graph.num_nodes
        self.num_pns = num_pns
        self.length_pns = length_pns
        self.connectivity_graph = connectivity_graph
        self.epsilon = 1e-6

        # Initialize tensors for markings, capacities, and weights
        self.markings = torch.zeros((num_pns, length_pns))
        self.capacities = torch.ones((num_pns, length_pns))
        self.weights = torch.ones((num_pns, length_pns))

        self.active_transitions = torch.zeros(
            (self.num_pns, self.length_pns), dtype=torch.bool)

    def set_initial_marking(self, pn_idx, marking):
        if pn_idx >= self.num_pns:
            raise ValueError("Petri net index is out of bounds.")
        if len(marking) != self.length_pns:
            raise ValueError(
                "Length of marking does not match the length of Petri net.")
        self.markings[pn_idx] = torch.tensor(marking)

    def set_capacity(self, pn_idx, place_idx, capacity):
        if pn_idx >= self.num_pns or place_idx >= self.length_pns:
            raise ValueError(
                "Petri net index or place index is out of bounds.")
        self.capacities[pn_idx, place_idx] = capacity

    def set_weight(self, pn_idx, weight_idx, weight):
        if pn_idx >= self.num_pns or weight_idx >= self.length_pns:
            raise ValueError(
                "Petri net index or weight index is out of bounds.")
        self.weights[pn_idx, weight_idx] = weight

    def fire_transition(self, pn_idx, transition_idx):
        if pn_idx >= self.num_pns or transition_idx >= self.length_pns:
            raise ValueError(
                "Petri net index or transition index is out of bounds.")

        # Check if the transition is enabled
        if self.active_transitions[pn_idx, transition_idx]:
            if transition_idx == 0:
                # Update markings
                self.markings[:, -
                              1] -= self.connectivity_graph.get_previous_nodes(pn_idx)
            else:
                self.markings[pn_idx, transition_idx -
                              1] -= self.weights[pn_idx, transition_idx - 1]
            self.markings[pn_idx,
                          transition_idx] += self.weights[pn_idx, transition_idx]

            # Update active transitions after firing
            self.update_active_transitions()
        else:
            print('cannot fire')

    def update_active_transitions(self):
        active_transitions = self.active_transitions
        M_transposed = self.connectivity_graph.adjacency_matrix.t()
        # Get the last markings
        v_t = self.markings[:, -1].t()
        epsilon = self.epsilon
        # Divide adjacency by markings
        result = M_transposed / (v_t + epsilon)
        final_result = result.t()
        # Take max value in each column and select the columns where the max is <1
        column_maxes = final_result.max(dim=0)[0]
        binary_results = (column_maxes < 1).bool()
        active_transitions[:, 0] = binary_results

        for place_idx in range(self.length_pns - 1):
            active_transitions[:, 1 + place_idx] = self.markings[:,
                                                                 place_idx] >= self.weights[:, place_idx]

        self.active_transitions = active_transitions & self.check_capacity()

    def check_capacity(self):
        if torch.any(self.markings > self.capacities):
            raise ValueError("Marking exceeds capacity in one or more places.")
        return self.markings + self.weights <= self.capacities

    def print_state(self):
        markings_df = pd.DataFrame(self.markings.numpy(), columns=[
                                   f"m{i}" for i in range(self.num_pns)])
        capacities_df = pd.DataFrame(self.capacities.numpy(), columns=[
                                     f"c{i}" for i in range(self.num_pns)])
        weights_df = pd.DataFrame(self.weights.numpy(), columns=[
                                  f"w{i}" for i in range(self.num_pns)])
        active_transitions_df = pd.DataFrame(self.active_transitions.numpy(), columns=[
                                             f"t{i}" for i in range(self.num_pns)])

        state_df = pd.concat([active_transitions_df] +
                             [capacities_df.add_prefix('c')] +
                             [markings_df.add_prefix('m')] +
                             [weights_df.add_prefix('w')], axis=1)
        state_df = state_df.reindex(sorted(state_df.columns), axis=1)
        print(state_df)


# # Functionality Check 1: Create a MultiPetriNet instance with length 2 and connectivity graph
# adjacency_matrix = torch.tensor([[0, 1],  # Connection from PN 0 to PN 1
#                                  [0, 0]])  # No connections from PN 1
# states = [None, None]  # Placeholder states for the nodes
# connectivity_graph = WeightedDirectedGraph(adjacency_matrix)
# mpn = MultiLinearPetriNet(length_pns=2, connectivity_graph=connectivity_graph)

# mpn.update_active_transitions()
# mpn.print_state()
# mpn.fire_transition(0, 0)

# mpn.print_state()
# mpn.fire_transition(0, 1)

# mpn.print_state()
# mpn.fire_transition(0, 0)

# mpn.print_state()
# mpn.fire_transition(1, 0)
# # # Set initial markings
# # mpn.set_initial_marking(0, [1, 0])
# # mpn.set_initial_marking(1, [0, 1])

# # # Set capacities
# # mpn.set_capacity(0, 0, 1)
# # mpn.set_capacity(1, 0, 1)

# # # Set weights
# # mpn.set_weight(0, 0, 1)  # Set weight of transition t0 in PN 0
# # mpn.set_weight(1, 1, 2)  # Set weight of transition t0 in PN 1

# # # Fire transition t0 in PN 0
# # mpn.fire_transition(0, 0)

# # # Print current state
# mpn.print_state()

# # # Functionality Check 2: Check active transitions after firing
# # active_transitions = mpn.active_transitions
# # print("Active Transitions after firing t0 in PN 0:")
# # print(active_transitions)
