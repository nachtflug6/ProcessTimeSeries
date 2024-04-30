import torch

from discrete_manufacturing_sim.dg.wdg import WeightedDirectedGraph
class MultiPetriNet:
    def __init__(self, num_pns, length_pns=2, connectivity_graph=None):
        self.num_pns = num_pns
        self.length_pns = length_pns
        self.connectivity_graph = connectivity_graph

        # Initialize tensors for markings, capacities, and weights
        self.markings = torch.zeros((num_pns, length_pns))
        self.capacities = torch.ones((num_pns, length_pns))
        self.weights = torch.ones((num_pns, length_pns))

        # Set weights according to the specified structure
        for i in range(length_pns - 1):
            weight_value = i * 2 + 1  # Calculate weight value based on index
            self.weights[:, i * 2] = weight_value  # Set transition-place weights
            self.weights[:, i * 2 + 1] = weight_value  # Set place-transition weights
        
        self.active_transitions = torch.zeros((self.num_pns, self.length_pns - 1), dtype=torch.bool)
        
        # Connect Petri net modules based on the connectivity graph
        if connectivity_graph is not None:
            self.connect_modules(connectivity_graph)

    def set_initial_marking(self, pn_idx, marking):
        self.markings[pn_idx] = torch.tensor(marking)

    def set_capacity(self, pn_idx, place_idx, capacity):
        self.capacities[pn_idx, place_idx] = capacity

    def set_weight(self, pn_idx, weight_idx, weight):
        self.weights[pn_idx, weight_idx] = weight

    def fire_transition(self, pn_idx, transition_idx):
        # Calculate the corresponding place index for the given transition
        place_idx = transition_idx // 2

        # Check if the transition is enabled
        if self.is_enabled(pn_idx, place_idx):
            # Update markings
            self.markings[pn_idx, place_idx] -= self.weights[pn_idx, transition_idx]
            
            # Update active transitions after firing
            self.update_active_transitions(pn_idx)

    def is_enabled(self, pn_idx, place_idx):
        # Check if the input place has enough tokens to fire the transition
        return self.markings[pn_idx, place_idx] >= self.weights[pn_idx, place_idx]
    
    def update_active_transitions(self, pn_idx):
        # Iterate through each place in the Petri net
        for place_idx in range(self.length_pns - 1):
            # Check if the transition is enabled for the current place
            self.active_transitions[pn_idx, place_idx] = self.is_enabled(pn_idx, place_idx)
    
    def connect_modules(self, connectivity_graph):
        if self.connectivity_graph.num_nodes != self.num_pns:
            raise ValueError("Number of nodes in the connectivity graph must match the number of Petri net modules")
        
        for source_node in range(self.num_pns):
            next_nodes = connectivity_graph.get_next_nodes(source_node, indices=True)
            for destination_node in next_nodes:
                # Set weight of transition from last place of source node to first transition of destination node
                weight = connectivity_graph.adjacency_matrix[source_node, destination_node]
                transition_idx = (self.length_pns - 1) * 2
                self.set_weight(source_node, transition_idx, weight)

    def print_state(self):
        for i in range(self.num_pns):
            print(f"Petri Net {i} Markings: {self.markings[i]}")
            print(f"Petri Net {i} Capacities: {self.capacities[i]}")
            print(f"Petri Net {i} Weights: {self.weights[i]}")

# Functionality Check 1: Create a MultiPetriNet instance with length 2 and connectivity graph
adjacency_matrix = torch.tensor([[0, 1],  # Connection from PN 0 to PN 1
                                 [0, 0]])  # No connections from PN 1
states = [None, None]  # Placeholder states for the nodes
connectivity_graph = WeightedDirectedGraph(adjacency_matrix, states)
mpn = MultiPetriNet(num_pns=2, length_pns=2, connectivity_graph=connectivity_graph)

# Set initial markings
mpn.set_initial_marking(0, [1, 0])
mpn.set_initial_marking(1, [0, 1])

# Set capacities
mpn.set_capacity(0, 0, 1)
mpn.set_capacity(1, 0, 1)

# Set weights
mpn.set_weight(0, 0, 1)  # Set weight of transition t0 in PN 0
mpn.set_weight(1, 3, 2)  # Set weight of transition t0 in PN 1

# Fire transition t0 in PN 0
mpn.fire_transition(0, 0)

# Print current state
mpn.print_state()

# Functionality Check 2: Check active transitions after firing
active_transitions = mpn.active_transitions
print("Active Transitions after firing t0 in PN 0:")
print(active_transitions)