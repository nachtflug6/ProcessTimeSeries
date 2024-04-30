import torch

class GenericDirectedGraph:
    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix
        self.num_nodes = adjacency_matrix.size(0)
        self.node_features = {}

    def add_edge(self, source_node, destination_node):
        self.adjacency_matrix[source_node, destination_node] = 1

    def remove_edge(self, source_node, destination_node):
        self.adjacency_matrix[source_node, destination_node] = 0

    def add_feature(self, key, values):
        if len(values) != self.num_nodes:
            raise ValueError("Number of values must be equal to the number of nodes")
        self.node_features[key] = values

    def remove_feature(self, key):
        if key in self.node_features:
            del self.node_features[key]

    def get_previous_nodes(self, node, indices=False):
        if indices:
            return torch.nonzero(self.adjacency_matrix[:, node]).squeeze()
        return self.adjacency_matrix[:, node]
    
    def get_next_nodes(self, node, indices=False):
        if indices:
            return torch.nonzero(self.adjacency_matrix[node]).squeeze()
        return self.adjacency_matrix[node]

    def get_node_feature(self, node, key):
        return self.node_features[node].get(key, None)

    def __repr__(self):
        return f"Adjacency Matrix:\n{self.adjacency_matrix}\nNode Features:\n{self.node_features}"

# Example usage