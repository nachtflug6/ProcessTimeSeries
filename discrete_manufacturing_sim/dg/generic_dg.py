import torch


class GenericDirectedGraph:
    """Simple directed graph wrapper around an adjacency matrix."""

    def __init__(self, adjacency_matrix):
        self.adjacency_matrix = torch.as_tensor(adjacency_matrix, dtype=torch.float32)
        if self.adjacency_matrix.ndim != 2 or self.adjacency_matrix.shape[0] != self.adjacency_matrix.shape[1]:
            raise ValueError("adjacency_matrix must be a square 2D matrix")

        self.num_nodes = self.adjacency_matrix.size(0)
        self.node_features = {}

    def add_edge(self, source_node, destination_node):
        self.adjacency_matrix[source_node, destination_node] = 1

    def remove_edge(self, source_node, destination_node):
        self.adjacency_matrix[source_node, destination_node] = 0

    def add_feature(self, key, values):
        if len(values) != self.num_nodes:
            raise ValueError("Number of values must be equal to the number of nodes")
        self.node_features[key] = torch.as_tensor(values)

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
        values = self.node_features.get(key)
        return None if values is None else values[node]

    def __repr__(self):
        return f"Adjacency Matrix:\n{self.adjacency_matrix}\nNode Features:\n{self.node_features}"

