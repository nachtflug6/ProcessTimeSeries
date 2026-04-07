import torch


class DGGenerator:
    """Generate adjacency matrices for simple directed-graph experiments."""

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    @staticmethod
    def _to_output(adjacency_matrix):
        """Return a NumPy-friendly matrix for tests and notebooks."""
        return adjacency_matrix.cpu().numpy()

    def generate_random_dag(self, density):
        adjacency_matrix = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.int)

        topological_ordering = torch.randperm(self.num_nodes)
        num_edges = int(self.num_nodes * (self.num_nodes - 1) * density / 2)
        edge_count = 0

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if edge_count >= num_edges:
                    break
                if topological_ordering[i] < topological_ordering[j]:
                    adjacency_matrix[topological_ordering[i], topological_ordering[j]] = 1
                    edge_count += 1

        return self._to_output(adjacency_matrix)

    def generate_random_dg(self, density):
        adjacency_matrix = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.int)

        for i in range(self.num_nodes - 1):
            target_node = torch.randint(i + 1, self.num_nodes, (1,))
            adjacency_matrix[i, target_node] = 1

        num_edges = max(int((self.num_nodes - 1) * density) - (self.num_nodes - 1), 0)
        for _ in range(num_edges):
            source_node = torch.randint(0, self.num_nodes - 1, (1,))
            target_node = torch.randint(source_node + 1, self.num_nodes, (1,))
            adjacency_matrix[source_node, target_node] = 1

        return self._to_output(adjacency_matrix)

    def generate_linear_dag(self):
        adjacency_matrix = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.int)

        for i in range(self.num_nodes - 1):
            adjacency_matrix[i, i + 1] = 1

        return self._to_output(adjacency_matrix)
