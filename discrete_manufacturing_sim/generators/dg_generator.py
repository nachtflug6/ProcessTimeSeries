import torch


class DGGenerator:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes

    def generate_random_dag(self, density):
        # Initialize adjacency matrix with zeros
        adjacency_matrix = torch.zeros(
            (self.num_nodes, self.num_nodes), dtype=torch.int)

        # Generate a random topological ordering
        topological_ordering = torch.randperm(self.num_nodes)

        # Randomly add edges based on the topological ordering
        num_edges = int(self.num_nodes * (self.num_nodes - 1) * density / 2)
        edge_count = 0
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if edge_count >= num_edges:
                    break
                if topological_ordering[i] < topological_ordering[j]:
                    adjacency_matrix[topological_ordering[i],
                                     topological_ordering[j]] = 1
                    edge_count += 1

        return adjacency_matrix

    def generate_random_dg(self, density):
        # Initialize adjacency matrix with zeros
        adjacency_matrix = torch.zeros(
            (self.num_nodes, self.num_nodes), dtype=torch.int)

        # Ensure all nodes are connected
        for i in range(self.num_nodes - 1):
            target_node = torch.randint(i + 1, self.num_nodes, (1,))
            adjacency_matrix[i, target_node] = 1

        # Randomly add additional edges to achieve desired density
        num_edges = int((self.num_nodes - 1) * density) - (self.num_nodes - 1)
        for _ in range(num_edges):
            source_node = torch.randint(0, self.num_nodes - 1, (1,))
            target_node = torch.randint(source_node + 1, self.num_nodes, (1,))
            adjacency_matrix[source_node, target_node] = 1

        return adjacency_matrix

    def generate_linear_dag(self):
        # Initialize adjacency matrix with zeros
        adjacency_matrix = torch.zeros(
            (self.num_nodes, self.num_nodes), dtype=torch.int)

        # Create linear DAG edges
        for i in range(self.num_nodes - 1):
            adjacency_matrix[i, i + 1] = 1

        return adjacency_matrix
