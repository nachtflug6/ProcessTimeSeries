from discrete_manufacturing_sim.dg.generic_dg import GenericDirectedGraph


class WeightedDirectedGraph(GenericDirectedGraph):
    def __init__(self, adjacency_matrix):
        super().__init__(adjacency_matrix)

    def add_edge(self, source_node, destination_node, weight=1):  # Override the add_edge method
        self.adjacency_matrix[source_node, destination_node] = weight
