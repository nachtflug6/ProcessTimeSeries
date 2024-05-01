import unittest
import torch

from discrete_manufacturing_sim.dg.generic_dg import GenericDirectedGraph

class TestDirectedGraph(unittest.TestCase):
    def test_get_previous_nodes(self):
        # Define adjacency matrix
        adjacency_matrix = torch.tensor([
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ], dtype=torch.float32)
        
        # Create graph
        dg = GenericDirectedGraph(adjacency_matrix)

        # Test get_previous_nodes() without indices
        previous_nodes = dg.get_previous_nodes(3)
        expected_previous_nodes = torch.tensor([0, 1, 1, 0, 0], dtype=torch.float32)
        self.assertTrue(torch.equal(previous_nodes, expected_previous_nodes))

        # Test get_previous_nodes() with indices
        previous_nodes_indices = dg.get_previous_nodes(3, indices=True)
        expected_previous_nodes_indices = torch.tensor([1, 2], dtype=torch.long)
        self.assertTrue(torch.equal(previous_nodes_indices, expected_previous_nodes_indices))

    def test_get_next_nodes(self):
        # Define adjacency matrix
        adjacency_matrix = torch.tensor([
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ], dtype=torch.float32)
        
        # Create graph
        dg = GenericDirectedGraph(adjacency_matrix)

        # Test get_next_nodes() without indices
        next_nodes = dg.get_next_nodes(1)
        expected_next_nodes = torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32)
        self.assertTrue(torch.equal(next_nodes, expected_next_nodes))

        # Test get_next_nodes() with indices
        next_nodes_indices = dg.get_next_nodes(1, indices=True)
        expected_next_nodes_indices = torch.tensor(3, dtype=torch.long)
        self.assertTrue(torch.equal(next_nodes_indices, expected_next_nodes_indices))

if __name__ == '__main__':
    unittest.main()