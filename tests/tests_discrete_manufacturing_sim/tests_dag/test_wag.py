import unittest
import torch

from discrete_manufacturing_sim.dg.wdg import WeightedDirectedGraph

class TestWeightedDirectedGraph(unittest.TestCase):
    def test_add_feature(self):
        # Define adjacency matrix and states
        adjacency_matrix = torch.tensor([
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ], dtype=torch.float32)
        
        # Create graph
        pdg = WeightedDirectedGraph(adjacency_matrix)
        
        # Test add_feature()
        pdg.add_feature('new_feature', torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32))
        self.assertIn('new_feature', pdg.node_features)
        self.assertTrue(torch.equal(pdg.node_features['new_feature'], torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32)))

    def test_add_edge(self):
        # Define adjacency matrix and states
        adjacency_matrix = torch.tensor([
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]
        ], dtype=torch.float32)
        
        # Create graph
        pdg = WeightedDirectedGraph(adjacency_matrix)
        
        # Test add_edge()
        pdg.add_edge(0, 4, 2)  # Add edge from node 0 to node 4 with weight 2
        self.assertEqual(pdg.adjacency_matrix[0, 4], 2)

if __name__ == '__main__':
    unittest.main()