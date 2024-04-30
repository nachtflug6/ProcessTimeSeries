import unittest
import numpy as np
from collections import Counter
from discrete_manufacturing_sim.generators.dg_generator import DGGenerator

class TestDGGenerator(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 5
        self.generator = DGGenerator(self.num_nodes)

    def test_generate_random_dag(self):
        density = 0.5
        adjacency_matrix = self.generator.generate_random_dag(density)
        self.assertEqual(adjacency_matrix.shape, (self.num_nodes, self.num_nodes))
        self.assertTrue(np.all(adjacency_matrix >= 0))
        self.assertTrue(np.all(adjacency_matrix <= 1))
        self.assertTrue(np.all(np.diag(adjacency_matrix) == 0))

    def test_generate_random_dg(self):
        density = 0.5
        adjacency_matrix = self.generator.generate_random_dg(density)
        self.assertEqual(adjacency_matrix.shape, (self.num_nodes, self.num_nodes))
        self.assertTrue(np.all(adjacency_matrix >= 0))
        self.assertTrue(np.all(adjacency_matrix <= 1))
        self.assertTrue(np.all(np.diag(adjacency_matrix) == 0))

    def test_generate_linear_dag(self):
        adjacency_matrix = self.generator.generate_linear_dag()
        self.assertEqual(adjacency_matrix.shape, (self.num_nodes, self.num_nodes))
        self.assertTrue(np.all(adjacency_matrix >= 0))
        self.assertTrue(np.all(adjacency_matrix <= 1))
        self.assertTrue(np.all(np.diag(adjacency_matrix) == 0))
        expected_edges = Counter([(i, i + 1) for i in range(self.num_nodes - 1)])
        actual_edges = Counter(zip(*np.where(adjacency_matrix == 1)))
        self.assertEqual(actual_edges, expected_edges)

if __name__ == '__main__':
    unittest.main()