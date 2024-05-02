import unittest
import numpy as np
from discrete_manufacturing_sim.generators.distribution_data_generator import DistributionDataGenerator

class TestDistributionDataGenerator(unittest.TestCase):
    def setUp(self):
        # Set up the means matrix for testing
        self.num_instances = 5
        self.num_distributions = 3
        self.means_matrix = np.random.randn(self.num_instances, self.num_distributions)

    def test_generate(self):
        # Test generate method
        generator = DistributionDataGenerator(self.means_matrix, uniform_params={"spread": 2})
        distributions_data = generator.generate()

        # Check dimensions of distributions data
        self.assertEqual(len(distributions_data), self.num_instances)
        for instance_data in distributions_data:
            self.assertEqual(len(instance_data), self.num_distributions)

        # Check that all distributions are either "uniform", "normal", or "exponential"
        for instance_data in distributions_data:
            for dist_data in instance_data:
                self.assertIn(dist_data["type"], ["uniform", "normal", "exponential"])

        # Check uniform parameters
        for i in range(self.num_instances):
            for j in range(self.num_distributions):
                if distributions_data[i][j]["type"] == "uniform":
                    spread = generator.uniform_params["spread"]
                    mean = self.means_matrix[i, j]
                    low_expected = mean - spread / 2
                    high_expected = mean + spread / 2
                    self.assertAlmostEqual(distributions_data[i][j]["params"]["low"], low_expected, places=5)
                    self.assertAlmostEqual(distributions_data[i][j]["params"]["high"], high_expected, places=5)

if __name__ == "__main__":
    unittest.main()
