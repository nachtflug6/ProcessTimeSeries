import unittest
import torch
import numpy as np

from discrete_manufacturing_sim.generators.multi_distribution_sampler import MultiDistributionSampler
from discrete_manufacturing_sim.generators.distribution_data_generator import DistributionDataGenerator

class TestMultiDistributionSampler(unittest.TestCase):
    def setUp(self):
        num_instances = 5
        num_distributions = 3
        
        # Generate distributions data using DistributionDataGenerator
        means_matrix = np.random.randn(num_instances, num_distributions)
        generator = DistributionDataGenerator(means_matrix)
        distributions_data = generator.generate()

        # Initialize MultiDistributionSampler
        self.sampler = MultiDistributionSampler(
            num_instances, num_distributions, distributions_data, device=torch.device("cpu")
        )
        self.distribution_sampler = self.sampler.create_sampler()

    def test_sample(self):
        sample = self.distribution_sampler.sample(0, 0)
        self.assertIsInstance(sample, torch.Tensor)

if __name__ == '__main__':
    unittest.main()