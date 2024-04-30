import torch

class DistributionSampler:
    def __init__(self, distributions, device="cpu"):
        self.distributions = distributions
        self.device = device
        
    def sample(self, i, j):
        sample_value = self.distributions[i][j].sample().to(self.device)
        # Return zero if sampled value is negative
        return torch.clamp_min(sample_value, 0)

class MultiDistributionSampler:
    def __init__(
        self,
        num_instances,
        num_distributions,
        distributions_data,
        device="cpu"
    ):
        self.num_instances = num_instances
        self.num_distributions = num_distributions
        self.device = device
        self.distributions = self.generate_distributions(distributions_data)

    def generate_distributions(self, distributions_data):
        distributions = []
        for i in range(self.num_instances):
            dists = []
            for j in range(self.num_distributions):
                dist_info = distributions_data[i][j]
                type = dist_info["type"]
                params = dist_info["params"]
                if type == "uniform":
                    dist = torch.distributions.Uniform(params["low"], params["high"])
                elif type == "normal":
                    dist = torch.distributions.Normal(params["mean"], params["std_dev"])
                elif type == "exponential":
                    dist = torch.distributions.Exponential(1 / params["lambda"])
                else:
                    raise ValueError(f"Unsupported distribution type: {type}")
                dists.append(dist)
            distributions.append(dists)
        return distributions

    def create_sampler(self):
        return DistributionSampler(self.distributions, self.device)