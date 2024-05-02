import torch


class MultiDistributionSampler:
    def __init__(
        self,
        gen,
        device="cpu"
    ):
        self.num_instances = gen.num_instances
        self.num_distributions = gen.num_distributions
        self.device = device
        self.distributions = self.generate_distributions(gen.generate())

    def generate_distributions(self, distributions_data):
        distributions = []
        for i in range(self.num_instances):
            dists = []
            for j in range(self.num_distributions):
                dist_info = distributions_data[i][j]
                type = dist_info["type"]
                params = dist_info["params"]
                if type == "uniform":
                    dist = torch.distributions.Uniform(
                        params["low"], params["high"])
                elif type == "normal":
                    dist = torch.distributions.Normal(
                        params["mean"], params["std_dev"])
                elif type == "exponential":
                    dist = torch.distributions.Exponential(
                        1 / params["lambda"])
                else:
                    raise ValueError(f"Unsupported distribution type: {type}")
                dists.append(dist)
            distributions.append(dists)
        return distributions

    def sample(self, i, j):
        sample_value = self.distributions[i][j].sample().to(self.device)
        # Return zero if sampled value is negative
        return torch.clamp_min(sample_value, 0)
