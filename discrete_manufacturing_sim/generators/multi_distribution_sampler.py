import torch


class MultiDistributionSampler:
    def __init__(
        self,
        gen_or_num_instances,
        num_distributions=None,
        distributions_data=None,
        device="cpu",
    ):
        """Create a sampler from either a generator object or explicit inputs.

        Supported forms:
        - `MultiDistributionSampler(generator, device="cpu")`
        - `MultiDistributionSampler(num_instances, num_distributions, distributions_data, device="cpu")`
        """
        self.device = device

        if distributions_data is None:
            gen = gen_or_num_instances
            self.num_instances = gen.num_instances
            self.num_distributions = gen.num_distributions
            distributions_data = gen.generate()
        else:
            self.num_instances = gen_or_num_instances
            self.num_distributions = num_distributions

        self.distributions = self.generate_distributions(distributions_data)

    def create_sampler(self):
        """Compatibility helper for older test and notebook code."""
        return self

    def generate_distributions(self, distributions_data):
        distributions = []
        for i in range(self.num_instances):
            dists = []
            for j in range(self.num_distributions):
                dist_info = distributions_data[i][j]
                dist_type = dist_info["type"]
                params = dist_info["params"]
                if dist_type == "uniform":
                    dist = torch.distributions.Uniform(
                        params["low"], params["high"]
                    )
                elif dist_type == "normal":
                    dist = torch.distributions.Normal(
                        params["mean"], params["std_dev"]
                    )
                elif dist_type == "exponential":
                    dist = torch.distributions.Exponential(1 / params["lambda"])
                else:
                    raise ValueError(f"Unsupported distribution type: {dist_type}")
                dists.append(dist)
            distributions.append(dists)
        return distributions

    def sample(self, i, j):
        sample_value = self.distributions[i][j].sample().to(self.device)
        return torch.clamp_min(sample_value, 0)
