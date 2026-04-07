import numpy as np


class DistributionDataGenerator:
    def __init__(
        self,
        means_matrix,
        distribution_types=["uniform", "normal"],
        uniform_params={"spread": 0.1},
        normal_params={"std_dev": 0.1},
        exponential_params={"lambda": 1},
    ):
        self.means_matrix = means_matrix
        self.num_instances, self.num_distributions = means_matrix.shape
        self.distribution_types = distribution_types
        self.uniform_params = uniform_params
        self.normal_params = normal_params
        self.exponential_params = exponential_params

    def generate(self):
        distributions_data = []
        for i in range(self.num_instances):
            instance_data = []
            for j in range(self.num_distributions):
                dist_type = np.random.choice(self.distribution_types)
                mean = self.means_matrix[i, j]

                if dist_type == "uniform":
                    spread = self.uniform_params["spread"]
                    params = {
                        "low": mean - spread / 2,
                        "high": mean + spread / 2,
                    }
                elif dist_type == "normal":
                    std_dev = self.normal_params["std_dev"]
                    params = {"mean": mean, "std_dev": std_dev}
                elif dist_type == "exponential":
                    params = {
                        "lambda": self.exponential_params.get("lambda", 1 / max(mean, 1e-6))
                    }
                else:
                    raise ValueError(f"Unsupported distribution type: {dist_type}")

                instance_data.append({"type": dist_type, "params": params})
            distributions_data.append(instance_data)
        return distributions_data
