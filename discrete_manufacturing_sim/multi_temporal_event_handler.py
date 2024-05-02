import torch


class MultiTemporalEventHandler:
    def __init__(self, mds):
        self.mds = mds
        num_instances = mds.num_instances
        num_distributions = mds.num_distributions
        self.num_instances = num_instances
        self.num_distributions = num_distributions
        self.event_schedule = torch.zeros(num_instances, num_distributions)
        self.epsilon = 1e-6
        self.runtime = 0

    def resample_element(self, i, j):
        self.event_schedule[i, j] = self.mds.sample(i, j)

    def find_min_element(self, active_elements):

        event_schedule = self.event_schedule
        event_schedule = torch.where(
            active_elements, event_schedule, 1/self.epsilon)

        flat_tensor = event_schedule.view(-1)

        # Find the minimum value and its index
        min_value, min_index = torch.min(flat_tensor, dim=0)

        # Convert flat index to row and column indices
        num_columns = event_schedule.size(1)
        min_i = min_index // num_columns
        min_j = min_index % num_columns

        event_schedule -= min_value
        self.runtime += min_value

        self.event_schedule = torch.where(
            active_elements, event_schedule, self.event_schedule)

        return min_i, min_j
