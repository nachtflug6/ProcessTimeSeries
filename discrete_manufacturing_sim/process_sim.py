"""Small process-level simulation primitives.

This module intentionally keeps only the maintained `ProcessSim` helper.
Older prototype graph simulation code has been retired from the active path.
"""

from discrete_manufacturing_sim.multi_linear_petri_net import MultiLinearPetriNet


class ProcessSim:
    """Track elapsed time for a named process connected to a Petri net."""

    def __init__(self, mlin_pn: MultiLinearPetriNet, process_name: str, duration: float):
        self.mlin_pn = mlin_pn
        self.process_name = process_name
        self.duration = duration
        self.time_elapsed = 0.0
        self.is_complete = False

    def simulate(self, time_to_simulate: float):
        """Advance the process by the requested amount of simulated time."""
        if self.is_complete:
            print(f"{self.process_name} has already completed.")
            return

        time_remaining = self.duration - self.time_elapsed
        simulated = min(time_to_simulate, time_remaining)
        self.time_elapsed += simulated

        if self.time_elapsed >= self.duration:
            self.time_elapsed = self.duration
            self.is_complete = True
            print(f"{self.process_name} completed.")
        else:
            print(f"{self.process_name} is still running.")


    #         case 2:
    #             if current_features[self.bout_idx] < current_features[self.limbout_idx]:
    #                 current_features[self.bout_idx] += 1
    #                 current_features[self.prod_token] = 0

    #                 num_supplies, supply_vec = self.move_supplies(node_index)

    #                 if num_supplies < 1:
    #                     # Switch to starved
    #                     current_features[self.state_idx] = 1
    #                 else:
    #                     # Start next part
    #                     current_features[self.state_idx] = 0
    #                     current_features[self.tc_idx] = max(
    #                         self.distributions[node_index][event_type_index].sample(), 1
    #                     )
    #                     current_features[self.prod_token] = 1
    #                     self.features[:, self.bout_idx] -= supply_vec

    #         case 3:
    #             if current_features[self.bout_idx] < current_features[self.limbout_idx]:
    #                 current_features[self.bout_idx] += 1
    #                 current_features[self.prod_token] = 0

    #                 num_supplies, supply_vec = self.move_supplies(node_index)

    #                 if num_supplies < 1:
    #                     # Switch to starved
    #                     current_features[self.state_idx] = 1

    #                 else:
    #                     # Start next part
    #                     current_features[self.state_idx] = 0
    #                     current_features[self.tc_idx] = max(
    #                         self.distributions[node_index][event_type_index].sample(), 1
    #                     )
    #                     current_features[self.prod_token] = 1
    #                     self.features[:, self.bout_idx] -= supply_vec
    #             else:
    #                 # Switch to blocked
    #                 current_features[self.state_idx] = 2

    #     # if current_features != features[node_index]:
    #     #     self.add_to_log(node_index)

    #     features[node_index] = current_features

    #     self.features = features
