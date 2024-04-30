import torch
import torch.nn.functional as F
import unittest

from discrete_manufacturing_sim.fsm.production_asset_fsm import ProductionAssetFSM

class TestProductionAssetFSM(unittest.TestCase):
    def setUp(self):
        self.fsm = ProductionAssetFSM()

    def test_transitions(self):
        # Define expected state transitions
        expected_transitions = {
            (0, torch.tensor([0, 0, 0, 1, 0])): 1,
            (0, torch.tensor([1, 0, 0, 0, 0])): 0,
            (1, torch.tensor([1, 0, 0, 0, 0])): 2,
            (1, torch.tensor([0, 1, 0, 0, 0])): 3,
            (2, torch.tensor([0, 0, 0, 0, 1])): 0,
            (2, torch.tensor([0, 0, 1, 0, 0])): 2,
            (3, torch.tensor([0, 0, 0, 1, 0])): 3,
            (3, torch.tensor([0, 0, 1, 0, 0])): 1,
        }

        for (state_index, event_vector), expected_state_index in expected_transitions.items():
            # Convert state_index to one-hot encoded vector
            current_state_vector = torch.zeros(self.fsm.num_states)
            current_state_vector[state_index] = 1

            # Perform transition
            result_vector = self.fsm.transition(current_state_vector, event_vector)

            # Convert expected_state_index to one-hot encoded vector
            expected_state_vector = torch.zeros(self.fsm.num_states)
            expected_state_vector[expected_state_index] = 1

            self.assertTrue(torch.equal(expected_state_vector, result_vector),
                            f"Failed for state {current_state_vector} with event {event_vector}")
