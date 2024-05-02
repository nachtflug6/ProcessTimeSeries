import torch
import torch.nn.functional as F
import unittest

from discrete_manufacturing_sim.fsm.generic_fsm import GenericFSM

class TestGenericFSM(unittest.TestCase):
    def setUp(self):
        self.num_states = 3
        self.num_events = 2
        self.fsm = GenericFSM(self.num_states, self.num_events)
        self.fsm.add_transition(0, 1, F.one_hot(torch.tensor(0), self.num_events))
        self.fsm.add_transition(1, 2, F.one_hot(torch.tensor(1), self.num_events))

    def test_transitions(self):
        # Testing each possible state and event vector combination
        for state_index in range(self.num_states):
            current_state_vector = torch.tensor([1 if i == state_index else 0 for i in range(self.num_states)])
            for event_index in range(2**self.num_events):  # Binary count to test all event combinations
                event_vector = torch.tensor([(event_index >> i) & 1 for i in range(self.num_events)])
                expected_state_index = state_index  # Assume no transition by default
                if state_index == 0 and torch.equal(event_vector, torch.tensor([1, 0])):
                    expected_state_index = 1
                elif state_index == 1 and torch.equal(event_vector, torch.tensor([0, 1])):
                    expected_state_index = 2
                
                expected_state_vector = torch.tensor([1 if i == expected_state_index else 0 for i in range(self.num_states)])
                result_vector = self.fsm.transition(current_state_vector, event_vector)
                self.assertTrue(torch.equal(expected_state_vector, result_vector),
                                f"Failed for state {current_state_vector} with event {event_vector}")
                
if __name__ == '__main__':
    unittest.main()