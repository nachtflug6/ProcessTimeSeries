import torch
import torch.nn.functional as F

from discrete_manufacturing_sim.fsm.generic_fsm import GenericFSM


class ProductionAssetFSM(GenericFSM):
    def __init__(self):
        # Suppose we define 5 states and 5 events for this production asset FSM
        num_states = 4
        num_events = 5

        super().__init__(num_states=num_states, num_events=num_events)

        # Transitions using one-hot encoded vectors
        self.add_transition(0, 1, F.one_hot(
            torch.tensor(3), num_events).float())
        self.add_transition(1, 2, F.one_hot(
            torch.tensor(0), num_events).float())

        # Other specified transitions
        self.add_transition(1, 3, F.one_hot(
            torch.tensor(1), num_events).float())
        self.add_transition(3, 1, F.one_hot(
            torch.tensor(2), num_events).float())
        self.add_transition(2, 0, F.one_hot(
            torch.tensor(4), num_events).float())
        self.set_start_state(0)
        self.set_temporal_event(0)
        self.set_temporal_event(1)
        self.set_temporal_event(2)
