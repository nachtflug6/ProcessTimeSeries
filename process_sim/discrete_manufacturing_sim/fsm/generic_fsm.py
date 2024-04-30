import torch
import unittest

class GenericFSM:
    def __init__(self, num_states, num_events):
        self.num_states = num_states
        self.num_events = num_events
        # Tensors to store transitions
        self.from_states = torch.zeros(0, num_states)  # Initially empty
        self.to_states = torch.zeros(0, num_states)  # Initially empty
        self.event_vectors = torch.zeros(0, num_events)  # Initially empty
    
    def add_transition(self, from_state, to_state, event_vector):
        """Add a new state transition."""
        # Convert states to one-hot vectors
        from_state_vector = torch.zeros(self.num_states)
        from_state_vector[from_state] = 1
        to_state_vector = torch.zeros(self.num_states)
        to_state_vector[to_state] = 1
        # Convert event vector to tensor
        # event_vector = torch.tensor(event_vector, dtype=torch.float32)

        # Append to existing tensors
        self.from_states = torch.cat((self.from_states, from_state_vector.unsqueeze(0)), 0)
        self.to_states = torch.cat((self.to_states, to_state_vector.unsqueeze(0)), 0)
        self.event_vectors = torch.cat((self.event_vectors, event_vector.unsqueeze(0)), 0)

    def transition(self, current_state_vector, event_vector):
        """Compute the next state based on the current state and the event vectors."""
        # current_state_vector = torch.tensor(current_state_vector, dtype=torch.float32)
        # event_vector = torch.tensor(event_vector, dtype=torch.float32)

        # Subtract and sum the differences for state and event vectors
        state_diff = torch.sum(torch.abs(self.from_states - current_state_vector), dim=1)
        event_diff = torch.sum(torch.abs(self.event_vectors - event_vector), dim=1)

        # Find indices where both differences are zero
        valid_indices = (state_diff == 0) & (event_diff == 0)

        # Check if there is any valid transition
        if valid_indices.any():
            # Return the to_state vector for the first valid transition found
            return self.to_states[valid_indices.nonzero(as_tuple=True)[0][0]]

        # If no matching transition, return the current state vector
        return current_state_vector
