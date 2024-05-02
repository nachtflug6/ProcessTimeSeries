import torch


class GenericFSM:
    def __init__(self, num_states, num_events):
        self.num_states = num_states
        self.num_events = num_events
        # Tensors to store transitions
        self.from_states = torch.zeros(0, num_states)  # Initially empty
        self.to_states = torch.zeros(0, num_states)  # Initially empty
        self.event_vectors = torch.zeros(0, num_events)  # Initially empty
        self.start_state = 0
        self.temporal_events = torch.zeros(num_events, dtype=torch.bool)
        # Property to store active events
        self.active_events = torch.zeros(
            num_states, num_events, dtype=torch.bool)

    def add_transition(self, from_state, to_state, event_vector):
        """Add a new state transition."""
        # Convert states to one-hot vectors
        from_state_vector = torch.zeros(self.num_states)
        from_state_vector[from_state] = 1
        to_state_vector = torch.zeros(self.num_states)
        to_state_vector[to_state] = 1

        # Append to existing tensors
        self.from_states = torch.cat(
            (self.from_states, from_state_vector.unsqueeze(0)), 0)
        self.to_states = torch.cat(
            (self.to_states, to_state_vector.unsqueeze(0)), 0)
        self.event_vectors = torch.cat(
            (self.event_vectors, event_vector.unsqueeze(0)), 0)

        # Update active events property
        self.update_active_events()

    def get_active_events(self, state_idx=None):
        if state_idx == None:
            return self.active_events
        else:
            return self.active_events[state_idx]

    def transition(self, current_state_vector, event_vector):
        """Compute the next state based on the current state and the event vectors."""
        # Subtract and sum the differences for state and event vectors
        state_diff = torch.sum(
            torch.abs(self.from_states - current_state_vector), dim=1)
        event_diff = torch.sum(
            torch.abs(self.event_vectors - event_vector), dim=1)

        # Find indices where both differences are zero
        valid_indices = (state_diff == 0) & (event_diff == 0)

        # Check if there is any valid transition
        if valid_indices.any():
            # Return the to_state vector for the first valid transition found
            return self.to_states[valid_indices.nonzero(as_tuple=True)[0][0]]

        # If no matching transition, return the current state vector
        return current_state_vector

    def compute_active_events(self, current_state):
        """Get active events for the current state."""
        current_state_vector = torch.zeros(self.num_states)
        current_state_vector[current_state] = 1

        # Find the indices where the current state matches from_states
        matching_indices = torch.all(
            self.from_states == current_state_vector, dim=1)

        # Get the event vectors corresponding to matching indices
        active_event_vectors = self.event_vectors[matching_indices]

        # Sum along the rows to get the active events vector
        active_events = torch.sum(active_event_vectors, dim=0)

        # Convert to boolean vector (1 for active, 0 for inactive)
        active_events_bool = active_events > 0

        return active_events_bool

    def update_active_events(self):
        """Update the active events property."""
        active_events = torch.zeros(
            self.num_states, self.num_events, dtype=torch.bool)
        for i in range(self.num_states):
            active_events[i] = self.compute_active_events(i)
        self.active_events = active_events

    def set_start_state(self, start_state):
        """Set the start state."""
        self.start_state = start_state

    def get_start_state(self):
        """Get the start state."""
        return self.start_state

    def set_temporal_event(self, event_index):
        self.temporal_events[event_index] = True

    def remove_temporal_event(self, event_index):
        self.temporal_events[event_index] = False

    def get_temporal_events(self):
        return self.temporal_events
