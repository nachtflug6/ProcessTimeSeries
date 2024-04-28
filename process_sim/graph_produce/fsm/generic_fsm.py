import torch

class GenericFSM:
    def __init__(self, num_states, num_events):
        self.num_states = num_states
        self.num_events = num_events
        # Initialize the transition matrix
        self.transition_tensor = torch.zeros((num_states, num_states, num_events), dtype=torch.float32)
        # Valid transitions matrix: Tracks whether a state-event combination is valid
        self.valid_transitions = torch.zeros((num_states, num_events), dtype=torch.bool)
        
    def add_transition(self, from_state, to_state, event):
        """Set a specific state transition based on an event."""
        # Set the transition in the tensor
        self.transition_tensor[from_state, to_state, event] = 1
        # Mark this state-event combination as valid
        self.valid_transitions[from_state, event] = True

    def transition(self, current_state_vector, event_vector):
        """Compute the next state based on the current state and event vectors."""

        # Ensure current_state_vector and event_vector are torch tensors of type float
        current_state_vector = torch.tensor(current_state_vector, dtype=torch.float32)
        event_vector = torch.tensor(event_vector, dtype=torch.float32)

        # Validate vectors
        if len(current_state_vector) != self.num_states:
            raise ValueError("Current state vector must have length equal to the number of states")
        if len(event_vector) != self.num_events:
            raise ValueError("Event vector must have length equal to the number of events")

        # Determine the current state index from the current state vector
        current_state_index = torch.argmax(current_state_vector).item()

        # Check if the transition is valid for the current state
        if not torch.any(self.valid_transitions[current_state_index] & event_vector.bool()):
            print("Invalid event for the current state. Transition not performed.")
            return current_state_vector  # Return the current state as no transition occurs

        # Compute the next state probabilities
        next_state_probs = torch.matmul(self.transition_tensor[current_state_index], event_vector)
        # Convert the probabilities to a one-hot vector of the next state
        next_state_vector = torch.zeros(self.num_states, dtype=torch.float)
        next_state_vector[torch.argmax(next_state_probs).item()] = 1

        return next_state_vector

# Example usage
fsm = GenericFSM(num_states=5, num_events=5)
fsm.add_transition(0, 1, 3)  # Add valid transition from state 0 to 1 on event 3
current_state_vector = [1, 0, 0, 0, 0]  # Currently in state 0
event_vector = [0, 0, 1, 1, 1]  # Event 3 occurs
next_state_vector = fsm.transition(current_state_vector, event_vector)
print("Next state vector:", next_state_vector)