from graph_produce.fsm.generic_fsm import GenericFSM

class FSMProductionAsset(GenericFSM):
    def __init__(self):
        # Suppose we define 5 states and 5 events for this production asset FSM
        super().__init__(num_states=5, num_events=5)
        
        # Define an intermediate state for handling "a and e" vs. "a and not e"
        self.qint = 4  # An arbitrary index for the intermediate state
        
        # Define transitions that were previously described
        self.add_transition(0, 1, 3)  # Transition from state q0 to q1 on event 'd'
        
        # Transitions from state q1 using an intermediate state qint
        self.add_transition(1, self.qint, 0)  # Transition from q1 to qint on event 'a'
        self.add_transition(self.qint, 0, 4)  # Transition from qint to q0 on event 'e'
        self.add_transition(self.qint, 2, 0)  # Transition from qint to q2 on event 'a' (assuming "a and not e")
        
        # Other specified transitions
        self.add_transition(1, 3, 1)  # Transition from q1 to q3 on event 'b'
        self.add_transition(3, 1, 2)  # Transition from q3 to q1 on event 'c'
        self.add_transition(2, 0, 4)  # Transition from q2 to q0 on event 'e'
        
