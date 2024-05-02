import torch
import random

from discrete_manufacturing_sim.multi_temporal_event_handler import MultiTemporalEventHandler
from discrete_manufacturing_sim.multi_linear_petri_net import MultiLinearPetriNet
from discrete_manufacturing_sim.fsm.production_asset_fsm import ProductionAssetFSM


class SimulationHandler:
    def __init__(self, multi_linear_petri_net: MultiLinearPetriNet, finite_state_machine: ProductionAssetFSM, multi_temp_event_handler: MultiTemporalEventHandler):
        self.mlin_pn = multi_linear_petri_net
        num_nodes = multi_linear_petri_net.num_pns
        self.num_nodes = num_nodes

        num_states = finite_state_machine.num_states
        self.num_states = num_states
        num_events = finite_state_machine.num_events
        self.num_events = num_events

        self.fsm = finite_state_machine
        self.state_matrix = torch.zeros(num_nodes, num_states)
        self.state_matrix[:, finite_state_machine.get_start_state()] = 1

        self.multi_temp_event_handler = multi_temp_event_handler
        self.active_events_by_state = finite_state_machine.active_events
        self.active_events = torch.zeros(
            num_nodes, num_events, dtype=torch.bool)

        self.temporal_events = self.fsm.get_temporal_events()
        self.not_temporal_events = torch.logical_not(
            self.fsm.get_temporal_events())

        self.outputs = torch.zeros(num_nodes, num_events, dtype=torch.bool)

    def update_active_events(self):
       # Convert active_events to float for matrix multiplication
        active_events_float = self.active_events_by_state.float()
        # Use torch.matmul to perform matrix multiplication to update active events
        self.active_events = torch.matmul(
            self.state_matrix, active_events_float.float()).bool()

    def update_available_outputs(self):
        self.mlin_pn.update_active_transitions()
        self.outputs[:, self.not_temporal_events] = self.mlin_pn.active_transitions

    def update(self):
        self.update_active_events()
        self.update_available_outputs()

    def execute_static_event(self, node_idx, event_idx):

        # Update Finite State Machine
        self.state_matrix[node_idx] = self.fsm.transition(
            self.state_matrix[node_idx], torch.eye(self.num_events)[event_idx])

        # Fire Petri Net
        if event_idx.item() == 3:
            self.mlin_pn.fire_transition(node_idx, 0)
            if self.multi_temp_event_handler.event_schedule[node_idx, 0] <= 0:
                self.multi_temp_event_handler.resample_element(node_idx, 0)
            if self.multi_temp_event_handler.event_schedule[node_idx, 1] <= 0:
                self.multi_temp_event_handler.resample_element(node_idx, 1)
        elif event_idx.item() == 4:
            self.mlin_pn.fire_transition(node_idx, 1)

        self.update()

    def execute_dynamic_event(self, node_idx, event_idx):
        self.state_matrix[node_idx] = self.fsm.transition(
            self.state_matrix[node_idx], torch.eye(self.num_events)[event_idx])

        # Resample Events
        if event_idx.item() == 1:
            if self.multi_temp_event_handler.event_schedule[node_idx, 2] <= 0:
                self.multi_temp_event_handler.resample_element(node_idx, 2)
        elif event_idx.item() == 2:
            if self.multi_temp_event_handler.event_schedule[node_idx, 0] <= 0:
                self.multi_temp_event_handler.resample_element(node_idx, 0)
            if self.multi_temp_event_handler.event_schedule[node_idx, 1] <= 0:
                self.multi_temp_event_handler.resample_element(node_idx, 1)

        self.update()

    def simulate(self):
        self.update()

        active_temporal_events = self.active_events & self.temporal_events

        if torch.any(active_temporal_events):
            min_i, min_j = self.multi_temp_event_handler.find_min_element(
                active_temporal_events[:, self.temporal_events])
            self.execute_dynamic_event(min_i, min_j)

        # Keep executing random static events as long as available
        choices = self.active_events & self.outputs

        while (torch.any(choices)):
            true_indices = torch.nonzero(choices, as_tuple=False)
            node_idx, event_idx = random.choice(true_indices)
            self.execute_static_event(node_idx, event_idx)
            choices = self.active_events & self.outputs
