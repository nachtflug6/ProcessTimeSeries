import torch
import pandas as pd



# class ProdGraph:
#     def __init__(
#         self,
#         adjacency_mtx,
#         distribution_mtx,
#         capacity_mtx,
#         transition_fsm,
#         batch_size=100,
#         device="cpu",
#     ):

#         self.adjacency_mtx = adjacency_mtx
#         n_nodes = adjacency_mtx.shape[0]
#         self.n_nodes = n_nodes

#         self.epsilon = 1e-6

#         self.node_state_mtx = torch.zeros(n_nodes, 5)

#         self.state_transition_mtx = torch.tensor(
#             [[0, 1, 0, 0], [1, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float
#         )

#         self.state_event_mtx = torch.tensor(
#             [[0, 0, 0, 1, 0], [1, 1, 0, 0, 1], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]], dtype=torch.float
#         )


#         self.markings_mtx = torch.zeros(n_nodes, 2)
#         self.capacity_mtx = capacity_mtx
#         self.weight_mtx = torch.zeros(n_nodes, 4)
#         self.node_state_mtx = torch.zeros(n_nodes, 4)
#         self.node_state_mtx[:, 0] = 1
#         self.transition_fire_mtx = torch.zeros(n_nodes, 2)
#         self.distribution_mtx = distribution_mtx
#         self.transition_fsm = transition_fsm
#         self.event_matrix = torch

#         self.steady_event_available_mtx = torch.zeros(n_nodes, 2)
#         self.timed_event_available_mtx = torch.zeros(n_nodes, 3)
#         self.event_available_mtx = torch.zeros(n_nodes, 5)

#         self.event_required_mtx = torch.zeros(n_nodes, 5)
#         self.steady_event_required_mtx = torch.zeros(n_nodes, 2)
#         self.timed_event_required_mtx = torch.zeros(n_nodes, 3)

#         self.steady_event_executable_mtx = torch.zeros(n_nodes, 2)

#         self.nodes_ready = torch.zeros(n_nodes, 1)

    # def update_steady_event_available(self):
    #     M_transposed = self.adjacency_mtx.t()
    #     v_t = self.markings_mtx[:, 1].t()
    #     epsilon = self.epsilon

    #     result = M_transposed / (v_t + epsilon)
    #     final_result = result.t()
    #     column_maxes = final_result.max(dim=0)[0]
    #     binary_results = (column_maxes < 1).int()
    #     self.event_available_mtx[:, -2] = binary_results
    #     self.event_available_mtx[:, -1] = torch.where(self.markings_mtx[:, 1] <= self.capacity_mtx[:, 1], 1, 0)

#     def update_event_required(self):
#         event_required_mtx = torch.matmul(self.node_state_mtx, self.state_event_mtx)
#         self.event_required_mtx = event_required_mtx
#         self.timed_event_required_mtx = event_required_mtx[:,0:3]
#         self.steady_event_required_mtx = event_required_mtx[:,-2:]

#     def update_nodes_ready(self):
#         self.update_steady_event_available()
#         print(self.steady_event_available_mtx)
#         print(self.event_required_mtx)
#         rows_match = (self.steady_event_available_mtx * self.event_required_mtx == self.event_required_mtx).all(dim=1)
#         self.nodes_ready = rows_match.int().unsqueeze(1)

#     def node_state_transition(self, i, event_vector):
#         self.node_state_mtx[i] = self.transition_fsm.transition(self.node_state_mtx[i], event_vector)


#     def forward(self):
#         self.update_event_active()

#         # Check if timed event
#         if torch.sum(self.timed_event_required_mtx) > 0:
#             pass

#         # Execute static events
#         self.update_steady_event_available()

#         self.update_beta()
#         self.update_gamma()
#         self.update_node_event()

    # def add_to_log(self, node_index):

    #     if self.batch_counter == self.batch_size:
    #         self.log = th.cat((self.log, self.output_data), dim=0)
    #         self.output_data = th.zeros(self.batch_size, self.num_output)
    #         self.batch_counter = 0
    #     else:
    #         self.output_data[self.batch_counter, 0] = self.lapsed_time
    #         self.output_data[self.batch_counter, 1:] = self.features[node_index]
    #         self.batch_counter += 1

    # def move_supplies(self, node_index):
    #     supply_vec = self.adjacency[:, node_index]

    #     num_supply_sets = float("inf")

    #     if th.sum(supply_vec) > 0:
    #         num_supply_sets = th.min(
    #             th.nan_to_num(self.features[:, 6] / supply_vec, nan=float("inf"))
    #         )

    #     return num_supply_sets, supply_vec

    # def get_entry(self):

    #     time_features = self.features[:, 1:4]

    #     # Define entry point
    #     remaining_time = th.zeros(self.n_nodes, 3)

    #     remaining_time[:, 0] = th.where(
    #         (time_features[:, 0] > 0), time_features[:, 0], float("inf")
    #     )
    #     remaining_time[:, 0] = th.where(
    #         (time_features[:, 0] > 0), time_features[:, 0], float("inf")
    #     )
    #     remaining_time[:, 0] = th.where(
    #         (time_features[:, 0] > 0), time_features[:, 0], float("inf")
    #     )
    #     # remaining_time[:, 1] = th.where(
    #     #     (features[:, 2] > 0), features[:, 2], float("inf")
    #     # )
    #     # remaining_time[:, 2] = th.where(
    #     #     (features[:, 3] > 0), features[:, 3], float("inf")
    #     # )

    #     min_value, flat_index = th.min(remaining_time.view(-1), dim=0)

    #     # Convert flat index to row-column indices
    #     entry_index, type_index = divmod(flat_index.item(), remaining_time.size(1))

    #     return min_value, entry_index, type_index

    # def forward(self):

    #     features = self.features

    #     min_value, entry_index, event_type_index = self.get_entry()

    #     # print(min_value, entry_index, event_type_index)

    #     if min_value.item() == th.inf:
    #         pass  # entry_index = th.randint(low=0, high=self.n_nodes, size=(1,)).item()
    #     else:
    #         self.lapsed_time += min_value.item()

    #         # lapsed_time = min_value.item()
    #         # features[:, 1] = th.where(
    #         #     features[:, 0] == 0, features[:, 1] - lapsed_time, features[:, 1]
    #         # )
    #         # features[:, 2] = th.where(
    #         #     features[:, 0] == 0, features[:, 2] - lapsed_time, features[:, 2]
    #         # )
    #         # features[:, 3] = th.where(
    #         #     features[:, 0] == 3, features[:, 3] - lapsed_time, features[:, 3]
    #         # )

    #     node_index = entry_index
    #     current_features = features[node_index]

    #     self.add_to_log(node_index)

    #     match current_features[self.state_idx]:
    #         case 0:
    #             match event_type_index:
    #                 case 0:
    #                     if (
    #                         current_features[self.bout_idx]
    #                         < current_features[self.limbout_idx]
    #                     ):
    #                         current_features[self.bout_idx] += 1
    #                         current_features[self.prod_token] = 0

    #                         num_supplies, supply_vec = self.move_supplies(node_index)

    #                         if num_supplies < 1:
    #                             # Switch to starved
    #                             current_features[self.state_idx] = 1
    #                         else:
    #                             # Start next part
    #                             current_features[self.tc_idx] = max(
    #                                 self.distributions[node_index][
    #                                     event_type_index
    #                                 ].sample(),
    #                                 1,
    #                             )
    #                             current_features[self.prod_token] = 1
    #                     else:
    #                         # Switch to blocked
    #                         current_features[self.state_idx] = 2

    #                 case 1:
    #                     # Switch to failed
    #                     current_features[self.state_idx] = 3

    #         case 1:
    #             num_supplies, supply_vec = self.move_supplies(node_index)

    #             if num_supplies >= 1:
    #                 if (
    #                     current_features[self.bout_idx]
    #                     < current_features[self.limbout_idx]
    #                 ):
    #                     current_features[self.tc_idx] = max(
    #                         self.distributions[node_index][event_type_index].sample(), 1
    #                     )
    #                     current_features[self.prod_token] = 1
    #                     self.features[:, self.bout_idx] -= supply_vec
    #                 else:
    #                     self.features[node_index, self.state_idx] = 1

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
