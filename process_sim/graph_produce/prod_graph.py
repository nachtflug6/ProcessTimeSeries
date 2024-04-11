
import torch as th
import pandas as pd


class ProdGraph:
    def __init__(self, adjacency, distributions, buffer_limits, batch_size=100):
        
        """
        This class represents a discrete production graph.

        Feature tensor rows:
        0: state,
        1: cycle time
        2: failure time
        3: fix time
        4: production token
        5: limit buffer out
        6: buffer out
        """
        
        self.state_idx = 0
        self.tc_idx = 1
        self.tf_idx = 2
        self.tfix_idx = 3
        self.prod_token = 4
        self.limbout_idx = 5
        self.bout_idx = 6
        
        n_nodes = adjacency.shape[0]
        self.n_nodes = n_nodes
        self.adjacency = th.from_numpy(adjacency)
        self.distributions = distributions
        
        self.features = th.zeros(n_nodes, 7)
        self.features[:, self.limbout_idx] = buffer_limits
                
        self.lapsed_time = 0
        
        self.batch_size = batch_size
        self.batch_counter = 0
        self.output_data = th.zeros(batch_size, 8)
        self.log = th.empty(0, 8)
        
        
    def add_to_log(self, node_index):
        
        if self.batch_counter == self.batch_size:
            self.log = th.cat((self.log, self.output_data), dim=0)
            self.output_data = th.zeros(self.batch_size, 8)
            self.batch_counter = 0
        else:
            self.output_data[self.batch_counter, 0] = self.lapsed_time
            self.output_data[self.batch_counter, 1:] = self.features[node_index]
            self.batch_counter += 1
            
    def move_supplies(self, node_index):
        supply_vec = self.adjacency[:, node_index]
        
        num_supply_sets = float('inf')
        
        if th.sum(supply_vec) > 0:
            num_supply_sets = th.min(th.nan_to_num(self.features[:, 6] / supply_vec, nan=float('inf')))

        return num_supply_sets, supply_vec
        
    def get_entry(self):
        
        features = self.features
        
        # Define entry point
        remaining_time = th.zeros(self.n_nodes, 3)
        remaining_time[:, 0] = th.where((features[:, 0] == 0) & (features[:, 1] >= 0), features[:, 1], float('inf'))
        remaining_time[:, 1] = th.where((features[:, 0] == 0) & (features[:, 2] >= 0), features[:, 2], float('inf'))
        remaining_time[:, 2] = th.where((features[:, 0] == 3) & (features[:, 3] >= 0), features[:, 3], float('inf'))
        
        min_value, flat_index = th.min(remaining_time.view(-1), dim=0)
        
        # Convert flat index to row-column indices
        entry_index, type_index = divmod(flat_index.item(), remaining_time.size(1))
        
        return min_value, entry_index, type_index
    
    def forward(self):
        
        
        
        features = self.features
        
        #print(features[:, 1:4])
        
        min_value, entry_index, event_type_index = self.get_entry()
        
        if min_value == th.inf:
            entry_index = th.randint(low=0, high=self.n_nodes, size=(1,)).item()
        else:
            lapsed_time = min_value.item()
            features[:, 1] = th.where(features[:, 0] == 0, features[:, 1] - lapsed_time, features[:, 1])
            features[:, 2] = th.where(features[:, 0] == 0, features[:, 2] - lapsed_time, features[:, 2])
            features[:, 3] = th.where(features[:, 0] == 3, features[:, 3] - lapsed_time, features[:, 3])
            self.lapsed_time += lapsed_time
        
        node_index = entry_index
        
        current_features = features[node_index]
        
        
        self.add_to_log(node_index)
        
        match current_features[self.state_idx]:
            case 0:
                match event_type_index:
                    case 0:
                        if current_features[self.bout_idx] < current_features[self.limbout_idx]:
                            current_features[self.bout_idx] += 1
                            current_features[self.prod_token] = 0
                            
                            num_supplies, supply_vec = self.move_supplies(node_index)
                        
                            if num_supplies < 1:
                                # Switch to starved
                                current_features[self.state_idx] = 1
                            else:
                                # Start next part
                                current_features[self.tc_idx] = max(self.distributions[node_index][event_type_index].sample(), 1)
                                current_features[self.prod_token] = 1
                        else:
                            # Switch to blocked
                            current_features[self.state_idx] = 2
                            
                    case 1:
                        # Switch to failed
                        current_features[self.state_idx] = 3

            case 1:
                num_supplies, supply_vec = self.move_supplies(node_index)
                
                if num_supplies >= 1:
                    if current_features[self.bout_idx] < current_features[self.limbout_idx]:
                        current_features[self.tc_idx] = max(self.distributions[node_index][event_type_index].sample(), 1)
                        current_features[self.prod_token] = 1
                        self.features[:, self.bout_idx] -= supply_vec
                    else:
                        self.features[node_index, self.state_idx] = 1
                
            case 2:
                if current_features[self.bout_idx] < current_features[self.limbout_idx]:
                    current_features[self.bout_idx] += 1
                    current_features[self.prod_token] = 0
                    
                    num_supplies, supply_vec = self.move_supplies(node_index)
                    
                    if num_supplies < 1:
                        # Switch to starved
                        current_features[self.state_idx] = 1
                    else:
                        # Start next part
                        current_features[self.state_idx] = 0
                        current_features[self.tc_idx] = max(self.distributions[node_index][event_type_index].sample(), 1)
                        current_features[self.prod_token] = 1
                        self.features[:, self.bout_idx] -= supply_vec
            
            case 3:
                if current_features[self.bout_idx] < current_features[self.limbout_idx]:
                    current_features[self.bout_idx] += 1
                    current_features[self.prod_token] = 0
                    
                    num_supplies, supply_vec = self.move_supplies(node_index)
                
                    if num_supplies < 1:
                        # Switch to starved
                        current_features[self.state_idx] = 1
                    
                    else:
                        # Start next part
                        current_features[self.state_idx] = 0
                        current_features[self.tc_idx] = max(self.distributions[node_index][event_type_index].sample(), 1)
                        current_features[self.prod_token] = 1
                        self.features[:, self.bout_idx] -= supply_vec
                else:
                    # Switch to blocked
                    current_features[self.state_idx] = 2
                    
        # if current_features != features[node_index]:
        #     self.add_to_log(node_index)
            
        features[node_index] = current_features
        
        self.features = features
                    