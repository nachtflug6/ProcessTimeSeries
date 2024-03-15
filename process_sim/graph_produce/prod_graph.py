
import torch as th
import networkx as nx
import pandas as pd


class ProdGraph:
    def __init__(self, adjacency, distributions, buffer_limits, batch_size=100):
        
        n_nodes = adjacency.shape[0]
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.adjacency = th.from_numpy(adjacency)
        self.distributions = distributions
        
        G = nx.DiGraph(adjacency)
        topological_order = th.tensor(list(nx.topological_sort(G)))
        self.topological_order = topological_order
        
        
        
        self.features = th.zeros(n_nodes, 8)
        
        for i in range(self.n_nodes):
            if th.sum(self.adjacency[:, i]) == 0:
                self.features[i, 6] = th.inf
                
        self.lapsed_time = 0
        
        self.batch_counter = 0
        self.output_data = th.zeros(batch_size, 5)
        self.log = None
        
        
    def add_to_log(self, data):
        
        if self.batch_counter == self.batch_size:
            self.log = pd.concat([self.log, pd.DataFrame({'time': self.output_data[:, 0].cpu().detach(), 
                                                           'node': self.output_data[:, 1].cpu().detach(), 
                                                           'input_buffer': self.output_data[:, 2].cpu().detach(), 
                                                           'output_buffer': self.output_data[:, 3].cpu().detach(), 
                                                           'state': self.output_data[:, 4].cpu().detach()})])
            self.output_data = th.zeros((self.batch_size, 5))
            self.batch_counter = 0
        else:
            self.output_data[self.batch_counter] = th.tensor(data)
            self.batch_counter += 1
            
    def move_supplies(self, node_index):
        supply_vec = self.adjacency[:, node_index]
        
        if th.sum(supply_vec) > 0:
            num_supply_sets = th.min(th.nan_to_num(self.output_buffer / supply_vec, nan=float('inf')))
            if num_supply_sets >= 1 and self.input_buffer[node_index] == 0:
                self.output_buffer -= supply_vec
                self.input_buffer[node_index] = 1
        
    def get_entry(self):
        
        features = self.features
        
        # Define entry point
        remaining_time = th.zeros(self.n_nodes, 3)
        remaining_time[:, 0] = th.where((features[:, 0] == 0) & (features[:, 1] >= 0), features[:, 1], float('inf'))
        remaining_time[:, 1] = th.where((features[:, 0] == 0) & (features[:, 1] >= 0), features[:, 1], float('inf'))
        remaining_time[:, 2] = th.where((features[:, 0] == 0) & (features[:, 1] >= 0), features[:, 1], float('inf'))
        
        min_value, flat_index = th.min(remaining_time.view(-1), dim=0)
        
        # Convert flat index to row-column indices
        entry_index, type_index = divmod(flat_index.item(), remaining_time.size(1))
        
        return min_value, entry_index, type_index
    
    def forward(self):
        
        features = self.features
        min_value, entry_index, type_index = self.get_entry()
        
        if min_value == th.inf:
            entry_index = th.randint(0, self.n_nodes)
        else:
            lapsed_time = min_value.item()
            features[:, 1] = th.where(features[:, 0] == 0, features[:, 1] - lapsed_time, features[:, 1])
            features[:, 2] = th.where(features[:, 0] == 0, features[:, 2] - lapsed_time, features[:, 2])
            features[:, 3] = th.where(features[:, 0] == 3, features[:, 3] - lapsed_time, features[:, 3])
            self.lapsed_time += lapsed_time
        
        node_index = entry_index
        
        match features[node_index, 0]:
            case 0:
                if min_value == 0:
                    if type_index == 0:
                        # Check if the part is finished
                        if self.output_buffer[node_index] < self.buffer_limits[node_index]:
                            self.output_buffer[node_index] += 1
                            
                            if self.input_buffer[node_index] > 0:
                                # Start next part
                                self.remaining_time[node_index, 0] = max(self.distributions[node_index][type_index].sample(), 1)
                                self.input_buffer[node_index] -= 1
                                self.add_to_log([self.lapsed_time, node_index.item(), self.input_buffer[node_index].item(), self.output_buffer[node_index].item(), 0])
                            else:
                                # Switch to starved
                                self.states[node_index, 0] = 0
                                self.states[node_index, 1] = 1
                                self.add_to_log([self.lapsed_time, node_index.item(), self.input_buffer[node_index].item(), self.output_buffer[node_index].item(), 1])
                        
                        else:
                            # Switch to blocked
                            self.states[node_index, 0] = 0
                            self.states[node_index, 2] = 1
                            self.add_to_log([self.lapsed_time, node_index.item(), self.input_buffer[node_index].item(), self.output_buffer[node_index].item(), 2])
                    elif type_index == 1:
                        # Switch to failed
                        self.states[node_index, 0] = 0
                        self.states[node_index, 3] = 1
                        self.remaining_time[node_index, 2] = max(self.distributions[node_index][type_index].sample(), 1)
                        self.add_to_log([self.lapsed_time, node_index.item(), self.input_buffer[node_index].item(), self.output_buffer[node_index].item(), 3])
                    
            case 1:
                if self.input_buffer[node_index] > 0:
                    # Start producing
                    self.input_buffer[node_index] -= 1
                    self.remaining_time[node_index, 0] = max(self.distributions[node_index][type_index].sample(), 1)
                    
                    # Switch to producing state
                    self.states[node_index, 0] = 1
                    self.states[node_index, 1] = 0
                    self.add_to_log([self.lapsed_time, node_index.item(), self.input_buffer[node_index].item(), self.output_buffer[node_index].item(), 0])
                
            case 2:
                if self.output_buffer[node_index] < self.buffer_limits[node_index]:
                    self.output_buffer[node_index] += 1
                    
                    if self.input_buffer[node_index] > 0:
                        # Start next part
                        self.remaining_time[node_index, 0] = max(self.distributions[node_index][type_index].sample(), 1)
                        self.input_buffer[node_index] -= 1
                        
                        # Switch to producing state
                        self.states[node_index, 0] = 1
                        self.states[node_index, 2] = 0
                        self.add_to_log([self.lapsed_time, node_index.item(), self.input_buffer[node_index].item(), self.output_buffer[node_index].item(), 0])
                    else:
                        # Switch to starved
                        self.states[node_index, 1] = 1
                        self.states[node_index, 2] = 0
                        self.add_to_log([self.lapsed_time, node_index.item(), self.input_buffer[node_index].item(), self.output_buffer[node_index].item(), 1])
            
            case 3:
                features[node_index, 0] = 0
                # self.states[node_index, 0] = 1
                # self.states[node_index, 3] = 0
                # self.remaining_time[node_index, 0] = max(self.distributions[node_index][type_index].sample(), 1)
                # self.remaining_time[node_index, 1] = max(self.distributions[node_index][type_index].sample(), 1)
                # self.add_to_log([self.lapsed_time, node_index.item(), self.input_buffer[node_index].item(), self.output_buffer[node_index].item(), 0])
        
        self.features = features