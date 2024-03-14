
import torch as th
import torch.distributions as dist
import networkx as nx

import pandas as pd

import matplotlib.pyplot as plt

class ProdGraph:
    def __init__(self, adjacency, distributions, initialization, buffer_limits, batch_size=100):
        self.adjacency = th.tensor(adjacency)
        self.distributions = distributions
        
        G = nx.DiGraph(adjacency)
        topological_order = th.tensor(list(nx.topological_sort(G)))
        self.topological_order = topological_order
        
        self.remaining_time = th.zeros(adjacency.shape[0])
        self.states = th.zeros((adjacency.shape[0], adjacency.shape[0]))
        self.output_buffer = th.zeros(adjacency.shape[0])
        self.input_buffer = th.zeros(adjacency.shape[0])
        self.buffer_limits = th.tensor(buffer_limits)
        self.n_nodes = adjacency.shape[0]
        self.batch_size = batch_size
                  
        
        
        # Initialize
        self.states = th.tensor(initialization)
        
        for i in range(self.n_nodes):
            self.remaining_time[i] = distributions[i].sample()
            if th.sum(self.adjacency[:, i]) == 0:
                self.input_buffer[i] = th.inf
                
        self.lapsed_time = 0
        
        self.log_entries = []
        
        self.output_data = th.zeros((self.batch_size, 5))
        self.batch_counter = 0
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
            
        
    def forward(self):
        
        # Define entry point
        remaining_time = th.where(self.states[:, 0] * self.remaining_time > 0, self.remaining_time, float('inf'))
        min_value, entry_index = th.min(remaining_time, dim=0)
        
        # Adjust times
        lapsed_time = min_value.item()
        self.remaining_time -= lapsed_time * self.states[:, 0]
        self.lapsed_time += lapsed_time
        
        # Get current topological order
        topological_index_entry_node = th.where(self.topological_order == entry_index)[0].item()
        current_topological_order = [self.topological_order[(topological_index_entry_node + i) % self.n_nodes] for i in range(self.n_nodes)]
        
        # Iterate through graph
        for node_index in current_topological_order:
            # Move supplies from output to input
            
            supply_vec = self.adjacency[:, node_index]
            if th.sum(supply_vec) > 0:
                num_supply_sets = th.min(th.nan_to_num(self.output_buffer / supply_vec, nan=float('inf')))
                if num_supply_sets >= 1 and self.input_buffer[node_index] == 0:
                    self.output_buffer -= supply_vec
                    self.input_buffer[node_index] = 1
                
            # Case producing
            if self.states[node_index, 0] == 1:
                # Check if this node 
                if self.remaining_time[node_index] == 0:
                    # Check if the part is finished
                    if self.output_buffer[node_index] < self.buffer_limits[node_index]:
                        self.output_buffer[node_index] += 1
                        
                        if self.input_buffer[node_index] > 0:
                            # Start next part
                            self.remaining_time[node_index] = self.distributions[node_index].sample()
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
                        
            # Case Starved 
            elif self.states[node_index, 1] == 1:
                # Check if supplies are now here
                if self.input_buffer[node_index] > 0:
                    # Start producing
                    self.input_buffer[node_index] -= 1
                    self.remaining_time[node_index] = self.distributions[node_index].sample()
                    
                    # Switch to producing state
                    self.states[node_index, 0] = 1
                    self.states[node_index, 1] = 0
                    self.add_to_log([self.lapsed_time, node_index.item(), self.input_buffer[node_index].item(), self.output_buffer[node_index].item(), 0])
                
            # Case Blocked 
            elif self.states[node_index, 2] == 1:
                # Check if space in buffer
                if self.output_buffer[node_index] < self.buffer_limits[node_index]:
                    self.output_buffer[node_index] += 1
                    
                    if self.input_buffer[node_index] > 0:
                        # Start next part
                        self.remaining_time[node_index] = self.distributions[node_index].sample()
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