import torch as th
import torch.distributions as dist
import networkx as nx

import pandas as pd
import matplotlib.pyplot as plt

class ProdGraph:
    def __init__(self, adjacency, distributions, ):
        self.adjacency = adjacency
        self.distributions = distributions
        
        G = nx.DiGraph(adjacency)
        topological_order = th.tensor(list(nx.topological_sort(G)))
        self.topological_order = topological_order
        
        
        self.remaining_time = th.zeros(adjacency.shape[0])
        self.states = th.zeros(adjacency.shape[0])
        
        
        # Initialize
        self.states[topological_order[0]] = 1
        self.lapsed_time = 0
        
        
        self.log = pd.DataFrame({'time': [], 'node': [], 'state': []})
        
    def forward(self):
        
        decision = th.where(self.states == 1, 1, 0)
        remaining_time = th.where(decision == 1, self.remaining_time, float('inf'))
        min_value, index = th.min(remaining_time, dim=0)
        lapsed_time = min_value.item()
        self.lapsed_time += lapsed_time

        
        
        self.remaining_time[index] = self.distributions[index].sample()
        
        new_row_data = pd.DataFrame([{'time': self.lapsed_time, 'node': index.item(), 'state': 'Active'}])
        self.log = pd.concat([self.log, new_row_data])
    
