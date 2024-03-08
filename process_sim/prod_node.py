import torch as th
import torch.distributions as dist

import pandas as pd

import matplotlib.pyplot as plt

class ProdNode:
    def __init__(self, batch_size, cycle_time_dist):
        self.batch_size = batch_size
        self.cycle_time_dist = cycle_time_dist
        self.remaining_time = 0
    
    def forward(self):
        lapsed_time = self.remaining_time
        self.remaining_time = self.cycle_time_dist.sample()
        return lapsed_time
    
