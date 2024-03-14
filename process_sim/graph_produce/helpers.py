import numpy as np
import torch as th

# Function to generate a random DAG adjacency matrix
def generate_random_dag(n, density):
    # Initialize adjacency matrix with zeros
    adjacency_matrix = np.zeros((n, n), dtype=int)
    
    # Create directed edges randomly
    for i in range(n):
        # Choose a random target node for the outgoing edge
        if i + 1 < n:
            target_node = np.random.randint(i + 1, n)
            adjacency_matrix[i, target_node] = 1
    
    return adjacency_matrix

def generate_random_distributions(n, k, expected_value, variance_degree, distribution_types=['uniform', 'normal', 'exponential']):
    distributions = []
    mean = expected_value
    for i in range(n):
        dists = []
        for j in range(k):
            distribution_type = np.random.choice(distribution_types)
            
            if distribution_type == 'uniform':
                # Generate a random uniform distribution within a reasonable range
                low =  mean * (1 - variance_degree) + (j * 100 * mean)
                high = mean * (1 + variance_degree) + (j * 100 * mean)
                dists.append(th.distributions.Uniform(low, high))
                
            elif distribution_type == 'normal':
                # Generate a random normal distribution with mean and standard deviation adjusted for the number of distributions
                std_dev = mean * variance_degree 
                dists.append(th.distributions.Normal(mean  + (j * 100 * mean), std_dev))
                
            elif distribution_type == 'exponential':
                # Generate a random exponential distribution with mean adjusted for the number of distributions
                lambd = 1 /  (mean + (j * 100 * mean))
                dists.append(th.distributions.Exponential(1 / lambd))
        distributions.append(dists)
    return distributions