import matplotlib.pyplot as plt
import pandas as pd


def plot_time_series_color(graph, n):
    # Create a figure and axis for each subplot
    fig, axs = plt.subplots(n, 1, sharex=True, figsize=(8, 1*n))
    
    # Define color mapping
    colors = {0: 'green', 1: 'yellow', 2: 'red'}
    
    # Plot each time series with colored backgrounds
    for i, ax in enumerate(axs):
        node_data = graph.log[graph.log['node'] == i]
        time_series = node_data['time']
        state_series = node_data['state']
        for t, state in zip(time_series, state_series):
            ax.axvspan(t - 1, t + 1, facecolor=colors[state], alpha=0.7)
        ax.set_yticks([0])
        ax.set_yticklabels(['0'])
        ax.set_ylabel(f'Node {i}')
    
    # Add legend
    legend_labels = {'0': 'Producing', '1': 'Idle', '2': 'Blocked'}
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[int(label)]) for label in legend_labels.keys()]
    fig.legend(legend_handles, legend_labels.values(), loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Set common xlabel
    plt.xlabel('Time (s)')
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
def plot_time_series_producing(graph, n):
    # Create a figure and axis for each subplot
    fig, axs = plt.subplots(n, 1, sharex=True, figsize=(8, 1*n))
    
    # Define color mapping
    colors = {0: 'green', 1: 'yellow', 2: 'red'}
    
    # Plot each time series with colored backgrounds
    for i, ax in enumerate(axs):
        node_data = graph.log[graph.log['node'] == i]
        time_series = node_data['time']
        state_series = node_data['output_buffer']
        
        ax.plot(time_series, state_series)
        ax.set_yticks([0, state_series.max()])  # Adjust y-ticks
        #ax.set_yticklabels(['1', '0'])  # Adjust y-tick labels
        ax.set_ylabel(f'Node {i}')
    
    # Add legend
    #legend_labels = {'0': 'Producing', '1': 'Idle', '2': 'Blocked'}
    #legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[int(label)]) for label in legend_labels.keys()]
    #fig.legend(legend_handles, legend_labels.values(), loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Set common xlabel
    plt.xlabel('Time (s)')
    
    # Adjust layout
    plt.tight_layout()
    
    # Show the plot
    plt.show()
