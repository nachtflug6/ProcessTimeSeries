"""Scriptable paper-style figures for the manufacturing simulator."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from process_timeseries.simulator.state import SimulationResult

STATE_COLORS = {
    0: "#f1c40f",  # idle
    1: "#2ecc71",  # producing
    2: "#f39c12",  # blocked
    3: "#e74c3c",  # failed
}
STATE_LABELS = {
    0: "Idle",
    1: "Producing",
    2: "Blocked",
    3: "Failed",
}


def _fill_state_bands(ax, times: np.ndarray, states: np.ndarray, ymin: float, ymax: float) -> None:
    if len(times) < 2:
        return

    for state_id, color in STATE_COLORS.items():
        matching = np.where(states == state_id)[0]
        for idx in matching:
            start = times[idx]
            end = times[idx + 1] if idx + 1 < len(times) else times[idx]
            if end > start:
                ax.fill_between([start, end], ymin, ymax, color=color, linewidth=0)


def plot_paper_figure(result: SimulationResult, output_path: str | Path | None = None):
    """Plot the paper-style multi-panel figure and optionally save it."""
    if result.buffer_levels.size == 0:
        raise ValueError("Cannot plot an empty simulation result")

    times = np.asarray(result.times, dtype=float)
    buffers = np.asarray(result.buffer_levels, dtype=float)
    states = np.asarray(result.machine_states, dtype=int)
    num_nodes = buffers.shape[1]
    max_produced = max(float(np.max(buffers[:, -1])), 1.0)
    scaling = max(max_produced / 5.0, 1.0)

    fig, axes = plt.subplots(num_nodes, 1, figsize=(12, max(8, 2.2 * num_nodes)), sharex=True)
    if num_nodes == 1:
        axes = [axes]

    for node_id, ax in enumerate(axes):
        if node_id == num_nodes - 1:
            ymin, ymax = -0.7 * scaling, -0.2 * scaling
        else:
            ymin, ymax = -0.7, -0.2

        _fill_state_bands(ax, times, states[:, node_id], ymin=ymin, ymax=ymax)
        ax.step(times, buffers[:, node_id], where="post", color="black", linewidth=1.5)
        ax.set_ylabel("Buffer Level")
        ax.grid(True, alpha=0.3)

        if node_id == num_nodes - 1:
            ax.set_title(f"Total Produced - Node {node_id + 1}")
            ax.set_ylim(min(ymin - 0.2, -scaling), float(np.max(buffers[:, node_id])) + 1.0)
        else:
            ax.set_title(f"Buffer Level over Time - Node {node_id + 1}")
            upper = max(5.5, float(np.max(buffers[:, node_id])) + 0.5)
            ax.set_ylim(ymin - 0.2, upper)
            ax.set_yticks(np.arange(0, max(6, int(np.ceil(upper)) + 1), 1.0))

    fig.text(0.5, 0.02, "Runtime (s)", ha="center")
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in STATE_COLORS.values()]
    fig.legend(legend_handles, list(STATE_LABELS.values()), loc="upper right", bbox_to_anchor=(0.97, 0.14))
    plt.tight_layout(rect=(0, 0.03, 1, 1))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig
