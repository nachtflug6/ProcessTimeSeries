"""Helpers for building and validating DAG-based manufacturing graphs."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import networkx as nx
import torch

from discrete_manufacturing_sim.dg.wdg import WeightedDirectedGraph


def build_adjacency_matrix(num_nodes: int, edges: Iterable[Mapping[str, Any]]) -> torch.Tensor:
    """Build a weighted adjacency matrix from a simple edge-list config."""
    if num_nodes <= 0:
        raise ValueError("graph.num_nodes must be a positive integer")

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    for edge in edges:
        source = int(edge["source"])
        target = int(edge["target"])
        weight = float(edge.get("weight", 1.0))

        if source < 0 or target < 0 or source >= num_nodes or target >= num_nodes:
            raise ValueError(f"Edge ({source} -> {target}) is out of bounds for num_nodes={num_nodes}")
        if source == target:
            raise ValueError("Self-loops are not supported; the manufacturing graph must remain a DAG")
        if weight <= 0:
            raise ValueError("Edge weights must be strictly positive")

        adjacency[source, target] = weight

    return adjacency


def validate_dag(adjacency_matrix: torch.Tensor) -> nx.DiGraph:
    """Validate the current graph is acyclic and return a NetworkX graph view."""
    graph = nx.from_numpy_array(adjacency_matrix.detach().cpu().numpy(), create_using=nx.DiGraph)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(
            "The manufacturing graph must be a DAG. Cycles and rework loops are not supported in the current simulator."
        )
    return graph


def topological_order(adjacency_matrix: torch.Tensor) -> list[int]:
    """Return a stable topological ordering for a validated DAG."""
    graph = validate_dag(adjacency_matrix)
    return list(nx.topological_sort(graph))


def build_graph_from_config(config: Mapping[str, Any]) -> WeightedDirectedGraph:
    """Create the weighted manufacturing graph from a loaded YAML config."""
    graph_config = config.get("graph", {})
    num_nodes = int(graph_config.get("num_nodes", 0))
    edges = graph_config.get("edges", [])
    adjacency = build_adjacency_matrix(num_nodes, edges)
    validate_dag(adjacency)
    return WeightedDirectedGraph(adjacency)
