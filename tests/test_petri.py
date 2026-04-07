from __future__ import annotations

import pytest
import torch

from discrete_manufacturing_sim.dg.wdg import WeightedDirectedGraph
from discrete_manufacturing_sim.multi_linear_petri_net import MultiLinearPetriNet


def build_single_node_petri() -> MultiLinearPetriNet:
    adjacency = torch.zeros((1, 1), dtype=torch.float32)
    graph = WeightedDirectedGraph(adjacency)
    petri = MultiLinearPetriNet(connectivity_graph=graph, length_pns=2)
    petri.update_active_transitions()
    return petri


def test_enabled_transition_logic() -> None:
    petri = build_single_node_petri()

    assert bool(petri.active_transitions[0, 0])
    assert not bool(petri.active_transitions[0, 1])


def test_token_updates_after_firing() -> None:
    petri = build_single_node_petri()

    petri.fire_transition(0, 0)
    assert torch.equal(petri.markings[0], torch.tensor([1.0, 0.0]))

    petri.update_active_transitions()
    assert bool(petri.active_transitions[0, 1])

    petri.fire_transition(0, 1)
    assert torch.equal(petri.markings[0], torch.tensor([0.0, 1.0]))


def test_capacity_enforcement() -> None:
    petri = build_single_node_petri()
    petri.set_capacity(0, 0, 1)
    petri.set_initial_marking(0, [2, 0])

    with pytest.raises(ValueError, match="Marking exceeds capacity"):
        petri.check_capacity()


def test_negative_markings_rejected() -> None:
    petri = build_single_node_petri()

    with pytest.raises(ValueError, match="non-negative"):
        petri.set_initial_marking(0, [-1, 0])
