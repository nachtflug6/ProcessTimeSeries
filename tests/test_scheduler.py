from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from discrete_manufacturing_sim.multi_temporal_event_handler import MultiTemporalEventHandler
from process_timeseries.simulator.scheduler import select_minimum_time_event


class DummySampler(SimpleNamespace):
    def sample(self, i: int, j: int) -> torch.Tensor:
        return torch.tensor(1.0)


def test_correct_minimum_time_event_selection() -> None:
    event = select_minimum_time_event(
        event_schedule=[[5.0, 9.0, 7.0], [4.0, 3.0, 8.0]],
        active_mask=[[True, False, True], [False, True, True]],
    )

    assert event is not None
    assert event.node_id == 1
    assert event.event_index == 1
    assert event.event_time == pytest.approx(3.0)


def test_deterministic_behavior_under_fixed_inputs() -> None:
    schedule = [[6.0, 2.0, 5.0], [4.0, 1.5, 7.0]]
    mask = [[False, True, True], [True, True, False]]

    first = select_minimum_time_event(schedule, mask)
    second = select_minimum_time_event(schedule, mask)

    assert first == second


def test_correct_clock_advancement() -> None:
    sampler = DummySampler(num_instances=2, num_distributions=3)
    handler = MultiTemporalEventHandler(sampler)
    handler.event_schedule = torch.tensor([[5.0, 2.0, 9.0], [4.0, 1.0, 7.0]])
    active = torch.tensor([[False, True, False], [True, True, False]])

    node_id, event_idx = handler.find_min_element(active)

    assert (int(node_id), int(event_idx)) == (1, 1)
    assert handler.runtime == pytest.approx(1.0)
    assert handler.event_schedule[0, 1].item() == pytest.approx(1.0)
    assert handler.event_schedule[1, 0].item() == pytest.approx(3.0)
