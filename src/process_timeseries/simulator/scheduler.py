"""Readable scheduler helpers mirroring the simulator's next-event logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from process_timeseries.simulator.events import EVENT_LABELS, EventType


@dataclass(frozen=True)
class ScheduledEvent:
    """A small structured description of the next scheduled event."""

    node_id: int
    event_index: int
    event_name: str
    event_time: float


def select_minimum_time_event(event_schedule: Any, active_mask: Any) -> ScheduledEvent | None:
    """Select the minimum-time active event in a deterministic, testable way."""
    schedule = np.asarray(event_schedule, dtype=float)
    mask = np.asarray(active_mask, dtype=bool)

    if schedule.shape != mask.shape:
        raise ValueError("event_schedule and active_mask must have the same shape")
    if not mask.any():
        return None

    masked = np.where(mask, schedule, np.inf)
    flat_index = int(np.argmin(masked))
    node_id, event_index = np.unravel_index(flat_index, masked.shape)
    event_time = float(masked[node_id, event_index])

    return ScheduledEvent(
        node_id=int(node_id),
        event_index=int(event_index),
        event_name=EVENT_LABELS.get(EventType(int(event_index)), f"event_{event_index}"),
        event_time=event_time,
    )


def preview_next_event(sim_handler: Any) -> ScheduledEvent | None:
    """Preview the next temporal event without mutating simulator state."""
    sim_handler.update()
    active_temporal = sim_handler.active_events & sim_handler.temporal_events
    return select_minimum_time_event(
        sim_handler.multi_temp_event_handler.event_schedule.detach().cpu().numpy(),
        active_temporal.detach().cpu().numpy(),
    )
