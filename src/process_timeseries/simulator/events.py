"""Named states and events for the manufacturing simulator."""

from __future__ import annotations

from enum import IntEnum


class MachineState(IntEnum):
    """Machine states used by the production asset FSM."""

    IDLE = 0
    PRODUCING = 1
    BLOCKED = 2
    FAILED = 3


class EventType(IntEnum):
    """Event indices used by the current event-driven simulator."""

    CYCLE_COMPLETE = 0
    FAILURE = 1
    REPAIR_COMPLETE = 2
    INPUT_AVAILABLE = 3
    OUTPUT_AVAILABLE = 4


STATE_LABELS = {
    MachineState.IDLE: "idle",
    MachineState.PRODUCING: "producing",
    MachineState.BLOCKED: "blocked",
    MachineState.FAILED: "failed",
}

EVENT_LABELS = {
    EventType.CYCLE_COMPLETE: "cycle_time",
    EventType.FAILURE: "failure_time",
    EventType.REPAIR_COMPLETE: "repair_time",
    EventType.INPUT_AVAILABLE: "input_available",
    EventType.OUTPUT_AVAILABLE: "output_available",
}


def state_name(state_id: int) -> str:
    """Return a stable human-readable name for a numeric machine state."""
    try:
        return STATE_LABELS[MachineState(int(state_id))]
    except (KeyError, ValueError):
        return f"unknown:{state_id}"
