"""Documented output column names used by exported simulation artifacts."""

BUFFER_LEVEL_COLUMNS = ["time", "node_id", "buffer_level"]
MACHINE_STATE_COLUMNS = ["time", "node_id", "state", "state_id"]

__all__ = ["BUFFER_LEVEL_COLUMNS", "MACHINE_STATE_COLUMNS"]
