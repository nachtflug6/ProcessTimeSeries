from __future__ import annotations

import torch
import torch.nn.functional as F

from discrete_manufacturing_sim.fsm.production_asset_fsm import ProductionAssetFSM


def state_vector(fsm: ProductionAssetFSM, state_id: int) -> torch.Tensor:
    vector = torch.zeros(fsm.num_states)
    vector[state_id] = 1
    return vector


def event_vector(fsm: ProductionAssetFSM, event_id: int) -> torch.Tensor:
    return F.one_hot(torch.tensor(event_id), fsm.num_events).float()


def test_idle_to_producing() -> None:
    fsm = ProductionAssetFSM()
    result = fsm.transition(state_vector(fsm, 0), event_vector(fsm, 3))
    assert int(torch.argmax(result).item()) == 1


def test_producing_to_blocked() -> None:
    fsm = ProductionAssetFSM()
    result = fsm.transition(state_vector(fsm, 1), event_vector(fsm, 0))
    assert int(torch.argmax(result).item()) == 2


def test_failed_to_repaired_restarted() -> None:
    fsm = ProductionAssetFSM()
    result = fsm.transition(state_vector(fsm, 3), event_vector(fsm, 2))
    assert int(torch.argmax(result).item()) == 1


def test_invalid_transition_is_not_active() -> None:
    fsm = ProductionAssetFSM()
    idle_state = state_vector(fsm, 0)
    invalid_event = event_vector(fsm, 0)

    assert not bool(fsm.get_active_events(0)[0])
    result = fsm.transition(idle_state, invalid_event)
    assert torch.equal(result, idle_state)
