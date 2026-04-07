"""Petri-net module exports.

The current repository already contains the working Petri-net-style machinery in
`discrete_manufacturing_sim.multi_linear_petri_net`. This module exposes that
implementation under the cleaner public package layout without changing the core
algorithm.
"""

from discrete_manufacturing_sim.multi_linear_petri_net import MultiLinearPetriNet

__all__ = ["MultiLinearPetriNet"]
