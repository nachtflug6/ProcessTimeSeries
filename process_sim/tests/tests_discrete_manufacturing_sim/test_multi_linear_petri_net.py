import unittest
import torch
from discrete_manufacturing_sim.multi_linear_petri_net import MultiLinearPetriNet
from discrete_manufacturing_sim.dg.wdg import WeightedDirectedGraph

class TestMultiLinearPetriNet(unittest.TestCase):
    def setUp(self):
        # Create a MultiLinearPetriNet instance with length 2 and connectivity graph
        adjacency_matrix = torch.tensor([[0, 1],  # Connection from PN 0 to PN 1
                                         [0, 0]])  # No connections from PN 1
        connectivity_graph = WeightedDirectedGraph(adjacency_matrix)
        self.mpn = MultiLinearPetriNet(length_pns=2, connectivity_graph=connectivity_graph)

    # def test_markings_and_transitions(self):
    #     # Ensure that markings and transition firing capability are correct after firing transitions
    #     expected_results = [
    #         {"cc0": 1.0, "cc1": 1.0, "mm0": 0.0, "mm1": 0.0, "t0": True, "t1": False, "ww0": 1.0, "ww1": 1.0},
    #         {"cc0": 1.0, "cc1": 1.0, "mm0": 1.0, "mm1": 0.0, "t0": False, "t1": True, "ww0": 1.0, "ww1": 1.0},
    #         {"cc0": 1.0, "cc1": 1.0, "mm0": 0.0, "mm1": 1.0, "t0": True, "t1": False, "ww0": 1.0, "ww1": 1.0},
    #         {"cc0": 1.0, "cc1": 1.0, "mm0": 1.0, "mm1": 1.0, "t0": False, "t1": False, "ww0": 1.0, "ww1": 1.0},
    #         {"cc0": 1.0, "cc1": 1.0, "mm0": 1.0, "mm1": 0.0, "t0": False, "t1": True, "ww0": 1.0, "ww1": 1.0}
    #     ]

    #     for i, expected_result in enumerate(expected_results):
    #         with self.subTest(i=i):
    #             self.mpn.update_active_transitions()
    #             self.mpn.fire_transition(i % 2, i // 2)
    #             current_state = self.mpn.get_state()
    #             self.assertEqual(current_state, expected_result)

if __name__ == '__main__':
    unittest.main()
