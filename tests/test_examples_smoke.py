from examples.minimal_simulation import build_demo, run_demo


def test_build_demo_smoke():
    sim_handler = build_demo()

    for _ in range(3):
        sim_handler.simulate()

    assert sim_handler.num_nodes == 2
    assert sim_handler.state_matrix.shape == (2, sim_handler.num_states)
    assert sim_handler.mlin_pn.markings.shape == (2, 2)


def test_run_demo_returns_handler():
    sim_handler = run_demo(steps=2)
    assert sim_handler.num_nodes == 2
