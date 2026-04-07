"""Paper-facing minimal simulation demo.

Run with:
    python3 examples/minimal_simulation.py
    python3 examples/minimal_simulation.py --steps 10 --nodes 3
"""

from argparse import ArgumentParser

from discrete_manufacturing_sim.demo import DemoConfig, build_demo, run_demo


def parse_args():
    """Parse simple CLI options for the supported demo."""
    parser = ArgumentParser(description="Run the paper-facing minimal simulation demo.")
    parser.add_argument("--steps", type=int, default=25, help="Number of simulation steps to run.")
    parser.add_argument("--nodes", type=int, default=2, help="Number of nodes in the linear production chain.")
    parser.add_argument("--length", type=int, default=2, help="Number of places/transitions per Petri net.")
    parser.add_argument("--mean", type=float, default=10.0, help="Mean value used for the uniform event timings.")
    parser.add_argument("--spread", type=float, default=2.0, help="Spread of the uniform event timings.")
    return parser.parse_args()


def main():
    """Run the supported demo from command-line arguments."""
    args = parse_args()
    config = DemoConfig(
        num_nodes=args.nodes,
        length_pns=args.length,
        mean=args.mean,
        spread=args.spread,
        steps=args.steps,
    )
    run_demo(config=config)


if __name__ == "__main__":
    main()
