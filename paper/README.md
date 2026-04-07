# Paper / Reproducibility Notes

Use this folder for artifacts that directly support the accompanying paper.

## Primary paper demo

The main runnable entry point for the paper-facing simulation is:

```bash
python3 examples/minimal_simulation.py
```

A slightly varied run can be explored with:

```bash
python3 examples/minimal_simulation.py --steps 10 --nodes 3
```

This demo is the canonical reference for the current repository narrative and should stay aligned with the paper description.

## Recommended contents

- `figures/`: scripts or exported figures used in the manuscript
- `tables/`: generated tables or CSV summaries
- `scripts/`: commands that reproduce the main results
- `README.md`: a map from paper sections to code and notebooks

## Suggested workflow

1. Keep the simulation library in `discrete_manufacturing_sim/`.
2. Keep `examples/minimal_simulation.py` as the first supported runnable workflow.
3. Put polished, paper-facing demos in `examples/` or `notebooks/tutorials/`.
4. Document exactly how to regenerate each figure or table here.

> If a notebook or script is required for the paper, it should be linked from this file and from the main `README.md`.
