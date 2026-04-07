# Paper / Reproducibility Notes

Use this folder for artifacts that directly support the accompanying paper.

Recommended contents:
- `figures/`: scripts or exported figures used in the manuscript
- `tables/`: generated tables or CSV summaries
- `scripts/`: commands that reproduce the main results
- `README.md`: a map from paper sections to code and notebooks

## Suggested workflow

1. Keep the simulation library in `discrete_manufacturing_sim/`.
2. Put polished, paper-facing demos in `examples/` or `notebooks/tutorials/`.
3. Document exactly how to regenerate each figure or table here.

> If a notebook or script is required for the paper, it should be linked from this file and from the main `README.md`.
