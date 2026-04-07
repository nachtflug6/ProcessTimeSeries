# ProcessTimeSeries

`ProcessTimeSeries` is a discrete manufacturing simulation and process time-series research repository built around the `discrete_manufacturing_sim` package.

This repository is being reorganized into a cleaner **research codebase for ongoing development** that also works well as a **paper companion repo**.

## Quick start

The verified local workflow is **CPU-first** on Linux/WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python3 examples/minimal_simulation.py
python3 -m pytest
```

> Note: the `torch` install can be large on Linux. The current repository has been verified in a local `.venv` using this CPU-first path.

## Repository layout

```text
ProcessTimeSeries/
├── discrete_manufacturing_sim/   # core simulation library
├── examples/                     # supported runnable demos
├── notebooks/                    # tutorial and exploratory notebooks
├── paper/                        # paper-facing reproducibility notes
├── tests/                        # automated tests
└── docs/assets/figures/          # generated or curated figures
```

## Supported entry points

- `examples/minimal_simulation.py` — smallest end-to-end runnable demo
- `main.py` — compatibility wrapper that launches the minimal demo
- `python3 -m pytest` — preferred test command

## Development environment

The codebase depends on **PyTorch**, `numpy`, `pandas`, `networkx`, and plotting utilities.

For day-to-day development, prefer a local CPU environment first and treat CUDA/cluster execution as an additional deployment target.

For local Linux/WSL work and container-based environments, these references remain useful:
- WSL + VS Code: <https://learn.microsoft.com/windows/wsl/tutorials/wsl-containers>
- CUDA on WSL: <https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl>

## Paper / reproducibility

Use the `paper/` folder for figure/table regeneration notes and links to any notebook or script that supports the manuscript.

## Current cleanup direction

- move demos out of the repo root
- keep exploratory notebooks separate from supported walkthroughs
- strengthen test coverage around the core simulation objects
- make install/test/demo steps reproducible from a fresh clone

