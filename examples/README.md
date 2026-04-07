# Examples

Supported runnable demonstrations for the repo.

## Main demo

- `minimal_simulation.py` — the primary paper-facing entry point and the recommended place to start

Run from the repository root:

```bash
python3 examples/minimal_simulation.py
python3 examples/minimal_simulation.py --steps 10 --nodes 3
```

## What this demo does

The demo builds a small linear production chain, runs the discrete-event simulation loop, and prints a concise summary of the resulting runtime, FSM states, and Petri-net markings.

## Main knobs

- `--steps` — how many simulation iterations to execute
- `--nodes` — how many production nodes are in the linear chain
- `--length` — Petri-net length per node
- `--mean` / `--spread` — timing distribution settings for the demo events

## Conceptual model

The supported demo intentionally focuses on three ideas:
1. **Petri net** → tracks flow/material state
2. **FSM** → tracks production asset state
3. **event scheduler** → chooses the next simulated event in time

