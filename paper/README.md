# Paper / Reproducibility Notes

This folder collects references for reproducing the paper-facing 5-node manufacturing demo.

## Primary reproduction command

```bash
python3 scripts/reproduce_wsc2024.py
```

This command loads `configs/paper/wsc2024_demo_5_node.yaml` and writes outputs to `experiments/wsc2024/outputs/`.

## Outputs

The supported reproduction path produces:

- `buffer_levels.csv`
- `machine_states.csv`
- `summary.json`
- `figure5.png`

## Related references

- `docs/paper_map.md` — paper concept → code traceability
- `docs/output_format.md` — output artifact schema
- `notebooks/tutorials/introduction.ipynb` — tutorial / convenience notebook layered on top of the scriptable path
