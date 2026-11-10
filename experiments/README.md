# Idealised experiments

This folder is reserved for the manuscript's idealised validation and sensitivity
experiments. **No experiment generators or runnable experiments have been added yet.**
The current executable demonstrations are in [applications/](../applications/README.md).

Planned groups include SOLO/DOPPIO/LATTE sampling geometry, method comparisons
on identical observations, noise sensitivity and background-flow sensitivity.
Specific configurations and generators will be added with the original tests.

## Conventions for future additions

- Import the shared root [`functions.py`](../functions.py); do not duplicate methods.
- Store each experiment's settings and random seeds as JSON in `configs/`.
- Use the common [`reporting.py`](../reporting.py) export conventions: a manuscript
  summary, parameter tables and run metadata with versions/configuration/hashes.
- Write generated files under `results/experiments/<experiment>/` (gitignored).
- Document which manuscript figures/tables each implemented experiment generates.

See the [root README](../README.md) for installation and repository scope.
