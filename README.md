# SWOT frontal cyclonic eddy

Standalone case study applying the existing LATTE method from Elliptical
Streamfunction Parametrisation (ESP) to SWOT snapshots of one frontal cyclonic
eddy. This branch is independent of the original manuscript reproduction work.

## Layout

- `functions.py` — unchanged baseline ESP/LATTE implementation from `fadb4be`.
- `AGENTS.md` — current project instructions.
- `context/study_notes.md` — agreed scope, input inventory and open questions.
- `data/cycle_004/` and `data/cycle_005/` — local SWOT NetCDF inputs, ignored.
- `archive/` — ignored local references from the original project, including
  example datasets. These are not study inputs or runtime dependencies.

## Run the first snapshot

Open `analysis.ipynb` from this directory. Edit `config.json`, restart the Python
kernel, and run all cells. Configuration controls the input fields, coordinate
origin, quality mask, spatial extent, subsampling, LATTE seed and selection
radii, exposed outer-fit options, sensitivity grid and plotting settings.
Internal optimiser settings in `functions.py` are unchanged.

Requires NumPy, pandas, SciPy, Matplotlib, xarray, netCDF4, IPython and a Jupyter
frontend/kernel. All notebook code cells have been run in order in a fresh
Python process; interactive Jupyter execution has not yet been tested.

The first snapshot uses supplied absolute geostrophic velocities from
1 November 2023, pass 545. No archived datasets, new gradient derivation,
background subtraction or gap filling are used. Selection radii are provisional.
Outputs remain cleared in the saved notebook. Inspect the observation maps,
fit statuses, reconstruction, residuals and radius sensitivity before interpreting
the fit. Preliminary diagnostics show substantial eastern frontal-current
residuals. See `context/study_notes.md` for findings and limitations.

## Version control

Work on `swot-frontal-eddy`, committing only when requested. Keep raw data,
archived references and generated results out of Git. The sibling checkout on
`main`, manuscript releases and Zenodo records remain unchanged.
