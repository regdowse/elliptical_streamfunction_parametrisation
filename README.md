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

The planned analysis entry point is `analysis.ipynb`; add `swot_processing.py`
only as reusable processing becomes useful. Neither has been created yet.

## Current status and requirements

Repository cleanup is complete. Data inspection and scientific processing have
not begun. Target location, velocity inputs, quality filtering and fit settings
remain to be established; see the tracked study notes.

Importing `functions.py` requires NumPy, pandas, SciPy and Matplotlib, plus the
Python standard library. SWOT analysis will also need a NetCDF reader (for
example xarray with netCDF4) and Jupyter for notebooks. No environment has been
installed or validated for this study yet.

Fits should use consistent local east/north coordinates and units. Establish
and document the observation/velocity definitions before calling LATTE; do not
assume SSH directly supplies horizontal velocity. Check fitting diagnostics and
sensitivity rather than treating convergence as physical validation.

## Version control

Work on `swot-frontal-eddy`, committing only when requested. Keep raw data,
archived references and generated results out of Git. The sibling checkout on
`main`, manuscript releases and Zenodo records remain unchanged.
