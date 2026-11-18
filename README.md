# SWOT frontal cyclonic eddy

Standalone case study applying the existing LATTE method from Elliptical
Streamfunction Parametrisation (ESP) to SWOT snapshots of two distinct frontal cyclonic
eddies (upstream and downstream). This branch is independent of the original manuscript reproduction work.

## Layout

- `functions.py` — unchanged baseline ESP/LATTE implementation from `fadb4be`.
- `AGENTS.md` — current project instructions.
- `context/study_notes.md` — agreed scope, input inventory and open questions.
- `data/cycle_004/` and `data/cycle_005/` — local SWOT NetCDF inputs, ignored.
- `archive/` — ignored local references from the original project, including
  example datasets. These are not study inputs or runtime dependencies.

## Run all eight applications

Open `analysis.ipynb` from this directory. Edit `config.json`, restart the Python
kernel, and run all cells. Configuration controls the input fields, coordinate
origin, quality mask, spatial extent, subsampling, LATTE seed and selection
radii, exposed outer-fit options, sensitivity grid and plotting settings.
Internal optimiser settings in `functions.py` are unchanged.

Requires NumPy, pandas, SciPy, Matplotlib, xarray, netCDF4, IPython and a Jupyter
frontend/kernel. All notebook code cells have been run in order in a fresh
Python process; interactive Jupyter execution has not yet been tested.

The notebook shows one observed/reconstructed pair and a Markdown summary for
each eddy/pass combination, grouped by eddy. It uses supplied absolute
geostrophic velocities. No archived datasets, new gradient derivation,
background subtraction or gap filling are used. Selection radii are provisional.
Outputs remain cleared in the saved notebook. Inspect the observation maps,
fit statuses, reconstruction and optional radius sensitivity before interpreting
the fit. Preliminary diagnostics show substantial eastern frontal-current
residuals. See `context/study_notes.md` for findings and limitations.

## Version control

Work on `swot-frontal-eddy`, committing only when requested. Keep raw data,
archived references and generated results out of Git. The sibling checkout on
`main`, manuscript releases and Zenodo records remain unchanged.

## Analysis modules and configuration

All selection/fitting code lives in `swot_processing.py`; all plotting code is
in `figures.py`. `functions.py` remains the baseline scientific implementation.
The notebook contains only setup and eight short analysis/display calls.

Config settings apply in order: shared defaults, the selected entry in `eddies`,
the snapshot entry, then `overrides[eddy][snapshot]`. Nested keys are merged.
For example, to change only the upstream inner radius on 1 November:

```json
"overrides": {
  "upstream": {
    "2023-11-01": {"latte": {"inner_radius_m": 20000}}
  }
}
```

Eddy origins: downstream (151.75, -34.75), upstream (152.5, -33.0).
No parameters propagate between them or across passes. The default 15/50 km
selection radii are provisional for every application. An unavailable fit is
shown explicitly, never replaced with invented parameters.

For optional diagnostics, use `figures.plot_radial_diagnostic(result)` and
`swot_processing.sensitivity_table(result)`, selecting an entry from the
notebook's `results[(eddy, snapshot)]` dictionary. Pairwise speed colour scales
are shared within each application, not across all eight.

Per-combination `coordinates.origin_longitude_deg` / `origin_latitude_deg`
overrides set each application's local origin. Its LATTE starting centre is
`latte.initial_center_m` (normally `[0, 0]` relative to that origin). Unspecified
combinations retain their eddy default. Summaries report both origin and fitted
centre, selection thresholds and sample counts. The orange ellipse marks
`rho = Rc / sqrt(2)`, the model tangential-speed peak ellipse; speed is not
constant around it. Red boundaries show inner and outer observation selections.

Display controls in `plot`:

```json
"show_selection_boundaries": true,
"show_starting_centre": true,
"starting_centre_color": "magenta"
```

Set either flag to false to hide that overlay. The fitted centre and orange
maximum-speed ellipse remain independent. Starting origins are fixed config
values; rerunning the notebook does not overwrite them with new fit results.

Every combination now has explicit `inner_radius_m` and `outer_radius_m` under
`overrides[eddy][snapshot].latte`. Edit those values for individual applications;
the shared radii are fallback defaults. No automatic radius search or candidate
selection runs. Upstream 11 October uses a provisional 25/50 km pair; other
combinations retain their previously effective radii.

## Inner-fit refinement

`latte.refine_inner` defaults to `true`: use the existing two-stage
`refine_latte` selection/refit. Set it to `false` in an application override
to run existing `latte` once on its initial centre/Q/radius selection, then
continue the outer radial fit. Upstream 11 October currently uses this trial
mode with the researcher's 20/50 km radii. The summary identifies the mode;
refined counts are not applicable in single-stage mode. Selection plots show
the original seed ellipse for that mode, rather than the fitted ellipse.
A converged single-stage fit does not resolve missing core coverage or
establish the identity/rotation of the targeted eddy.
