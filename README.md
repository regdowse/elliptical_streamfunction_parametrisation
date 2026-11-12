# Elliptical Streamfunction Parametrisation (ESPresso)

Python implementations of **SOLO**, **DOPPIO** and **LATTE** for estimating ocean
eddy geometry and velocity structure from horizontal velocity observations.
This repository accompanies *A Geometric Framework for Mapping Ocean Eddies
Using Elliptical Streamfunction Parameterisation* (https://doi.org/10.5194/egusphere-2026-3599) and provides the idealised
tests and real-data applications used in the manuscript.

| Method | Observation geometry |
| --- | --- |
| **SOLO** | A single approximately straight transect; assumes an axisymmetric eddy |
| **DOPPIO** | Two intersecting orthogonal transects |
| **LATTE** | Scattered or multi-platform velocity observations |

The methods estimate inner-core parameters, then fit a Gaussian radial profile
to refine angular velocity and estimate the characteristic radius and
streamfunction amplitude. Applications and experiments use the same estimators
in [`functions.py`](functions.py).

## Installation

Install Python with NumPy, SciPy, pandas, xarray, Matplotlib and netCDF4.
JupyterLab is needed for the application notebooks:

```bash
python3 -m pip install numpy scipy pandas xarray matplotlib netCDF4 jupyterlab
```

Run the commands below from the repository root. The default application inputs
are bundled in [`data/`](data/); no additional download is required.

## Reproduce the manuscript

1. Run the four notebooks listed in the [application guide](applications/README.md).
   Restart each kernel and run all cells in order, using its supplied JSON configuration.
2. Run the main idealised tests and schematic:

   ```bash
   python3 experiments/run.py idealised
   ```

3. Run the supplementary noise/sampling and background-flow tests:

   ```bash
   python3 experiments/run.py all
   ```

The [experiment guide](experiments/README.md) describes the test designs,
individual commands and configuration files. Experiments use scripts and do not
require Jupyter. Once experiment results exist, append `--plot-only` to either
command to redraw the figures without rerunning the fits.

## Figures and numerical results

All **14 main figures and supplementary Figures S1–S3** are exported as PNGs to:

```text
results/
├── manuscript_figures/          # all 17 PNGs and manuscript_figures.md index
├── applications/               # fitted parameters, summaries and provenance
└── experiments/                # test results, tables, summaries and provenance
```

The generated `manuscript_figures.md` index links each PNG to its generator and
results summary, and identifies missing exports. Only manuscript figures are
automatically saved; notebooks also display diagnostics for inspecting fits.

Each application or experiment saves a **`manuscript_summary.md`** with its
settings and numerical results, alongside full-precision CSVs and
`run_metadata.json`. Experiments also save rendering provenance in
`plot_metadata.json`. The main idealised groups produce a combined summary at
`results/experiments/idealised_manuscript_summary.md`.

Results are generated locally and gitignored. Rerunning replaces the corresponding
outputs. Use a separate application `output_dir` or experiment `--output-root`
to retain another run; custom runs use their own figure directory, as described
in the two guides.

## Repository guide

| Location | Contents |
| --- | --- |
| [`applications/`](applications/README.md) | Four real-data notebooks, their configurations and dataset provenance |
| [`experiments/`](experiments/README.md) | Main idealised tests, supplementary tests and command-line runner |
| [`functions.py`](functions.py) | Shared estimation, reconstruction and selection/tracking functions |
| [`reporting.py`](reporting.py) | Application configuration and numerical results exports |
| [`manuscript_figures.py`](manuscript_figures.py) | Manuscript PNG registry, export helper and generated index |
| [`data/`](data/) | Shipboard ADCP, numerical-model, satellite and drifter extracts |

## Coordinates, units and interpretation

The applications use a local Cartesian origin at **154.3°E, 37.9°S**, with x
positive eastward and y positive northward. Fits use metres, velocities in m/s,
angular velocity in s⁻¹ and streamfunction amplitude in m²/s. Plotted horizontal
coordinates are generally in kilometres. `Rc` is the Gaussian characteristic
radius; the maximum normalised tangential velocity occurs at `rho = Rc / sqrt(2)`.

Observation-selection radii and fitted `Rc` are separate quantities. The default
outer fitter has no upper radius bound; failed or unresolved fits return missing
parameters. Reports distinguish preliminary inner-core rates from final refined
rates. A converged fit does not itself establish adequate sampling or physical
model validity. Plotted prediction intervals are approximate and do not account
for spatial or temporal correlation.

### CE23 ADCP timestamp correction

`decode_ce23_mat_time` applies a **−12-hour correction** to the bundled ADCP MAT
file's decoded labels, checked against independent CSIRO Data Trawler UTC data.
The raw values are unchanged. This correction is specific to that file; do not
apply it to already-corrected UTC data or other datasets. Application selections
and satellite interpolation use the corrected times.

To apply ESP to another dataset, use the general functions with positions and
velocity components in SI units. Adapt the data loader, coordinate conversion
and selection settings to the observations; changing a file path alone does not
adapt an example to another data format or sampling geometry.

## Citation

Please cite the manuscript and the archived code version used:

Dowse, R. W., Keating, S. R., and Roughan, M. (2026).
*A Geometric Framework for Mapping Ocean Eddies Using Elliptical Streamfunction
Parameterisation*. EGUsphere [preprint].
[doi:10.5194/egusphere-2026-3599](https://doi.org/10.5194/egusphere-2026-3599).

Archived code versions: [original archive](https://doi.org/10.5281/zenodo.19309976)
and [v1.01](https://doi.org/10.5281/zenodo.19364198). These identify those releases;
use the version-specific DOI corresponding to the code you run. Please also
acknowledge the providers listed in the [data provenance guide](applications/README.md#datasets-and-provenance).

**Author:** Reg Dowse — UNSW Sydney
