# Elliptical Streamfunction Parametrisation (ESPresso)

Python implementations of **SOLO**, **DOPPIO** and **LATTE** for estimating ocean
eddy geometry and velocity structure from transects, gridded fields and
multi-platform observations.

These notebooks demonstrate the ESPresso framework on a cyclonic eddy in the
Tasman Sea / East Australian Current system during 2023–2024. Input datasets can
be replaced with your own velocity observations. The associated manuscript is
[*A Geometric Framework for Mapping Ocean Eddies Using Elliptical Streamfunction
Parameterisation*](https://doi.org/10.5194/egusphere-2026-3599).

## Methods

| Method | Sampling configuration | Application in this repository |
| --- | --- | --- |
| **SOLO** | A single approximately straight transect; assumes axisymmetry | Shipboard ADCP transect |
| **DOPPIO** | Two intersecting orthogonal transects | Gridded numerical-model velocities |
| **LATTE** | Irregular, scattered or multi-platform velocity observations | ADCP–satellite and drifter–satellite combinations |

The methods first estimate the inner-core parameters: centre, geometry and
angular velocity. `out_core_param_fit` then uses inner- and outer-region
observations to refine angular velocity and estimate Gaussian radius and
streamfunction amplitude.

**Scope:** this repository contains four observational/numerical application
notebooks and their input data, plus script-based supplementary noise/sampling
and background-flow experiments. Other idealised validation experiments are not
yet included.

## Repository contents

```text
.
├── README.md
├── functions.py
├── reporting.py
├── data/
├── applications/
│   ├── README.md
│   ├── configs/                  # one JSON per real-data demonstration
│   └── *.ipynb                  # four existing application notebooks
├── experiments/
│   ├── README.md                # idealised experiment instructions
│   ├── configs/                 # idealised groups and shared sensitivity settings
│   ├── run.py                   # run experiments or redraw saved results
│   ├── sensitivity.py
│   ├── plotting.py
│   ├── solo.py
│   ├── solo_plotting.py
│   ├── idealised.py
│   ├── idealised_plotting.py
│   └── outputs.py
└── results/                     # generated locally; gitignored
    ├── applications/
    └── experiments/
```

| Area | Purpose |
| --- | --- |
| [`applications/`](applications/README.md) | Real-data notebooks, configurations, dataset provenance and output instructions |
| [`experiments/`](experiments/README.md) | Script-based idealised validation, sensitivity tests and figure/results exports |
| [`functions.py`](functions.py) | Shared ESP estimation, reconstruction and selection/tracking helpers |
| [`reporting.py`](reporting.py) | Configuration/path utilities and printed/exported results summaries |
| [`data/`](data/) | Bundled ADCP, model, satellite and drifter extracts |

Applications and experiments use the same shared method implementations.

## Installation

Python **3.9 or newer**, with NumPy, SciPy, pandas, xarray, Matplotlib and netCDF4.
A Jupyter environment is required to run the notebooks interactively. For example:

```bash
python -m pip install numpy scipy pandas xarray matplotlib netCDF4 jupyterlab
```

## Run a demonstration

1. Edit its JSON file in [`applications/configs/`](applications/configs/).
2. Launch `jupyter lab` from the repository root or `applications/`.
3. Open the matching notebook, restart its kernel and run all cells in order.
4. Inspect its figures and collect its exported summary and numerical results.

See the [application guide](applications/README.md) for the notebook/config pairs,
data sources and settings. Paths in JSON are relative to the repository root;
no additional data download is needed for the default examples.

Each run saves `manuscript_summary.md`, `parameters.csv` and `run_metadata.json`
under `results/applications/<demonstration>/`. The Markdown report is intended
for manuscript updates; the CSV retains numerical precision and the JSON records
configuration, versions, hashes, counts and fit statuses. Output files are replaced
on rerun; choose another configured output directory to retain a separate run.

The notebooks retain their existing figures and diagnostic plots. The application workflow
does not require running scripts.

## Run idealised experiments

Generate all eight main-paper idealised figures (SOLO, DOPPIO, LATTE and the ESP schematic):

```bash
python experiments/run.py idealised
```

Figures, parameter tables and individual manuscript summaries are saved under
`results/experiments/<experiment>/`. The combined report is
`results/experiments/idealised_manuscript_summary.md`. Settings are in
[`experiments/configs/`](experiments/configs/). Run one group with
`solo_idealised`, `doppio_idealised`, `latte_idealised` or `schematic`.
Add `--plot-only` to redraw saved results without fitting again.

Run the supplementary sensitivity tests:

```bash
python experiments/run.py all
```

This runs the noise/sampling and background-flow tests with the configured
100 noise realisations per nonzero condition. It exports Figures S1–S3 as PNG,
trial/summary tables and manuscript reports under `results/experiments/`.

To redraw saved figures without fitting again:

```bash
python experiments/run.py all --plot-only
```

See the [experiment guide](experiments/README.md) for configuration, separate
runs, quick checks and the output/provenance files. Jupyter is not needed for
these experiments.

## Data sources

The bundled extracts combine RV *Investigator* voyage **IN2023_V06** shipboard
ADCP data, Mercator/Copernicus numerical currents, IMOS OceanCurrent satellite
velocities, and NOAA Global Drifter Program observations. Source links, cruise
information, coverage, depth/resolution and notebook selections are documented
in [Datasets and provenance](applications/README.md#datasets-and-provenance).

## Implementation notes

### Coordinates and units

The examples use a local Cartesian origin at **154.3°E, 37.9°S**, with positive
x eastward and positive y northward. Longitude/latitude offsets are converted
using 111 km per degree of latitude and the cosine of the reference latitude.

Fits use **metres**, **m/s**, angular velocity in **s⁻¹**, and streamfunction
amplitude in **m²/s**. Only display coordinates are converted to kilometres.
`Rc` is the Gaussian characteristic radius; peak normalised tangential speed
occurs at `rho = Rc / sqrt(2)`.

### CE23 ADCP timestamp correction

`decode_ce23_mat_time` applies a dataset-specific **−12-hour** correction to the
original MAT-derived labels, validated through an independent comparison with
CSIRO Data Trawler UTC positions and velocities. The raw MAT values are unchanged.
This is an encoding/interpretation correction for this file, not a general change
to the standard Julian-date convention.

SOLO’s selection anchor shifts with the decoder to preserve its original
observations. The ADCP–satellite example uses native satellite times to bracket
the corrected mean ADCP time, and plots the same interpolation used for fitting.
Already-corrected UTC data and other products’ timestamps must not be shifted.

### Fitting, geometry and tracking

- **Outer fit:** there is no default upper limit on fitted `Rc`. The 100-km
  observation-selection window is a separate setting. An explicit `Rc_max` is an
  optimizer bound; a bound-hit solution is unavailable. Optimization uses scaled
  variables and returns SI covariance. Failures and unresolved radial fits return
  NaNs, never initial guesses. Pass `fit_details={}` to retain status and diagnostics.
- **LATTE geometry:** unsuccessful, rank-deficient and non-elliptic fits are
  rejected. Q must be symmetric positive-definite with determinant one.
  Refinement uses the preliminary fitted centre and Q.
- **Tracking:** only accepted inner fits advance state. Displacement is checked
  against the previous accepted centre, and rotation sign follows the first
  accepted fit. Either rotation sign is supported. Failed outer fits leave the
  Gaussian reconstruction unavailable; `Omega_core` and `w_core` retain valid
  inner-fit estimates separately.
- **DOPPIO polynomials:** the implementation uses unshifted cubics with exact
  root constraints. Polynomial intercepts differ from tangent-line intercepts;
  derivatives and tangents are evaluated at the fitted roots.

The summaries include radius standard errors. A converged fit alone does not
establish adequate outer-core sampling or physical model validity. The plotted
95% prediction intervals are approximate and do not account for spatial or
temporal correlation.

## Citation

If you use these methods, please cite the manuscript:

Dowse, R. W., Keating, S. R., and Roughan, M. (2026).
*A Geometric Framework for Mapping Ocean Eddies Using Elliptical Streamfunction
Parameterisation*. EGUsphere [preprint].
[doi:10.5194/egusphere-2026-3599](https://doi.org/10.5194/egusphere-2026-3599).

For the code, cite the version-specific Zenodo record corresponding to the code
you use:

Dowse, R. (2026). *regdowse/elliptical_streamfunction_parametrisation: ESP*. Zenodo.

Existing archives are:

- [Original archive, DOI 10.5281/zenodo.19309976](https://doi.org/10.5281/zenodo.19309976).
- [Version v1.01, DOI 10.5281/zenodo.19364198](https://doi.org/10.5281/zenodo.19364198), cited in the manuscript.

The corrections on this revision branch are not represented by those historical
archives. The corrected release’s DOI will be added when that version is published.
Please also acknowledge the original data providers linked above.

## Author

**Reg Dowse** — UNSW Sydney

## Manuscript figure exports

All 14 main figures and supplementary Figures S1–S3 are saved as **PNG only**
in `results/manuscript_figures/`. The generated `manuscript_figures.md` in that
folder links each figure to its generator and results summary, and flags missing
PNGs. Run all four application notebooks, `python experiments/run.py idealised`,
and `python experiments/run.py all` to generate the complete set. Existing saved
experiment results can be redrawn using `--plot-only`.

Only manuscript figures are exported; diagnostic plots remain visible in the
notebooks. Figure 6 (`solo_sadcp.png`) is the combined depth-dependent SOLO figure.
Numerical outputs, run metadata and summaries stay in their application/experiment
folders. Rerunning a figure replaces its PNG. A custom application output directory
or experiment `--output-root` uses a separate `manuscript_figures/` folder alongside
its run directories, keeping custom runs out of the default manuscript folder.
