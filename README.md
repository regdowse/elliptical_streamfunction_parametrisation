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
notebooks and their input data. Generators for the manuscript’s idealised and
sensitivity experiments are not included.

## Repository contents

```text
.
├── data/
│   ├── sadcp_data.mat
│   ├── gridded_numerical_data.nc
│   ├── satellite_data.nc
│   └── drifter_data.nc
├── functions.py
├── reporting.py
├── SOLO_transect_data.ipynb
├── DOPPIO_gridded_data.ipynb
├── LATTE_app1_multiplatform_data.ipynb
└── LATTE_app2_multiplatform_data.ipynb
```

| File | Purpose |
| --- | --- |
| [`functions.py`](functions.py) | ESP model evaluation, parameter estimation and selection/tracking helpers |
| [`reporting.py`](reporting.py) | Printed manuscript results summaries |
| [`SOLO_transect_data.ipynb`](SOLO_transect_data.ipynb) | Single-transect ADCP application |
| [`DOPPIO_gridded_data.ipynb`](DOPPIO_gridded_data.ipynb) | Numerical-model application and temporal tracking |
| [`LATTE_app1_multiplatform_data.ipynb`](LATTE_app1_multiplatform_data.ipynb) | ADCP and satellite application, with individual-platform comparisons |
| [`LATTE_app2_multiplatform_data.ipynb`](LATTE_app2_multiplatform_data.ipynb) | Drifter and satellite tracking, with a drifter-only inner-core comparison |

## Installation and usage

### Requirements

Python **3.9 or newer**, with NumPy, SciPy, pandas, xarray, Matplotlib and netCDF4.
A Jupyter environment is required to run the notebooks interactively.

For example, install the dependencies and JupyterLab in your Python environment:

```bash
python -m pip install numpy scipy pandas xarray matplotlib netCDF4 jupyterlab
```

### Run an application

1. Open a terminal in the repository root and launch `jupyter lab`.
2. Open one of the four application notebooks.
3. Restart its kernel and run all cells in order.
4. Inspect the figures and the final **Manuscript results summary**.

Keep `functions.py`, `reporting.py` and `data/` alongside the notebooks; relative
paths assume the repository root is the working directory. No additional data
download is needed to run the bundled examples.

### Results summaries

The final cell reports the current run’s UTC intervals, selection settings,
observation counts, fitted parameters and velocity-reconstruction R². Tracking
notebooks also report actual plotted snapshot dates and time-series ranges.
Additional precision, package versions and file hashes help compare runs.

The reports distinguish **velocity-reconstruction R²** from **normalised
tangential-fit R²**, and identify unavailable fits. The drifter-only comparison
reports inner-core results separately, without inventing outer-core parameters.
Use these computed summaries when checking or updating manuscript values; the
current branch includes corrections to the original archived implementation.

## Datasets and provenance

The bundled files are study extracts, not the complete source archives. File
coverage below was checked against the arrays and embedded metadata. **All
observation times are UTC.** Notebook selections are narrower than file coverage.

| Bundled file | Provider / data type | Bundled coverage | Used by |
| --- | --- | --- | --- |
| [`sadcp_data.mat`](data/sadcp_data.mat) | RV *Investigator*, IN2023_V06; shipboard ADCP | 8 October–31 October 2023 after the MAT clock correction; 2,174 records and 79 depth levels | SOLO; LATTE application 1 |
| [`gridded_numerical_data.nc`](data/gridded_numerical_data.nc) | Mercator Ocean International / Copernicus Marine; daily model currents | 19 October 2023–11 February 2024; 149–158°E, 42–34°S | DOPPIO |
| [`satellite_data.nc`](data/satellite_data.nc) | IMOS OceanCurrent / AODN; gridded surface geostrophic currents | 18 October 2023 18:00–11 February 2024 06:00; 149–158°E, 42–34°S | Both LATTE applications |
| [`drifter_data.nc`](data/drifter_data.nc) | NOAA Global Drifter Program; quality-controlled six-hourly trajectories and velocities | 19 October 2023–16 August 2024; two drifters, 2,403 records | LATTE application 2 |

### Shipboard ADCP — RV *Investigator*, IN2023_V06

The observations were collected during the CSIRO Marine National Facility voyage
**IN2023_V06**, *Understanding eddy interactions and their impacts in the East
Australian Current System*, led by Moninya Roughan (UNSW). The
[CSIRO voyage page](https://www.csiro.au/en/about/facilities-collections/MNF/Voyages-schedules/Voyages/2023/October/IN2023_V06)
lists the voyage as **9 October–2 November 2023**, Sydney to Sydney. Voyage records,
ADCP metadata and processing reports are linked from
[CSIRO Data Trawler](https://www.cmar.csiro.au/data/trawler/survey_details.cfm?survey=IN2023_V06).

The MAT extract contains eastward/northward velocities (`u_sadcp`, `v_sadcp`),
positions, depth bins and timestamps. Depths span **27–807 m**, with approximately
**15-minute** sampling. Both ADCP examples select the **37-m** bin.

The corrected extract spans **8 October 2023 23:56:08–31 October 2023 16:37:59**
(rounded to seconds). The notebook subsets are:

- **SOLO:** 96 original observations, **17 October 01:42:31–18 October 01:27:30**.
- **LATTE application 1:** 433 observations south of or on 37°S,
  **16 October 22:27:31–21 October 10:27:30**. Satellite interpolation targets their
  mean time, **19 October 04:27:31**.

These are analysis subsets; neither represents the complete voyage record.
See [the timestamp correction](#ce23-adcp-timestamp-correction) below.

### Gridded numerical velocities — Mercator / Copernicus Marine

The file identifies **Mercator Ocean International**, source **MOI GLO12**, and
*daily mean fields from Global Ocean Physics Analysis and Forecast*. It contains
`uo` and `vo` on an approximately **1/12°** grid (109 longitudes × 97 latitudes),
at the near-surface model depth **0.494025 m**.

Source access: [Copernicus Marine Data Store](https://data.marine.copernicus.eu/).
The provider’s [Global Ocean Physics Analysis and Forecast description](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description)
describes this product family. The precise original download request and dataset
identifier are not retained in the bundled file.

The DOPPIO notebook restricts the record to **27 January 2024**. Some encoded
model timestamps differ from nominal midnight by a few minutes because the
bundled time offsets use single precision. The notebook summaries expose the
actual decoded timestamps; this is separate from the ADCP clock correction.

### Satellite velocities — IMOS OceanCurrent / AODN

The extract contains gridded sea level and surface geostrophic velocities
(`UCUR`, `VCUR`) from **IMOS OceanCurrent**, distributed by the **Australian Ocean
Data Network (AODN)**. The grid spacing is **0.2° (1/5°)**, with 46 longitudes ×
41 latitudes. Access the source through the
[IMOS OceanCurrent GSLA catalogue](https://thredds.aodn.org.au/thredds/catalog/IMOS/OceanCurrent/GSLA/catalog.html),
which provides near-real-time and delayed-mode products. The bundled file records
`product_type=NRT00`.

There are 121 native analysis times; their spacing is mostly daily but is not
uniform. LATTE application 1 interpolates directly from native samples to the
corrected mean ADCP time. Application 2 constructs a daily **06:00 UTC** analysis
grid through **27 January 2024**.

Acknowledgement supplied with the data:

> Data was sourced from the Integrated Marine Observing System (IMOS) - IMOS is a national collaborative research infrastructure, supported by the Australian Government.

### Drifter velocities — NOAA Global Drifter Program

The bundled metadata identify the **Global Drifter Program six-hour interpolated,
quality-controlled product**, with eastward/northward velocities `ve` and `vn`.
Data access: [NOAA AOML GDP ERDDAP](https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/drifter_6hour_qc.html).
Dataset citation: [Lumpkin and Centurioni (2019), DOI 10.25921/7ntx-z961](https://doi.org/10.25921/7ntx-z961).
The study also references the [NOAA OSMC observation portal](https://osmc.noaa.gov/erddap/tabledap/OSMC_flattened.html);
the bundled file specifically identifies the GDP QC product.

| Drifter ID in the file | WMO identifier | First bundled observation | Last bundled observation |
| --- | --- | --- | --- |
| `300534062034380` | `5802097` | 19 October 2023 06:00 | 16 August 2024 00:00 |
| `300534062784620` | `5501721` | 22 October 2023 12:00 | 16 August 2024 00:00 |

The file records a **15-m drogue centre depth**. LATTE application 2 selects data
through **27 January 2024 06:00** and associates drifter velocities with each daily
satellite field using a **±1-day window**. Its analysis therefore uses only part
of the bundled drifter record.

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
