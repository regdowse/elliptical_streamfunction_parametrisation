# Real-data applications

Four notebooks demonstrate ESP on a cyclonic eddy in the Tasman Sea / East
Australian Current system during 2023–2024. They use the shared
[`functions.py`](../functions.py), with one JSON configuration per demonstration.

## Run a demonstration

1. Install the dependencies in the [root README](../README.md#installation).
2. Open the notebook and its configuration from the table below. Use the supplied
   configuration to reproduce the example; edit it to explore another selection.
3. Restart the kernel and **run all cells in order**. Notebooks can be launched
   from the repository root or `applications/`.
4. Collect manuscript PNGs from `results/manuscript_figures/` and numerical
   outputs from `results/applications/<demonstration>/`.

| Demonstration | Notebook | Configuration | Results subfolder |
| --- | --- | --- | --- |
| SOLO shipboard ADCP | [SOLO_transect_data.ipynb](SOLO_transect_data.ipynb) | [solo.json](configs/solo.json) | `solo/` |
| DOPPIO numerical velocities | [DOPPIO_gridded_data.ipynb](DOPPIO_gridded_data.ipynb) | [doppio.json](configs/doppio.json) | `doppio/` |
| LATTE ADCP–satellite | [LATTE_app1_multiplatform_data.ipynb](LATTE_app1_multiplatform_data.ipynb) | [latte_adcp_satellite.json](configs/latte_adcp_satellite.json) | `latte_adcp_satellite/` |
| LATTE drifter–satellite | [LATTE_app2_multiplatform_data.ipynb](LATTE_app2_multiplatform_data.ipynb) | [latte_drifter_satellite.json](configs/latte_drifter_satellite.json) | `latte_drifter_satellite/` |

### Configuration

Paths in JSON are relative to the repository root; absolute paths are also
accepted. `CONFIG_PATH` in the first cell selects the configuration. Restart and
run all cells after edits so the exported settings match the computation.

| Section | Controls |
| --- | --- |
| `inputs` | Bundled MAT/NetCDF file paths |
| `coordinates` | Local longitude/latitude origin |
| `analysis` | Observation selection, search windows, initial estimates and tracking acceptance |
| `outer_fit` | Optimisation budget and optional `Rc_max_m` bound; `null` leaves the fitted radius unbounded |
| `plot` | Manuscript snapshots, map limits, quivers and section/slice settings, as applicable |
| `reconstruction` | SOLO reconstruction grid |
| `output_dir` | Numerical results and report directory |

Configuration keys specify units. Selection thresholds control which observations
enter a fit; they are not bounds on the fitted eddy radius. Notebook diagnostics
are displayed without automatic figure exports.

## Manuscript figures

All filenames below are PNGs in `results/manuscript_figures/`.

| Figure | Filename | Content |
| --- | --- | --- |
| 6 | `solo_sadcp.png` | SOLO horizontal slices and depth-dependent centreline, zonal vorticity section, and Ω/Rc profiles |
| 9 | `doppio_num.png` | Numerical velocity observations and DOPPIO reconstruction |
| 10 | `doppio_num_analysis.png` | Five eddy-centred velocity snapshots and parameter time series |
| 12 | `latte_sadcp_sat.png` | Combined ADCP–satellite observations, LATTE reconstruction and radial fit |
| 13 | `latte_drifter_sat.png` | Drifter–satellite observations and LATTE reconstruction |
| 14 | `latte_multi_analysis.png` | Multi-platform and drifter-only parameter time series |

### SOLO: reconstruction through depth

The transect is fitted once, starting at **37 m** and proceeding through measured
depth bins. Each accepted inner centre seeds the next along-track search. The
current fixed thresholds are a **20-km inner-core half-width** about the
preliminary normal-velocity crossing, a **100-km search half-width** about the
preceding centre's projection, and an **80-km outer-selection radius** about the
new centre. An inner-fit rejection stops tracking; an outer-fit failure leaves
a gap in final Ω/Rc without discarding an accepted centre.

Figure 6 combines three horizontal reconstructions with the centreline, a zonal
relative-vorticity section through the 37-m centre, and final Ω/Rc profiles.
`plot.slice_depths_m` requests **37, 280 and 537 m**, selecting measured bins
**37, 277 and 537 m**. A `null` value selects shallowest, middle and deepest
supported reconstructions. Explicitly requested slices require supported outer
fits. Offsets reference the 37-m centre; its section plane stays fixed with depth.
The red dashed section curves intersect the maximum tangential velocity contour
at that plane, accounting for centre displacement at each depth.

The analysis assumes an axisymmetric eddy at each depth and reconstructs
horizontal velocity only. The deepest accepted fit is a sampling/fit limit,
not an estimate of the eddy's bottom. `plot.diagnostic_depth_m` currently selects
277 m and affects the displayed diagnostics, independently of Figure 6's slices.

### DOPPIO: numerical-model tracking

At each time, two orthogonal transects are extracted around the preceding
accepted centre. The default transect half-length is **50 km** and the outer
selection radius is **100 km**. The configured record ends on **27 January 2024**.

Figure 9 uses the configured **22 October 2023 00:00 UTC** snapshot. Figure 10
uses **5, 25 and 28 December 2023, and 2 and 20 January 2024**, selected by
`plot.analysis_snapshots_utc`. Snapshot selection uses the configured time
tolerance and reports the actual decoded times.

### LATTE: ADCP–satellite observations

The notebook combines **37-m ADCP** velocities with satellite velocities
interpolated to the mean selected ADCP time, **19 October 2023 04:27:31 UTC**.
The supplied selection uses observations south of or on 37°S; the default
inner/outer selection radii are **35/100 km**. Figure 12 uses `plot.source: "multi"`.

### LATTE: drifter–satellite tracking

Daily analyses at **06:00 UTC** combine satellite velocities with drifter
observations within **±24 hours**. The default inner/outer radii are **35/100 km**,
and the record ends on **27 January 2024 06:00 UTC**. Figure 13 uses the configured
**29 October 2023 06:00 UTC** snapshot.

Figure 14 compares the multi-platform and drifter-only estimates. Rc and
velocity-reconstruction R² are available only for the multi-platform fit;
the drifter-only analysis estimates inner-core parameters.

## Numerical outputs

Each demonstration saves the following in its configured `output_dir`:

| File | Contents |
| --- | --- |
| `manuscript_summary.md` | Settings, selections, counts, fitted results and fit-quality measures |
| `parameters.csv` | Full-precision SI parameters; Q expanded into `q11`, `q12`, `q21`, `q22` |
| `run_metadata.json` | Configuration, software versions, input/code hashes and run-specific provenance |

SOLO also saves `observation_selection.csv` with native observation masks and
`vertical_sections.npz` with section coordinates and reconstructed fields. Its
parameter CSV includes all attempted depths, including the stopping failure.
Drifter–satellite outputs identify `multi` and `drifter` sources separately.
Missing values use empty CSV fields or JSON `null`. `Omega_core`/`w_core` retain
inner estimates; `Omega`/`w` describe the refined outer fit where available.
`R2` measures velocity reconstruction; `r2_latte` measures the inner linear fit.

Rerunning replaces outputs. For a separate run, change `output_dir`; a directory
outside the standard application results parent uses a `manuscript_figures/`
folder alongside that run. Runs under `results/applications/` share the default
manuscript figure folder. The generated figure index links PNGs to summaries.

## Datasets and provenance

The bundled files are study extracts. All times below are UTC; analysis selections
are narrower than file coverage. The applications use the local origin
154.3°E, 37.9°S. See the [coordinate and unit conventions](../README.md#coordinates-units-and-interpretation)
and [ADCP timestamp correction](../README.md#ce23-adcp-timestamp-correction).

| Bundled file | Source and coverage |
| --- | --- |
| [`sadcp_data.mat`](../data/sadcp_data.mat) | RV *Investigator* voyage IN2023_V06; 2,174 records, 79 depth bins (27–807 m), approximately 15-minute sampling; corrected coverage 8–31 October 2023 |
| [`gridded_numerical_data.nc`](../data/gridded_numerical_data.nc) | Mercator/Copernicus daily currents, approximately 1/12° spacing, near-surface depth 0.494 m; 19 October 2023–11 February 2024 |
| [`satellite_data.nc`](../data/satellite_data.nc) | IMOS OceanCurrent surface geostrophic currents, 0.2° spacing; 18 October 2023 18:00–11 February 2024 06:00 |
| [`drifter_data.nc`](../data/drifter_data.nc) | NOAA Global Drifter Program six-hourly QC trajectories and velocities; two drifters, 2,403 records; 19 October 2023–16 August 2024 |

**Shipboard ADCP.** Voyage **IN2023_V06**, *Understanding eddy interactions and
their impacts in the East Australian Current System*, was led by Moninya Roughan
(UNSW), aboard RV *Investigator*, Sydney–Sydney, 9 October–2 November 2023.
The [CSIRO voyage page](https://www.csiro.au/en/about/facilities-collections/MNF/Voyages-schedules/Voyages/2023/October/IN2023_V06)
and [Data Trawler record](https://www.cmar.csiro.au/data/trawler/survey_details.cfm?survey=IN2023_V06)
provide source information and processing records. SOLO selects 96 observations
from 17 October 01:42:31 to 18 October 01:27:30; LATTE selects 433 observations
from 16 October 22:27:31 to 21 October 10:27:30.

**Numerical currents.** The file identifies Mercator Ocean International,
MOI GLO12, daily fields from Global Ocean Physics Analysis and Forecast, over
149–158°E and 42–34°S. Source: [Copernicus Marine product description](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description).
The exact original download request is not retained. Single-precision encoded
time offsets can differ from nominal midnight by minutes; this is separate
from the ADCP correction.

**Satellite currents.** Source: [IMOS OceanCurrent/AODN GSLA catalogue](https://thredds.aodn.org.au/thredds/catalog/IMOS/OceanCurrent/GSLA/catalog.html).
The extract records `product_type=NRT00` and covers the same geographic extent
as the numerical file. Its 121 native times are mostly daily but not uniformly
spaced; interpolation uses these native times.

> Data was sourced from the Integrated Marine Observing System (IMOS) - IMOS is a national collaborative research infrastructure, supported by the Australian Government.

**Drifters.** Source: [NOAA GDP six-hourly QC product](https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/drifter_6hour_qc.html);
citation: [Lumpkin and Centurioni (2019)](https://doi.org/10.25921/7ntx-z961).
The two platform IDs are `300534062034380` (WMO `5802097`) and `300534062784620`
(WMO `5501721`); the file records a 15-m drogue centre depth. The analysis uses
only the portion of these longer trajectories overlapping the study period.
