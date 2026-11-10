# Real-data applications

Four notebook demonstrations use the shared [`functions.py`](../functions.py)
and [`reporting.py`](../reporting.py) at the repository root. Their existing
analysis steps, manuscript plots and diagnostic plots are retained.

## Choose a demonstration

| Notebook | Configuration | Default results directory |
| --- | --- | --- |
| [SOLO ADCP transect](SOLO_transect_data.ipynb) | [solo.json](configs/solo.json) | `results/applications/solo/` |
| [DOPPIO model tracking](DOPPIO_gridded_data.ipynb) | [doppio.json](configs/doppio.json) | `results/applications/doppio/` |
| [LATTE ADCP–satellite](LATTE_app1_multiplatform_data.ipynb) | [latte_adcp_satellite.json](configs/latte_adcp_satellite.json) | `results/applications/latte_adcp_satellite/` |
| [LATTE drifter–satellite](LATTE_app2_multiplatform_data.ipynb) | [latte_drifter_satellite.json](configs/latte_drifter_satellite.json) | `results/applications/latte_drifter_satellite/` |

## Run a notebook

1. Install the dependencies in the [root README](../README.md#installation).
2. Edit the corresponding JSON configuration in `applications/configs/`.
3. Open the notebook from the repository root or `applications/`, restart the
   kernel, and run all cells in order.
4. Collect figures from the notebook and the exported files from its results folder.

The first cell locates the repository without changing the working directory.
`CONFIG_PATH` selects the JSON file. To use a separate configuration, edit that
assignment (or set it before running the first cell). Input and output paths
inside JSON are **relative to the repository root**, regardless of where the
notebook was launched; absolute paths are also accepted. Configuration files
are shared with the repository; generated results are gitignored.

### Configuration settings

- `inputs`: paths to the bundled MAT/NetCDF files.
- `coordinates`: longitude/latitude of the local Cartesian origin.
- `analysis`: depth, UTC windows, velocity filters, initial centre/geometry,
  selection radii, tracking displacement limit and drifter aggregation settings,
  as applicable to the demonstration. Names specify units (`_m`, `_hours`, etc.).
- `outer_fit`: iteration budget and optional fitted-radius bound. `Rc_max_m: null`
  retains the current unlimited fitted-radius default; it does not change the
  observation-selection radius.
- `plot`: snapshot date, map limits in km, quiver settings and diagnostic options.
  `diagnostics: true` preserves the existing intermediate plots; set it to false
  to suppress diagnostic fits/scatters while retaining the main results figures.
- `reconstruction`: SOLO's reconstructed grid extent/resolution.
- `output_dir`: where this run's three results files are saved.

DOPPIO and drifter–satellite snapshots are selected by configured UTC dates.
`snapshot_tolerance_hours` permits small differences such as model time-encoding
precision; a missing nearby fit raises an error instead of choosing a distant
date. The report identifies the **actual** plotted time. Default configurations
preserve the previous notebook snapshots (22 October for DOPPIO; 29 October for
LATTE drifter–satellite), not necessarily dates in older manuscript captions.

The CE23 MAT decoder remains dataset-specific. Changing a file path alone does
not adapt a loader to a different data format or time convention.

## Manuscript results exports

The final cell keeps the printed summary and writes:

```text
results/applications/<demonstration>/
├── manuscript_summary.md
├── parameters.csv
└── run_metadata.json
```

| File | Content |
| --- | --- |
| `manuscript_summary.md` | Current settings, UTC intervals, counts, rounded and detailed results, parameter ranges, snapshot dates and fit diagnostics |
| `parameters.csv` | Full-precision fitted parameters in SI units; Q expanded into `q11`, `q12`, `q21`, `q22`; timestamps explicitly in UTC |
| `run_metadata.json` | Loaded configuration, package versions, code/data/configuration hashes, Git revision and working-tree state, counts, fit diagnostics and run-specific selections |

For drifter–satellite LATTE, the parameter CSV includes both `multi` and `drifter`
sources. Drifter-only outer parameters remain unavailable. In JSON, unavailable
numeric values are `null`; CSV uses empty fields. `R2` is velocity-reconstruction
R², while `r2_latte` is the linear inner-core fit. Tangential-fit R² is recorded
separately in the summary and metadata.

Give `manuscript_summary.md` to ChatGPT with the relevant manuscript section;
include the CSV/JSON when numerical precision or provenance is needed. These
reports describe the current computation and do not assume that old manuscript
values still agree.

**Rerunning replaces the three files in the configured output directory.** Set a
new `output_dir` to keep a separate run. Restart and run all cells after changing
configuration; export refuses a config edited after it was loaded. Most figures
remain interactive notebook outputs. The final time-series plot in
`LATTE_app2_multiplatform_data.ipynb` also exports manuscript Figure 14 as
`figures/latte_multi_analysis.pdf` and `.png` under its configured output directory.
It compares multi-platform and drifter-only LATTE, with Rc and Gaussian velocity
R² shown only for the multi-platform reconstruction.

## Datasets and provenance

The bundled files are study extracts, not the complete source archives. File
coverage below was checked against the arrays and embedded metadata. **All
observation times are UTC.** Notebook selections are narrower than file coverage.

| Bundled file | Provider / data type | Bundled coverage | Used by |
| --- | --- | --- | --- |
| [`sadcp_data.mat`](../data/sadcp_data.mat) | RV *Investigator*, IN2023_V06; shipboard ADCP | 8 October–31 October 2023 after the MAT clock correction; 2,174 records and 79 depth levels | SOLO; LATTE application 1 |
| [`gridded_numerical_data.nc`](../data/gridded_numerical_data.nc) | Mercator Ocean International / Copernicus Marine; daily model currents | 19 October 2023–11 February 2024; 149–158°E, 42–34°S | DOPPIO |
| [`satellite_data.nc`](../data/satellite_data.nc) | IMOS OceanCurrent / AODN; gridded surface geostrophic currents | 18 October 2023 18:00–11 February 2024 06:00; 149–158°E, 42–34°S | Both LATTE applications |
| [`drifter_data.nc`](../data/drifter_data.nc) | NOAA Global Drifter Program; quality-controlled six-hourly trajectories and velocities | 19 October 2023–16 August 2024; two drifters, 2,403 records | LATTE application 2 |

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
See [the timestamp correction](../README.md#ce23-adcp-timestamp-correction) below.

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

