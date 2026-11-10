# Idealised validation experiments

Scripts run controlled tests using the general estimators in the root
[`functions.py`](../functions.py). Real-data demonstrations remain in
[`applications/`](../applications/README.md).

## SOLO validation

```bash
python experiments/run.py solo_idealised
python experiments/run.py solo_idealised --plot-only
```

Settings are in [`configs/solo_idealised.json`](configs/solo_idealised.json).
The run produces four PDF/PNG figures under
`results/experiments/solo_idealised/figures/`:

| Figure | Test |
| --- | --- |
| `SOLO` | Six transects varying proximity and full length; observations and fitted polynomials |
| `SOLO_outercore_fits` | Normalised tangential observations and stored outer fits for the same six cases |
| `Improved_MOCA` | SOLO versus MOCA using identical Gaussian observations |
| `SOLO_AR_Err` | Centre error over 17 determinant-one deformations, aspect ratios 1–3 |

Default truth: circular Gaussian, Rc=85 km, psi0=250,000 m²/s, centre (0,0).
P1–P3 are at y=20/50/80 km, each **140 km long**, sampled every 10 km.
L1–L3 are at y=20 km with full lengths 80/140/200 km. All points are used in
inner fits by default (`solo_core_radius_m: null`). The comparison uses 221
points from x=−110 to 110 km at y=44 km. Deformation uses the P1 transect,
with Q proportional to `[[1,q12],[q12,1]]`, q12=0…0.8.

The run saves `parameters.csv` (including errors and fit status), `sampling.csv`,
`comparison.csv`, `deformation.csv`, a manuscript table and
**`manuscript_summary.md`**. The summary contains the truth, case definitions,
parameter estimates, numerical claims and manuscript discrepancies. Provide it
alongside the manuscript when updating its text.

The MOCA comparison fits both methods to the same observations. Green lines
are SOLO's fitted polynomial tangents. Exact inner-core velocities and numerical
slopes are not substituted for fitted quantities. This correction changes the
comparison results; existing manuscript values must be checked.

`curves.json` stores observations, coefficients and fit diagnostics; `field.npz`
stores the display velocity field. `run_metadata.json` records the configuration,
data/source hashes, Git state and software versions. Plot-only mode verifies
saved data and redraws all four figures without fitting or evaluating a velocity
model. Scientific settings and the display grid remain frozen from the run;
current colors, limits and export styles are used. Numerical files, the manuscript
report and original run metadata remain unchanged; fresh rendering provenance
is written to `plot_metadata.json`.

Use `--config path/to/config.json` to select another configuration and
`--output-root results/experiments/my_run` to keep a separate run. Full runs
overwrite their output directory. DOPPIO/LATTE idealised validation and the
method schematic are not yet implemented in these scripts.

## Supplementary sensitivity experiments

Scripts reproduce the **noise/sampling** and **uniform-background-flow** tests
using the shared root [`functions.py`](../functions.py). No notebooks or external
data are required.

The experimental design follows the [original supplementary test suite](https://github.com/regdowse/UNSW-PhD/tree/8cde2c20503be988f298cd258f413932742fe099/esp/os_paper_additional_tests).
Sampling geometry, random streams, noise levels and background directions are
preserved. The fitting code now uses the corrected shared methods directly in
SI units; old implementation-specific scaling, extracted inversions and radius
fallbacks are not carried over. Results should be checked against the manuscript,
not assumed identical to previous runs.

## Run the experiments

Install the dependencies in the [root README](../README.md#installation), then
run from the repository root:

```bash
# Full runs: 100 trials per nonzero noise condition, followed by background flow.
python experiments/run.py all

# Or run either experiment separately.
python experiments/run.py noise_sampling
python experiments/run.py background_flow
```

`all` currently selects the two supplementary sensitivity experiments; run
`solo_idealised` separately.

The default design produces **2,709 noise trial records** (including nine clean
baselines) and **23 background-flow records**. Every attempted trial is retained,
including numerical failures. Acceptance means a valid full fit, not low error;
large valid errors are not trimmed.

For a quick execution check, use a separate output root:

```bash
python experiments/run.py all --replicates 2 --output-root results/experiments/smoke
```

This overrides the noise replicate count and records that override. Smoke-test
results are not substitutes for the full experiment.

## Configuration and design

All settings are in [`configs/sensitivity.json`](configs/sensitivity.json).
Both tests share this configuration so their HIGH-density coordinates match.
Paths in `output_directory` and `--output-root` are repository-relative; absolute
paths are also supported. The script itself can be launched from another directory.

| Setting | Default |
| --- | --- |
| Truth | Circular stationary Gaussian; centre (0,0), Rc=85 km, psi0=250,000 m²/s, Q=I |
| Reference speed | Maximum true vector speed, Uref ≈2.522835 m/s |
| Sampling extent | ±1.2 Rc on transects; radius 1.2 Rc for LATTE |
| SOLO densities | 11 / 31 / 101 unique observations |
| DOPPIO densities | 11 / 31 / 101 per transect; 21 / 61 / 201 unique observations |
| LATTE densities | 21 / 61 / 201 unique observations; nested rotational/reflection-symmetric clouds |
| Noise | Independent component SD: 0, 2.5, 5, 10% of Uref; 100 trials per nonzero condition |
| Background | HIGH density; speeds 0, 5, 10, 20, 30, 50% of Uref; no noise |
| Directions | SOLO eastward and northward; DOPPIO/LATTE northward only |
| Seed | 20261027, with separate coordinate/noise streams |

Velocity noise is independent at each unique location; it is not symmetrised.
DOPPIO reuses the same noisy vector at its shared transect intersection.
Standard-normal draws are paired across noise amplitudes. Geometry is immutable
within a run and is reused exactly across both experiments.

**Controlled-fit selection differs from the real-data applications.** The
historical supplementary tests fit their synthetic observations without adding
the observational 30/35-km core-selection workflow. The default
`solo_core_radius_m: null` explicitly keeps all SOLO transect points. DOPPIO and
LATTE also receive all synthetic observations. The shared outer fitter applies
its own finite-value/sign filters. `outer_Rc_max_m: null` uses the corrected
unlimited-radius default. Changing these settings changes the experiment.

Coordinates and fitting parameters use SI units throughout; only displayed
positions are converted to km. Error scoring uses the **final outer-fit Omega**,
not the preliminary inner estimate. SOLO's aspect ratio is fixed, so its error
is marked as not estimated.

The config also contains plotting styles, output formats, percentiles and
manuscript figure/table labels. Reference transect hashes check the original
design; intentional changes to truth radius, extent or transect counts require
updating those hashes or disabling `verify_reference_transects`. LATTE group
counts must remain consistent with the configured sampling counts.

Use a new `--output-root` when changing truth/sampling, so saved geometry and
results from different designs are not mixed. Existing result files in a run's
output folders are overwritten when the experiment is rerun.

## Figures and manuscript outputs

| Figure | Output | Purpose |
| --- | --- | --- |
| **S1** | `noise_sampling/figures/noise_sampling.pdf` and `.png` | Median errors and 10th–90th Monte Carlo percentiles |
| **S2** | `background_flow/figures/background_flow.pdf` and `.png` | Deterministic background-flow errors; no uncertainty bands |
| **S3** | `noise_sampling/figures/sampling_geometry.pdf` and `.png` | Actual LOW/MEDIUM/HIGH locations and unique inner/outer counts |

Figures are PDF plus 300-dpi PNG by default. S3's HIGH configurations also supply
the background-flow experiment; no separate background-geometry figure is made.
Figure numbers are configurable labels, not dependencies of the calculations.

```text
results/experiments/
├── sampling_coordinates_<signature>.json
├── noise_sampling/
│   ├── manuscript_summary.md
│   ├── manuscript_table.csv
│   ├── trials.csv
│   ├── summary.csv
│   ├── sampling.csv
│   ├── run_metadata.json
│   ├── plot_metadata.json
│   └── figures/
└── background_flow/
    └── same result files and its figures/
```

**Give `manuscript_summary.md` to ChatGPT** with the relevant manuscript section.
It includes the experiment definition, manuscript-rounded table, condition-median
ranges, acceptance counts, baseline errors, sampling counts and configuration.
Noise table centre errors are percentages of true Rc; figures display centre
error in km. Other plotted/table errors are absolute relative percentages.

- `trials.csv` stores every attempt, parameters, warnings and status, without
  accuracy-based rejection. Failed parameters remain unavailable.
- `summary.csv` contains unrounded accepted-fit statistics and counts. Noise
  bands are Monte Carlo spread, not confidence intervals. Background results
  are deterministic and have no invented uncertainty bands.
- `run_metadata.json` records effective settings and command overrides, source
  config hash, exact coordinates, source/helper hashes, Git state, versions,
  definitions, self-tests and CSV hashes. JSON uses null for unavailable values.
- `plot_metadata.json` records each rendering's settings, plotting-code hash,
  original run-metadata hash and figure hashes.

## Redraw figures without rerunning fits

```bash
python experiments/run.py all --plot-only

# A named run can be redrawn separately.
python experiments/run.py noise_sampling --plot-only --output-root results/experiments/smoke
```

Plot-only mode verifies saved CSV hashes, then uses the saved scientific settings,
sampling coordinates and trial results. It reads current plotting styles and manuscript labels
from the config, regenerates figures/reports, and leaves trial CSVs, numerical
summaries and original run metadata unchanged. It does not fit models or run
fitting self-tests. Changing statistical percentiles requires a full run.

## Code layout and checks

- [`run.py`](run.py): command-line orchestration and rendering.
- [`sensitivity.py`](sensitivity.py): deterministic geometry, perturbations,
  shared-method calls, scoring, trial validation and summaries.
- [`plotting.py`](plotting.py): results and sampling figures.
- [`outputs.py`](outputs.py): manuscript reports, tables and provenance.
- [`solo.py`](solo.py): SOLO validation design, shared-method calls and saved results.
- [`solo_plotting.py`](solo_plotting.py): the four SOLO figures from saved fits.

Self-tests verify sampling counts/symmetry, exact HIGH-coordinate reuse,
intersection handling, streamfunction derivative signs, Uref, final-Omega error
definitions and valid clean baselines. Trial checks verify the complete expected
condition/replicate counts and explicit statuses. Numerical acceptance alone
does not establish accuracy or general robustness outside this experimental design.
