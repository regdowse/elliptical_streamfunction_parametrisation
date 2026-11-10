# Idealised validation experiments

Scripts run controlled tests using the general estimators in the root
[`functions.py`](../functions.py). Real-data demonstrations remain in
[`applications/`](../applications/README.md).

## Run all main-paper idealised figures

```bash
python experiments/run.py idealised
python experiments/run.py idealised --plot-only
```

This runs SOLO, DOPPIO, LATTE and the ESP schematic with their own configurations.
Together they export the eight main-paper idealised figures as PDF and 300-dpi
PNG, per-group results and **`results/experiments/idealised_manuscript_summary.md`**.
That combined file contains all four manuscript summaries for transferring the
current results into the paper. It is assembled only after every group completes.
The schematic is illustrative and performs no parameter estimation.


Figure numbers follow the manuscript, including the intervening real-data figures:

| Manuscript figure | Idealised output filename |
| --- | --- |
| 1 | `ESP_schematic.png` |
| 2 | `Improved_MOCA.png` |
| 3 | `SOLO.png` |
| 4 | `SOLO_outercore_fits.png` |
| 5 | `SOLO_AR_Err.png` |
| 7 | `DOPPIO.png` |
| 8 | `DOPPIO_outercore_fits.png` |
| 11 | `LATTE.png` |

Use `--output-root results/experiments/my_run` for a separate run. A custom
`--config` applies to one named experiment; the `idealised` group uses each
experiment's default config. Generated results remain gitignored.

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
| `SOLO` | Six transects varying proximity and full length; observations and inner-core linearisations |
| `SOLO_outercore_fits` | Normalised tangential observations and stored outer fits for the same six cases |
| `Improved_MOCA` | Conceptual modification of MOCA using ideal inner-core SOLO inputs |
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

The MOCA figure is a **conceptual illustration**, matching the original layout.
SOLO receives ideal inner-core constant/linear velocities; MOCA receives the
Gaussian observations. Green guides show the local observed U maximum and
finite-difference V slope. Its near-perfect SOLO centre demonstrates the ideal
model, rather than recovery from the same observations used by MOCA. This
distinction is recorded in the report and must be clear in the manuscript.

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
overwrite their output directory.

## DOPPIO validation and paired LATTE comparison

```bash
python experiments/run.py doppio_idealised
python experiments/run.py doppio_idealised --plot-only
```

Configuration: [`configs/doppio_idealised.json`](configs/doppio_idealised.json).
Outputs: `results/experiments/doppio_idealised/`.

- `DOPPIO.pdf/png`: proximity cases on the truth field; all six sets of observed
  velocities and their inner-core linearisations. Solid lines are observations;
  faint lines are tangents at the fitted roots, matching the original layout.
- `DOPPIO_outercore_fits.pdf/png`: the same six stored outer fits, with observed
  points, faded extrapolation, predicted peak and true normalised peak radius.
- `parameters.csv` and `manuscript_table.csv`: six DOPPIO fits and six LATTE fits
  to matching unique observations, with inner/final rates kept separate.
- `comparison.csv` and `comparison_table.csv`: paired absolute errors,
  improvement percentages, thresholds, every exclusion reason and metric means.

Both methods use Gaussian truth with Rc=85 km, psi0=250,000 m²/s and
Q proportional to `[[1,0.4],[0.4,1]]`, normalised to determinant one.
EP1–EP3 have cross intersections (10,10), (−25,25), (−30,−30) km and half
length 30 km. EL1–EL3 use (10,10) km with half lengths 20/30/40 km. Each
arm is sampled every 10 km, with both endpoints included. DOPPIO receives
complete horizontal/vertical arms. LATTE and both outer fits receive their
unique union, counting the shared intersection once. All observations are used
for inner estimation; each method's own estimated centre and Q feed its outer fit.

Improvement is `100*(LATTE error - DOPPIO error)/LATTE error`; positive favours
DOPPIO. Entries are unavailable for failed paired fits, zero reference error,
or differences below configured tolerances: 1 km (centre/Rc), 1e−7 s⁻¹ (rates),
0.005 (aspect ratio). These thresholds affect the comparison table only; all
attempts and raw errors remain saved. Means use full precision before rounding.
Large valid errors are retained. Scientific settings, not the manuscript's old
numerical values, determine the results.

## LATTE validation

```bash
python experiments/run.py latte_idealised
python experiments/run.py latte_idealised --plot-only
```

Configuration: [`configs/latte_idealised.json`](configs/latte_idealised.json).
Outputs: `results/experiments/latte_idealised/`, including `figures/LATTE.pdf/png`,
parameters, errors, manuscript table and summary. Four examples use the same
elliptical truth as DOPPIO:

| Example | Sampling |
| --- | --- |
| Ex1 | Sinusoidal ship transect, 15 points |
| Ex2 | Exaggerated Euler drifter path, 30 post-update positions |
| Ex3 | Two oblique transects, 10 points each |
| Ex4 | 5×5 grid at −50, −30, −10, 10, 30 km in each coordinate |

Velocities use bilinear interpolation from a 1-km truth grid, preserving the
original off-grid sampling. The drifter advances by **6000 s (100 min)** for
30 steps: 50 hours total. This is the SI equivalent of the original update
using km positions, m/s velocities and `dt=6`; it is an exaggerated illustrative
trajectory rather than a high-accuracy particle integrator. Leaving the sampling
grid raises an error rather than silently clipping or inventing velocities.

All points enter the inner estimator. Outer panels draw their saved fitted
parameters and observations, without refitting for plotting. Reported vorticity
uses final Omega times trace(Q); preliminary inner rates remain in the CSV.

## ESP schematic

```bash
python experiments/run.py schematic
python experiments/run.py schematic --plot-only
```

Configuration: [`configs/schematic.json`](configs/schematic.json).
Outputs: `results/experiments/schematic/figures/ESP_schematic.pdf/png`, sampling,
summary and metadata. The original circular sampling design is preserved:
40 core draws, 60 outer draws, 30-km core circle, 120-km outer selection and
175-km sampling extent, snapped to a 1-km grid. A fixed configurable seed makes
this previously unseeded illustration repeatable. Known truth defines the
illustrative masks; no estimator is run or fitted recovery accuracy claimed.

## Saved results and redrawing

All new groups save `sampling.csv`, `manuscript_summary.md`, `run_metadata.json`
and `plot_metadata.json`. Fit groups also save unrounded parameters, diagnostics
and curves in `curves.json`, and the display field in `field.npz`. Run metadata
records the effective configuration, source/data hashes, Git state and versions.
`--plot-only` verifies saved hashes, uses saved science and current presentation
styles, and never fits or evaluates a velocity model. Numerical files, reports
and original run metadata remain unchanged. Changing a scientific setting needs
a full run; changing the sampling grid can change the experiment.

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

`all` selects the two supplementary sensitivity experiments. Use `idealised`
for all four main-paper groups; the two selections are independent.

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
- [`idealised.py`](idealised.py): DOPPIO/LATTE sampling, shared estimators,
  paired error scoring, schematic sampling and results/provenance.
- [`idealised_plotting.py`](idealised_plotting.py): the remaining four figures.

Self-tests verify sampling counts/symmetry, exact HIGH-coordinate reuse,
intersection handling, streamfunction derivative signs, Uref, final-Omega error
definitions and valid clean baselines. Trial checks verify the complete expected
condition/replicate counts and explicit statuses. Numerical acceptance alone
does not establish accuracy or general robustness outside this experimental design.
