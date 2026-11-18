# SWOT frontal-eddy study notes

## Agreed scope and setup

- Apply existing LATTE to one frontal cyclonic eddy using SWOT snapshots only.
- Independent worktree and branch: `swot-frontal-eddy`; leave `main` intact.
- Preserve baseline `functions.py` (inherited from `fadb4be`).
- Only `data/cycle_*/` files are study inputs. Other datasets belong to the
  original manuscript examples and have been moved to ignored `archive/data/`.
- Track `context/`; keep operative instructions in root `AGENTS.md`.
- Previous code and instructions are retained locally in ignored `archive/`.
  They remain available in the original checkout and Git history as well.

## Input inventory (filenames only; contents not yet inspected)

- cycle_004: pass 532, 2023-10-10; pass 545, 2023-10-11.
- cycle_005: pass 532, 2023-10-31; pass 545, 2023-11-01.
- Filenames identify SWOT L3 LR SSH Expert v3.0. Verify against metadata.

## Next steps

1. Inspect all four files' metadata, coverage, variables, units and quality flags.
2. Locate the target eddy and agree the scientific questions.
3. Establish velocity inputs, processing assumptions and local coordinates.
4. Plot and fit one snapshot, examine diagnostics and selection sensitivity,
   then apply the agreed workflow to the remaining snapshots.

## Open questions

- Target eddy location, approximate extent and initial visual identification.
- Data provider/download provenance and any preprocessing before delivery.
- Which height/velocity fields and quality criteria are appropriate?
- Treat each pass separately or combine nearby passes? Assess timing and coverage
  before deciding; cycle labels alone do not imply simultaneous observations.
- Desired comparisons and scientific interpretation across snapshots.

No data contents, fits or scientific results have been validated yet.

## First snapshot notebook and configuration

- Researcher selected 1 November 2023, pass 545, approximate centre
  151.75 E / 34.75 S, and absolute geostrophic flow only.
- `analysis.ipynb` reads `config.json`. Uses supplied `ugos_filtered` and
  `vgos_filtered`, quality code 0, no gap filling or extra smoothing.
- File metadata inspected: L3 v3.0; supplied velocities use spline-derived
  filtered SSH gradients plus mean dynamic currents. Local nearest-centre time
  is 03:16:53 UTC, rather than the initially approximate 03:27.
- Local Cartesian approximation follows the archived application, with origin
  at the researcher seed and 111000 m per degree (longitude cosine scaled).
- Provisional inner/outer radii: 15/50 km, identity initial Q. Sensitivity grid:
  inner 10/15/20 km and outer 40/50/60 km. These are trial choices, not validated.
- All notebook code cells executed in order in a fresh Python process using an
  Agg plotting backend; diagnostic figure inspected. Notebook outputs remain
  cleared. Baseline functions.py unchanged. This is not a Jupyter kernel test.
- Trial result: centre ~151.647 E / 34.658 S, axis ratio ~1.30, Rc ~49.9 km;
  inner R2 ~0.917, all-candidate outer vector R2 ~0.495, vector RMSE ~0.677 m/s.
  Cyclonic rotation. Strong eastern-current residuals and radius sensitivity
  mean this is preliminary, not a validated physical eddy reconstruction.
- NumPy emitted matrix-multiplication runtime warnings inside the unchanged
  tangential_velocity helper during the sensitivity run; reported fit outputs
  were finite. The cause has not been investigated; do not silently suppress.
- No confidence/prediction bands shown: spatial correlation is not modelled.
- Remaining decisions: review selection footprint, background frontal influence,
  filtering limitations, and fit adequacy. No archived non-SWOT data are used.

## Eight-application refactor

The researcher clarified that upstream (152.5 E, 33.0 S) and downstream
(151.75 E, 34.75 S) are distinct eddies. Approved display: eight labelled
observed/reconstructed map pairs, each followed by a rendered summary.
Processing and plotting now live in separate modules. Config supports eddy
settings and combination overrides; shared trial radii and existing plot and
sensitivity preferences were retained. No temporal tracking is assumed.

All eight applications were executed and rendered with the shared 15/50 km
trial settings. Only both 1 November cases produced full fits. Downstream:
10 October initial non-elliptic; 11 October refined non-elliptic; 31 October
insufficient initial samples (no valid vectors in configured region).
Upstream: 10 October initial non-elliptic; 11 October insufficient refined
samples; 31 October insufficient initial samples. These failures are displayed.
The upstream 1 November fit has vector R2 approximately -0.125, RMSE 0.911 m/s,
centre 152.357 E / 33.053 S and Rc 39.8 km; convergence is not a good flow match.
The downstream 1 November result agrees with the earlier single-case run.
Review coverage and seeds on each date before revising settings; do not tune
radii solely to obtain convergence. Representative success/failure figures were
visually inspected. Outputs remain cleared; functions.py unchanged.

## Researcher-provided origin refinements

Converted notebook comments with the existing local Cartesian approximation
(111000 m/degree and cosine of the ORIGINAL origin latitude):

| Application | Offset east/north (km) | New longitude | New latitude |
| --- | --- | --- | --- |
| downstream 2023-10-10 | 20, 10 | 151.96929149 | -34.65990991 |
| downstream 2023-10-11 | 10, 10 | 151.85964574 | -34.65990991 |
| upstream 2023-10-11 | 50, -10 | 153.03710058 | -33.09009009 |

These are per-combination config overrides, with local seed (0,0). Other five
origins were preserved. Recentring the origin also recentres the configured
map/selection extent and uses the new latitude for the longitude scale.
This is a researcher-specified starting point, not an automatic search.
Notebook comments now record the conversion; outputs cleared to avoid stale plots.

Summaries now show starting origin, prescribed inner/outer elliptical-radius
thresholds and initial/refined sample counts, including when a fit fails.
Plots show the actual refined inner selection boundary (preliminary centre/Q),
the final outer boundary and, for full fits, an orange rho=Rc/sqrt(2) ellipse.
Speed varies around that ellipse; it is not a constant-speed contour.
No fitting mathematics changed.

Checked conversion round-trips and executed/rendered all eight applications.
Both refined downstream October seeds now give optimized fits. Upstream
11 October remains initially non-elliptic; other fit statuses unchanged.
Numerical warnings again occurred in the baseline tangential_velocity helper;
no warnings were suppressed or baseline edits made. Representative plot checked.

## Fixed refined starting origins

Successful full-fit centres were recomputed using the frozen prior config and saved once as
new per-combination origins (not an iterative search). The explicit upstream 11 October
(-10, 0) km instruction takes precedence, relative to its original eddy origin.
Other failed-case origins are unchanged. Local seeds are (0,0). Origins:

- downstream 2023-10-10: 151.93620719, -34.62235658.
- downstream 2023-10-11: 151.88240068, -34.60006933.
- downstream 2023-11-01: 151.64714352, -34.65847701.
- upstream 2023-11-01: 152.35675783, -33.05312998.
- upstream 2023-10-11: 152.39257988, -33.00000000.

Plot settings now include show_selection_boundaries, show_starting_centre and
starting_centre_color (magenta). Orange speed ellipse is independent of both flags.
Axes shortened to East (km) / North (km). Old notebook outputs and obsolete seed
comments cleared; config.json remains the source of starting origins.

Verification after fixed-seed update: all eight cases executed and rendered, all
four display-flag combinations checked, and representative figure inspected.
Four full fits remain available (three downstream, one upstream). Updated
upstream 11 October fails at refined insufficient samples. Baseline numerical
warnings remain visible. functions.py unchanged.

## Upstream 11 October additional refinement

Applied notebook comment (+40, 0) km relative to the previously plotted origin
(152.39257988, -33.00000000), giving (152.82226035, -33.00000000).
Saved as fixed config override, local seed (0,0); other applications unchanged.
Affected notebook output cleared. Executed and rendered this case: inner status
initial_non_elliptic_flow, radial status not_attempted.

## Further upstream 11 October starting-point trial

Applied (+20, -20) km relative to previous plotted origin
(152.82226035, -33.00000000). New fixed origin:
(153.03710058, -33.18018018). Other settings/applications unchanged.
Case executed and plotted: inner initial_insufficient_samples,
radial not_attempted. Affected notebook outputs cleared.

## Explicit per-application radii

Researcher chose manual per-application radii rather than automatic search.
All eight overrides now explicitly specify inner and outer radii. Preserved
previous effective values except upstream 11 October, now a trial 25/50 km
pair. Origins and plot preferences unchanged. Validated all eight resolved
radius pairs and executed/rendered the changed case: inner status
refined_insufficient_samples, radial status not_attempted.
The affected notebook output was cleared. No fitting mathematics changed.

## Optional single-stage inner fit

Researcher approved disabling the second selection/refit stage for upstream
11 October. Added latte.refine_inner (default true), with false only for that
application. Preserved its current user-selected 20 km inner / 50 km outer
radii and all origins. Single-stage mode calls existing latte on the initial
seed/Q/radius mask. Outer fitting still uses the fitted centre/Q and the same
outer radius. It classifies actual initial-fit samples as inner data; no
refined samples are claimed. Summary and boundary geometry identify the mode.

All eight cases executed and rendered; checked inner results against direct
latte in single-stage mode and direct refine_latte for all seven defaults.
Representative single-stage plot inspected. functions.py unchanged. Existing
baseline matrix-multiplication warnings remain visible on other cases.

Upstream 11 October trial: both fits optimized; initial samples 18; centre
(7.87, 26.62) km, approximately 153.1218 E / 32.9404 S; Rc 32.60 km;
vector R2 0.451, vector RMSE 0.357 m/s. Rotation is ANTICYCLONIC, an important
mismatch with the intended cyclonic target. This is not an accepted physical
identification. No rotation sign or centre constraints were imposed.

## Starting-position diagnostic scan (upstream 11 October)

Researcher requested testing whether seed movement improves this case. Tested
169 seed offsets (-30 to +30 km east/north in 5 km steps) relative to the current
fixed origin. Kept the current USER settings: single-stage inner fit, 20 km
inner radius, 60 km outer radius, identity initial Q. Held coordinate origin,
data domain and quality mask fixed; no saved config changes or seed promotion.
Evaluated all full reconstructions on the same 948 valid vectors within 60 km
of the original plot origin. Target comparison: plot coordinates (10,10) km.

20 full fits, including five cyclonic. Baseline (0,0): fitted (7.87,26.62) km,
anticyclonic, common-area R2 0.549 and vector RMSE 0.258 m/s. Best common-area
trial (-5,15): fitted (17.85,10.47) km, anticyclonic, 81 initial samples,
R2 0.765 and RMSE 0.186 m/s. Closest-to-target trial (0,25): fitted (8.58,2.59)
km, anticyclonic, 55 initial samples, R2 0.723 and RMSE 0.202 m/s.
All five cyclonic fits lie 72–95 km from the target and have negative common-area
R2. Moving the seed improves the empirical velocity match but does not provide
a convincing cyclonic identification near the expected centre in this grid.

Complete candidate table, frozen settings, diagnostic figures and rerunnable
script are local ignored artifacts in results/starting_position_test/. Ten
numerical warnings were captured across the scan and recorded per candidate
in the CSV, not silently discarded. This is a seed-sensitivity exploration,
not physical validation or proof that another solution cannot exist.

## Corrected upstream 11 October target identification

Researcher identified that the previous trials targeted the wrong feature.
The intended small cyclonic eddy is at (-10,+60) km on the previous grid,
whose origin was (153.03710058, -33.18018018).
Converted with that grid's coordinate scale to new fixed origin
(152.92946011, -32.63963964), local starting seed (0,0). Previous anticyclonic
trial fits are not identifications of this corrected target. Preserved current
inner/outer radii (10000, 20000 m)
and refinement flag; researcher will reduce radii for the smaller eddy.
Other applications unchanged; affected notebook output cleared.
