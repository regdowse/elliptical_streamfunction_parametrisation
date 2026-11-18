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
