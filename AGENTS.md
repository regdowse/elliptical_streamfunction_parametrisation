# SWOT frontal cyclonic eddy — project instructions

## Scope and boundaries

This is a standalone case study applying existing LATTE/ESP methods to a small
number of SWOT snapshots of two distinct frontal cyclonic eddies (upstream and downstream). It is not an extension
of the manuscript reproduction project or a general ESP release.

Work in this worktree on `swot-frontal-eddy`. The sibling original checkout is
on `main`; leave it intact. Do not merge to main, create release tags, publish
to Zenodo, rewrite history, force-push, or delete branches without explicit
instruction. Commit only when asked. Before destructive Git operations check
`git status`, `git branch --show-current`, and `git worktree list`.

## Project files

- `functions.py`: baseline ESP implementation inherited from commit `fadb4be`.
  Reuse its LATTE functions; preserve it unchanged unless a demonstrated need
  is discussed with the researcher. Check signatures and conventions in code.
- `data/cycle_*/`: the only observational inputs in scope for this study.
  Raw data remain local and ignored. Record product provenance and input paths.
- `context/`: tracked scientific notes, decisions, and open questions.
- `archive/`: ignored local reference material from the original project,
  including its applications, experiments, helper modules, README, example
  datasets, and initial instructions. It must not become a runtime dependency.
- `analysis.ipynb`: eight independent applications (two eddies × four passes),
  each showing observed/reconstructed flow followed by a Markdown summary.
- `swot_processing.py`: configuration, data selection, fitting and summaries.
- `figures.py`: all plotting and display; optional radial diagnostic.
- `config.json`: shared defaults, eddy origins/settings, snapshot files and
  per-combination overrides. No fit state propagates between applications.

Do not use archived drifter, ADCP, model, or conventional satellite datasets as
observations for this case study. Consult archived LATTE application 2 for the
direct fitting sequence, not its old scientific settings or tracking assumptions.
Keep the project small; avoid unnecessary modules and abstractions.

## Scientific workflow

1. Inspect SWOT metadata, dimensions, coordinates, times, variables, units,
   missing values, masks, and quality flags before choosing inputs.
2. Establish the target location and legitimate velocity inputs. Do not assume
   SSH provides horizontal velocity directly. If using provided or derived
   geostrophic velocity, document its definition, approximations, coordinate
   treatment, filtering and limitations near fronts and ageostrophic motion.
3. Agree a reproducible spatial selection and inspect one snapshot first.
   Document any smoothing, interpolation, subsampling or masking.
4. Reuse LATTE with local east/north positions and consistent units; verify the
   actual implementation. Separate inner selection from outer radial fitting.
5. Compare observations and reconstruction, residuals, spatial sampling and
   sensitivity to selection and preprocessing before extending to other files.
6. Record uncertainties and failed fits. Convergence alone is not physical
   validation. Never invent or silently replace missing observations.

Explain proposed changes to fitting mathematics, coordinate transforms or
velocity derivation and their consequences before implementing them. Do not
assume product fields, study domain, scientific hypotheses or fit settings.
Keep notebooks reproducible from a fresh kernel. Use small checks for nontrivial
processing. Prefer NumPy, SciPy, pandas, xarray, Matplotlib and Jupyter.

## Collaboration

Make focused changes and report what changed, why, and what was verified.
Keep secrets, raw data, caches, environments and large generated products out of
Git. Track context notes and meaningful checks. Update `context/study_notes.md`
with agreed decisions; keep operative agent instructions in this root file.
The initial branch cleanup was approved; further broad restructuring requires
agreement. Trial fits exist; their convergence is not physical validation.
