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
