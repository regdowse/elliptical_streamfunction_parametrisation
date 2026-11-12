"""PNG-only manuscript exports and an index linking figures to their results."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
# Number, filename stem, result folder, generator.
FIGURES = (
    ('1', 'ESP_schematic', 'experiments/schematic', 'experiments/run.py schematic'),
    ('2', 'improved_MOCA', 'experiments/solo_idealised', 'experiments/run.py solo_idealised'),
    ('3', 'SOLO', 'experiments/solo_idealised', 'experiments/run.py solo_idealised'),
    ('4', 'SOLO_outercore_fits', 'experiments/solo_idealised', 'experiments/run.py solo_idealised'),
    ('5', 'SOLO_AR_err', 'experiments/solo_idealised', 'experiments/run.py solo_idealised'),
    ('6', 'solo_sadcp', 'applications/solo', 'applications/SOLO_transect_data.ipynb'),
    ('7', 'DOPPIO', 'experiments/doppio_idealised', 'experiments/run.py doppio_idealised'),
    ('8', 'DOPPIO_outercore_fits', 'experiments/doppio_idealised', 'experiments/run.py doppio_idealised'),
    ('9', 'doppio_num', 'applications/doppio', 'applications/DOPPIO_gridded_data.ipynb'),
    ('10', 'doppio_num_analysis', 'applications/doppio', 'applications/DOPPIO_gridded_data.ipynb'),
    ('11', 'LATTE', 'experiments/latte_idealised', 'experiments/run.py latte_idealised'),
    ('12', 'latte_sadcp_sat', 'applications/latte_adcp_satellite', 'applications/LATTE_app1_multiplatform_data.ipynb'),
    ('13', 'latte_drifter_sat', 'applications/latte_drifter_satellite', 'applications/LATTE_app2_multiplatform_data.ipynb'),
    ('14', 'latte_multi_analysis', 'applications/latte_drifter_satellite', 'applications/LATTE_app2_multiplatform_data.ipynb'),
    ('S1', 'noise_sampling', 'experiments/noise_sampling', 'experiments/run.py noise_sampling'),
    ('S2', 'background_flow', 'experiments/background_flow', 'experiments/run.py background_flow'),
    ('S3', 'sampling_geometry', 'experiments/noise_sampling', 'experiments/run.py noise_sampling'),
)


def figure_directory(run_directory):
    """Share canonical exports; keep custom/smoke runs isolated under their root."""
    run = Path(run_directory).resolve()
    if run.parent in (ROOT/'results/applications', ROOT/'results/experiments'):
        return ROOT/'results/manuscript_figures'
    return run.parent/'manuscript_figures'


def write_index(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    canonical = directory.resolve() == ROOT/'results/manuscript_figures'
    lines = ['# Manuscript figures', '',
             '14 main figures and S1–S3. PNG only. Rerunning replaces the corresponding PNG.',
             'Presence means a file exists; it does not certify a fresh or complete run.', '',
             '| Figure | PNG | Status | Results summary | Generator |',
             '| --- | --- | --- | --- | --- |']
    for number, stem, result, generator in FIGURES:
        path = directory/f'{stem}.png'
        group, name = result.split('/')
        run = ROOT/'results'/result if canonical else directory.parent/name
        summary = run/'manuscript_summary.md'
        summary_link = f'[Summary]({os.path.relpath(summary, directory)})' if summary.is_file() else 'Not generated here'
        lines.append(f'| {number} | [{path.name}]({path.name}) | {"Saved" if path.is_file() else "Missing"} | {summary_link} | `{generator}` |')
    (directory/'manuscript_figures.md').write_text('\n'.join(lines)+'\n')


def save_manuscript_figure(figure, stem, run_directory, *, dpi=300, **kwargs):
    """Save one registered figure, without fitting or duplicating numerical outputs."""
    if stem not in {row[1] for row in FIGURES}:
        raise ValueError(f'Unregistered manuscript figure: {stem}')
    directory = figure_directory(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'{stem}.png'
    figure.savefig(path, dpi=dpi, **kwargs)
    write_index(directory)
    return path
