"""Persist supplementary trials, manuscript tables, provenance and figure renders."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from reporting import _json_safe
from experiments.sensitivity import Config, ROOT, METRICS, summarize, sampling_summary, interpretation, _bank_from_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def markdown_table(frame):
    def text(v):
        if isinstance(v, (float, np.floating)):
            return "—" if not np.isfinite(v) else f"{v:.2f}"
        return str(v).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                       "| " + " | ".join("---" for _ in frame.columns) + " |",
                       *["| " + " | ".join(text(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]])


def manuscript_table(results):
    summary = summarize(results)
    noise = results.attrs["experiment"] == "noise_sampling"
    if noise:
        summary = summary.loc[summary.noise_fraction > 0].copy()
        summary["Noise (% Uref)"] = 100*summary.noise_fraction
        labels = {"method": "Method", "density": "Density"}
        keys = ["Method", "Density", "Noise (% Uref)"]
    else:
        summary = summary.copy()
        summary["Background (% Uref)"] = 100*summary.background_fraction
        labels = {"method": "Method", "background_direction": "Direction"}
        keys = ["Method", "Direction", "Background (% Uref)"]
    summary = summary.rename(columns={**labels, "centre_error_pct_rc_median": "Centre (% Rc)",
        "Omega_error_pct_median": "Omega (%)", "alpha_error_pct_median": "Aspect ratio (%)", "rc_error_pct_median": "Rc (%)"})
    summary["Accepted/attempted"] = summary.accepted.astype(int).astype(str) + "/" + summary.attempted.astype(int).astype(str)
    return summary[keys + ["Centre (% Rc)", "Omega (%)", "Aspect ratio (%)", "Rc (%)", "Accepted/attempted"]]


def write_report(results, context, directory):
    c, truth = context.config, context.truth
    noise = results.attrs["experiment"] == "noise_sampling"
    table = manuscript_table(results)
    name = "Noise and sampling sensitivity" if noise else "Uniform-background-flow sensitivity"
    figure = c.values["manuscript"]["noise_sampling_figure" if noise else "background_flow_figure"]
    table_id = c.values["manuscript"]["noise_table" if noise else "background_table"]
    lines = [f"# {name} — manuscript results", "", f"Figure **{figure}**; **{table_id}**.", "",
        "Results use the shared ESP code recorded in run_metadata.json. Redrawing figures does not refit them. Do not assume agreement with older manuscript values.", "",
        "## Experiment definition", "",
        f"- Circular Gaussian truth: centre (0, 0) m; Rc={truth['rc_m']/1000:g} km; psi0={truth['psi0_m2_s']:g} m²/s; Q=I.",
        f"- True Omega={truth['Omega_s_1']:.12g} s⁻¹; centre vorticity={truth['vorticity_s_1']:.12g} s⁻¹; Uref={truth['u_ref_m_s']:.12g} m/s.",
        f"- Sampling extent: ±{c.values['sampling']['extent_rc']:g} Rc for transects; disk radius {c.values['sampling']['extent_rc']:g} Rc for LATTE.",
        f"- Sampling seed: {c.seed}. DOPPIO's shared intersection receives one noise vector, reused by both transects.",
        "- LATTE locations have exact quarter-turn and coordinate-axis reflection symmetry. Velocity noise remains independent at each unique point.",
        ("- Inner fits receive all synthetic observations; SOLO's null core radius means all transect points, not the real-data 30-km selection."
         if c.values["inversion"]["solo_core_radius_m"] is None else
         f"- SOLO uses a {c.values['inversion']['solo_core_radius_m']:g}-m along-track core radius; DOPPIO and LATTE use all synthetic observations."),
        "- Density labels do not equalise observation budgets; compare the actual unique counts below.",
        "- Outer fits receive all observations, subject to the shared finite-value/sign filters. No truth-based selection or error-based trimming is applied.", ""]
    if noise:
        lines += [f"Noise SD per component is {', '.join(f'{100*v:g}%' for v in c.values['noise']['fractions'])} of Uref. "
            f"There are {c.values['noise']['replicates']} trials per nonzero condition and one deterministic clean trial per method/density. "
            f"Noise draws are paired across amplitudes. Locations remain fixed. Bands are the {c.values['plotting']['percentiles'][0]:g}–{c.values['plotting']['percentiles'][1]:g} Monte Carlo percentile range, not confidence intervals.", ""]
    else:
        lines += ["HIGH-density geometry is reused exactly from the noise experiment. Background flow is uniform and unmodelled; "
            "SOLO tests eastward (parallel) and northward (perpendicular), while DOPPIO/LATTE test northward only. "
            "There is no noise or replication; each method has one shared zero-background baseline. No uncertainty bands are inferred.", ""]
    lines += ["## Manuscript table", "", "Rounded to two decimals. Centre errors are percentages of true Rc; plot centre errors use km. "
        "SOLO's aspect ratio is prescribed, so its error is not estimated (—). Medians/percentiles are conditional on accepted full fits.", "",
        markdown_table(table), "", "## Numerical interpretation and acceptance", "", interpretation(results), ""]
    summary = summarize(results)
    if noise:
        lines += ["### Ranges of condition medians", "", "The ranges below span all methods and sampling densities at each noise level.", ""]
        ranges = []
        for fraction, group in summary.loc[summary.noise_fraction > 0].groupby("noise_fraction"):
            row = {"Noise (% Uref)": 100*fraction}
            for label, col in [("Centre (% Rc)", "centre_error_pct_rc"), ("Omega (%)", "Omega_error_pct"), ("Rc (%)", "rc_error_pct")]:
                a = group[col+"_median"].dropna()
                row[label] = f"{a.min():.6g}–{a.max():.6g}" if len(a) else "unavailable"
            ranges.append(row)
        lines += [markdown_table(pd.DataFrame(ranges)), ""]
    else:
        solo = results.loc[(results.method == "SOLO") & results.success]
        fractions = solo.loc[solo.background_fraction > 0, "background_fraction"]
        if len(fractions):
            level = fractions.max()
            a = solo.loc[solo.background_fraction == level].set_index("background_direction")
            if {"eastward", "northward"}.issubset(a.index) and a.loc["northward", "rc_error_pct"] > 0:
                reduction = 100*(1-a.loc["eastward", "rc_error_pct"]/a.loc["northward", "rc_error_pct"])
                lines += [f"At {100*level:g}% background speed, SOLO Rc errors are {a.loc['eastward', 'rc_error_pct']:.9g}% eastward "
                    f"and {a.loc['northward', 'rc_error_pct']:.9g}% northward: {reduction:.9g}% relative reduction for this geometry.", ""]
    lines += ["## Actual sampling counts", "", markdown_table(sampling_summary(context)), "",
        f"Geometry figure: **{c.values['manuscript']['sampling_geometry_figure']}**. Counts are unique vectors; N_outer_true uses the true peak-speed contour.", "",
        "## Files and reproducibility", "",
        "- `trials.csv`: every attempt, fitted parameters in SI, warnings and failure statuses.",
        "- `summary.csv`: unrounded medians, noise percentiles and accepted/attempted counts.",
        "- `sampling.csv`: unique/inner/outer counts for every geometry.",
        "- `run_metadata.json`: effective config, source-file hash, seed, coordinates, method/helper hashes, Git state, versions, self-tests and CSV hashes.",
        "- `plot_metadata.json`: current rendering configuration and figure hashes, including plot-only redraws.",
        "- `figures/`: PDF and PNG manuscript figures. Replotting never refits or changes the saved trials.", "",
        "## Effective configuration", "", "```json", json.dumps(c.values, indent=2, allow_nan=False), "```", ""]
    (directory / "manuscript_summary.md").write_text("\n".join(lines))
    table.to_csv(directory / "manuscript_table.csv", index=False, float_format="%.17g")


def save_results(results, context, overrides=None):
    experiment = results.attrs["experiment"]
    directory = context.config.output / experiment
    directory.mkdir(parents=True, exist_ok=True)
    results.to_csv(directory / "trials.csv", index=False, float_format="%.17g")
    summarize(results).to_csv(directory / "summary.csv", index=False, float_format="%.17g")
    sampling_summary(context).to_csv(directory / "sampling.csv", index=False, float_format="%.17g")
    try:
        git = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip(),
               "changes": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).splitlines()}
        git["dirty"] = bool(git["changes"])
    except (OSError, subprocess.CalledProcessError):
        git = {"commit": None, "dirty": None}
    code = [ROOT / "functions.py", ROOT / "reporting.py", *[Path(__file__).with_name(f) for f in ["run.py", "sensitivity.py", "plotting.py", "outputs.py"]]]
    metadata = dict(schema_version=1, experiment=experiment,
        generated_at_utc=datetime.now(timezone.utc).isoformat(), configuration=context.config.values,
        configuration_file=str(context.config.path), configuration_file_sha256=sha256(context.config.path),
        command_line_overrides=overrides or {}, git=git, experiment_metadata=dict(results.attrs),
        code_sha256={str(p.relative_to(ROOT)): sha256(p) for p in code},
        csv_sha256={f: sha256(directory / f) for f in ["trials.csv", "summary.csv", "sampling.csv"]},
        sampling_manifest_sha256=sha256(context.manifest))
    (directory / "run_metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2, allow_nan=False)+"\n")
    return directory


def load_saved(directory, plot_config):
    """Use frozen scientific inputs/results; apply only current plotting styles."""
    directory = Path(directory)
    meta = json.loads((directory / "run_metadata.json").read_text())
    for name, digest in meta["csv_sha256"].items():
        if sha256(directory / name) != digest:
            raise ValueError(f"Saved {name} differs from its recorded hash; refusing to render mismatched results")
    values = deepcopy(meta["configuration"])
    if plot_config.values["plotting"]["percentiles"] != values["plotting"]["percentiles"]:
        raise ValueError("Plot-only redraws retain saved statistical percentiles; use a full run to change them")
    values["plotting"] = deepcopy(plot_config.values["plotting"])
    values["manuscript"] = deepcopy(plot_config.values["manuscript"])
    config = Config(values, plot_config.path)
    results = pd.read_csv(directory / "trials.csv", float_precision="round_trip")
    results.attrs = meta["experiment_metadata"]
    results.attrs["config"] = values
    truth = deepcopy(results.attrs["truth"]);truth["q"] = np.asarray(truth["q"], float)
    context = SimpleNamespace(config=config, truth=truth,
        geometries=_bank_from_json(results.attrs["sampling_coordinates_m"]))
    return results, context


def save_plot_metadata(directory, context, paths):
    directory = Path(directory)
    metadata = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_metadata_sha256": sha256(directory / "run_metadata.json"),
        "plotting": context.config.values["plotting"], "manuscript_labels": context.config.values["manuscript"],
        "plotting_code_sha256": sha256(Path(__file__).with_name("plotting.py")),
        "figure_sha256": {str(p.relative_to(directory)): sha256(p) for p in paths}}
    (directory / "plot_metadata.json").write_text(json.dumps(_json_safe(metadata),indent=2,allow_nan=False)+"\n")
