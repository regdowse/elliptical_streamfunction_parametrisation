"""Deterministic SOLO validation: sampling, shared estimators and saved results.

SI throughout. This module does not implement or copy an ESP estimator.
"""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import warnings

import numpy as np
import pandas as pd

import functions as esp
from reporting import _json_safe
from experiments.outputs import sha256, markdown_table

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments/configs/solo_idealised.json"
DATA_FILES = ("parameters.csv", "sampling.csv", "deformation.csv", "comparison.csv", "curves.json", "field.npz")


def load_config(path=DEFAULT_CONFIG):
    values = json.loads(Path(path).read_text())
    if values.get("schema_version") != 1 or values.get("experiment") != "solo_idealised":
        raise ValueError("Expected SOLO configuration schema_version=1")
    truth = values["truth"]
    if not np.isfinite(list(truth.values())).all() or truth["rc_m"] <= 0 or truth["psi0_m2_s"] == 0:
        raise ValueError("Truth must be finite, with positive Rc and nonzero psi0")
    names = [s["name"] for s in values["cases"]]
    if names != ["P1", "P2", "P3", "L1", "L2", "L3"]:
        raise ValueError("Supply six named cases P1–P3, L1–L3 in order")
    for spec in [*values["cases"], values["comparison"], values["deformation"]["transect"]]:
        coordinates(spec)
    for key in ("solo_core_radius_m", "outer_Rc_max_m"):
        value = values["inversion"][key]
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(f"{key} must be null or positive and finite")
    q12 = np.asarray(values["deformation"]["q12_inputs"], float)
    if q12.ndim != 1 or q12.size < 2 or not np.isfinite(q12).all() or np.any(np.abs(q12) >= 1):
        raise ValueError("Deformation inputs must have |q12|<1 for positive-definite Q")
    plot = values["plot"]
    for key in ("limits_km", "grid_limits_m", "velocity_limits_m_s"):
        bounds = plot[key]
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid {key}")
    for key in ("grid_points", "quiver_stride", "dpi"):
        if not isinstance(plot[key], int) or plot[key] < (2 if key == "grid_points" else 1):
            raise ValueError(f"Invalid {key}")
    if len(plot["colors"]) != 3 or not plot["formats"] or not set(plot["formats"]) <= {"pdf", "png"}:
        raise ValueError("Need three colors and pdf/png output formats")
    if not np.isfinite([plot["quiver_scale"], plot["outer_radius_limit_km"]]).all() or min(plot["quiver_scale"], plot["outer_radius_limit_km"]) <= 0:
        raise ValueError("Plot scales must be positive and finite")
    return values


def coordinates(spec):
    mid, y, half, step = [float(spec[k]) for k in ("x_mid_m", "y_m", "half_length_m", "step_m")]
    if not np.isfinite([mid, y, half, step]).all() or half <= 0 or step <= 0:
        raise ValueError("Invalid transect geometry")
    intervals = 2*half/step
    if not np.isclose(intervals, round(intervals), rtol=0, atol=1e-8) or intervals < 3:
        raise ValueError("Transects need at least four samples and an integral number of steps")
    x = np.linspace(mid-half, mid+half, int(round(intervals))+1)
    return x, np.full_like(x, y)


def observe(spec, truth, Q):
    x, y = coordinates(spec)
    omega = -2*truth["psi0_m2_s"]/truth["rc_m"]**2
    u, v = esp.model_uv_at_xy(x, y, truth["xc_m"], truth["yc_m"], Q, omega, truth["rc_m"])
    return x, y, u, v


def inner_fit(x, y, u, v, core):
    details = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        l0, r0, w, Q, omega = esp.solo(x-x[0], u, v,
            core_thresh=np.inf if core is None else core, selection_details=details)
    row = dict(xc_m=l0+x[0], yc_m=r0+y[0], inner_vorticity_s_1=w, inner_Omega_s_1=omega,
               warnings="; ".join(str(a.message) for a in caught))
    if "x0" in details:
        # Save evaluated polynomial and tangent data; rendering never calls a fitter.
        dl = x-x[0]-details["x0"]
        details.update(x_m=x, y_m=y, u_m_s=u, v_m_s=v,
            polynomial_u=details["A"]+details["B"]*dl**2,
            polynomial_v=details["C"]*dl+details["D"]*dl**3,
            tangent_u=np.full_like(x, details["A"]), tangent_v=details["C"]*dl)
    else:
        details.update(x_m=x, y_m=y, u_m_s=u, v_m_s=v,
            **{k: np.full_like(x, np.nan) for k in ("polynomial_u", "polynomial_v", "tangent_u", "tangent_v")})
    return row, Q, details


def run_experiment(config):
    truth = config["truth"]
    omega = -2*truth["psi0_m2_s"]/truth["rc_m"]**2
    rows, samples, curves = [], [], {}
    for spec in config["cases"]:
        name = spec["name"]
        x, y, u, v = observe(spec, truth, np.eye(2))
        row, Q, details = inner_fit(x, y, u, v, config["inversion"]["solo_core_radius_m"])
        outer = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            rc, psi, fitted_omega = esp.out_core_param_fit(x, y, u, v, row["xc_m"], row["yc_m"], Q,
                Omega0=row["inner_Omega_s_1"], Rc_max=config["inversion"]["outer_Rc_max_m"],
                fit_details=outer, core_mask=details.get("core_mask"))
        row.update(case=name, Rc_m=rc, psi0_m2_s=psi, Omega_s_1=fitted_omega,
            vorticity_s_1=2*fitted_omega, status=outer["status"],
            warnings="; ".join(filter(None, [row["warnings"], *(str(a.message) for a in caught)])),
            count=len(x), length_m=x[-1]-x[0], y_transect_m=y[0],
            centre_error_m=np.hypot(row["xc_m"]-truth["xc_m"], row["yc_m"]-truth["yc_m"]),
            Rc_error_m=abs(rc-truth["rc_m"]), peak_radius_error_m=abs(rc-truth["rc_m"])/np.sqrt(2),
            Omega_error_s_1=abs(fitted_omega-omega), vorticity_error_s_1=2*abs(fitted_omega-omega))
        rows.append(row)
        samples.extend(dict(case=name, index=i, x_m=a, y_m=b, u_m_s=c, v_m_s=d)
                       for i, (a,b,c,d) in enumerate(zip(x,y,u,v)))
        details["outer"] = outer
        curves[name] = details
    deformed = []
    spec = config["deformation"]["transect"]
    for q12 in config["deformation"]["q12_inputs"]:
        Q = np.array([[1., q12], [q12, 1.]])/np.sqrt(1-q12*q12)
        x,y,u,v = observe(spec, truth, Q)
        row, _, _ = inner_fit(x,y,u,v,config["inversion"]["solo_core_radius_m"])
        row.update(q12_input=q12, Q11=Q[0,0], Q12=Q[0,1], Q22=Q[1,1],
            aspect_ratio=esp.axis_ratio_from_Q(Q),
            true_Omega_s_1=omega, true_vorticity_s_1=omega*np.trace(Q),
            centre_error_m=np.hypot(row["xc_m"]-truth["xc_m"], row["yc_m"]-truth["yc_m"]),
            status="optimized" if np.isfinite([row["xc_m"],row["yc_m"],row["inner_Omega_s_1"]]).all() else "inner_failed")
        deformed.append(row)
    # Both methods use precisely the same observed Gaussian vectors.
    x,y,u,v = observe(config["comparison"], truth, np.eye(2))
    row, _, details = inner_fit(x,y,u,v,config["inversion"]["solo_core_radius_m"])
    row.update(method="SOLO", Omega_s_1=row["inner_Omega_s_1"])
    pu, pv = np.polyfit(x,u,1), np.polyfit(x,v,1)
    x0 = -pv[1]/pv[0]
    y0 = y[0]+np.polyval(pu,x0)/pv[0]
    baseline = dict(method="MOCA", xc_m=x0, yc_m=y0, Omega_s_1=pv[0])
    comparison = pd.DataFrame([row,baseline])
    comparison["centre_error_m"] = np.hypot(comparison.xc_m-truth["xc_m"],comparison.yc_m-truth["yc_m"])
    details.update(moca_u=np.polyval(pu,x), moca_v=np.polyval(pv,x))
    curves["comparison"] = details
    lo,hi = config["plot"]["grid_limits_m"]
    axis = np.linspace(lo,hi,config["plot"]["grid_points"])
    X,Y = np.meshgrid(axis,axis,indexing="ij")
    u,v = esp.model_uv_at_xy(X,Y,truth["xc_m"],truth["yc_m"],np.eye(2),omega,truth["rc_m"])
    field = dict(x_m=axis,y_m=axis,u_m_s=u,v_m_s=v)
    return dict(parameters=pd.DataFrame(rows),sampling=pd.DataFrame(samples),
        deformation=pd.DataFrame(deformed),comparison=comparison,curves=curves,field=field)


def write_report(data, config, directory):
    truth = config["truth"]
    params = data["parameters"]
    table = params[["case","xc_m","yc_m","vorticity_s_1","Omega_s_1","Rc_m","psi0_m2_s","status"]].copy()
    for name in ("xc_m","yc_m","Rc_m"):
        table[name.replace("_m","_km")] = table.pop(name)/1000
    # Scientific notation prevents small SI rates from printing as 0.00.
    for name in ("vorticity_s_1","Omega_s_1"):
        table[name] = table[name].map(lambda v: f"{v:.8e}")
    table.to_csv(directory/"manuscript_table.csv", index=False)
    deformation = data["deformation"]
    comparison = data["comparison"][["method","xc_m","yc_m","Omega_s_1","centre_error_m"]].copy()
    comparison["Omega_s_1"] = comparison.Omega_s_1.map(lambda v: f"{v:.8e}")
    lines = ["# SOLO idealised validation — manuscript results", "",
        "Figures: SOLO.png, SOLO_outercore_fits.png, Improved_MOCA.png, SOLO_AR_Err.png; SOLO parameter table.", "",
        "## Truth and design", "",
        f"Gaussian truth: centre ({truth['xc_m']:g}, {truth['yc_m']:g}) m; Rc={truth['rc_m']/1000:g} km; psi0={truth['psi0_m2_s']:g} m²/s; Q=I.",
        f"Omega={-2*truth['psi0_m2_s']/truth['rc_m']**2:.12g} s⁻¹; vorticity=2 Omega. Peak radius=Rc/√2={truth['rc_m']/np.sqrt(2)/1000:.9g} km.",
        f"Inner core selection radius (m): {config['inversion']['solo_core_radius_m']} (null means all observations). Outer Rc upper bound (m): {config['inversion']['outer_Rc_max_m']}.", "",
        markdown_table(params[["case","count","length_m","y_transect_m","status"]]), "",
        "## Parameters", "", markdown_table(table), "",
        "Rates above are final outer-fit rates; inner estimates are separately saved in parameters.csv. Scientific CSVs retain full precision in SI; manuscript_table.csv formats rates and labels displayed km columns explicitly. Unavailable fits remain NaN/null; no accuracy-based filtering.", "",
        "## Errors", "",
        f"Maximum centre error: {params.centre_error_m.max()/1000:.9g} km; maximum Rc error: {params.Rc_error_m.max()/1000:.9g} km; maximum peak-radius error: {params.peak_radius_error_m.max()/1000:.9g} km.", "",
        "## MOCA comparison", "", markdown_table(comparison), "",
        "Both methods use the same Gaussian observations, with the configured SOLO core selection. SOLO green lines are tangents of its fitted polynomials; MOCA uses linear regressions for both velocity components. No exact inner-core velocities, true centre or finite-difference slopes are substituted into the estimates.", "",
        "## Deformation", "",
        "Q=[[1,q12],[q12,1]] / sqrt(1-q12²); fixed Gaussian Rc and psi0. Only inner SOLO centre estimates are used; there is no outer refit in this test.", "",
        markdown_table(deformation[["q12_input","aspect_ratio","xc_m","yc_m","centre_error_m","status"]]), "",
        "## Manuscript checks", "",
        "The default P transects have full length 140 km (half-length 70 km), not the 70-km length stated in the current manuscript. L lengths are 80/140/200 km. Sampling follows the actual calculations.",
        "The previous comparison supplied exact inner-core velocities to SOLO but Gaussian observations to MOCA. This run corrects that unequal-input comparison. Recheck its claimed centre and rotation estimates.",
        "Deformation truth uses Omega=-2 psi0/Rc² and vorticity=Omega trace(Q), correcting the earlier factor-of-two truth label. Only centre error is plotted.",
        "Shared estimators use SI and the current corrected outer fitter; older manuscript values need checking against this run. Original run provenance and data hashes are in run_metadata.json.", ""]
    (directory/"manuscript_summary.md").write_text("\n".join(lines))


def save_results(data, config, config_path, output_root=None):
    root = Path(output_root or config["output_directory"])
    directory = (root if root.is_absolute() else ROOT/root)/"solo_idealised"
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("parameters","sampling","deformation","comparison"):
        data[name].to_csv(directory/f"{name}.csv",index=False)
    (directory/"curves.json").write_text(json.dumps(_json_safe(data["curves"]),allow_nan=False))
    np.savez_compressed(directory/"field.npz",**data["field"])
    write_report(data,config,directory)
    def git(*args):
        result = subprocess.run(["git",*args],cwd=ROOT,text=True,capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    metadata = dict(schema_version=1,experiment="solo_idealised",utc=datetime.now(timezone.utc).isoformat(),
        config=config,config_path=str(Path(config_path).resolve()),config_sha256=sha256(config_path),
        output_root_override=None if output_root is None else str(output_root),
        git_commit=git("rev-parse","HEAD"),git_status=git("status","--short"),python=platform.python_version(),
        versions={name:importlib.metadata.version(name) for name in ("numpy","scipy","pandas","matplotlib")},
        source_sha256={name:sha256(ROOT/name) for name in ("functions.py","reporting.py","experiments/solo.py","experiments/run.py")},
        data_sha256={name:sha256(directory/name) for name in (*DATA_FILES,"manuscript_table.csv")})
    (directory/"run_metadata.json").write_text(json.dumps(_json_safe(metadata),indent=2,allow_nan=False)+"\n")
    return directory


def load_saved(directory, current_config):
    metadata = json.loads((directory/"run_metadata.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("experiment") != "solo_idealised":
        raise ValueError("Unsupported saved SOLO experiment")
    for name, digest in metadata["data_sha256"].items():
        if sha256(directory/name) != digest:
            raise ValueError(f"Saved data changed: {name}")
    data = {name:pd.read_csv(directory/f"{name}.csv") for name in ("parameters","sampling","deformation","comparison")}
    data["curves"] = json.loads((directory/"curves.json").read_text())
    with np.load(directory/"field.npz",allow_pickle=False) as field:
        data["field"] = {key:field[key].copy() for key in field.files}
    config = deepcopy(metadata["config"])
    # Scientific settings remain frozen. Presentation settings can be changed.
    config["plot"] = deepcopy(current_config["plot"])
    for key in ("grid_limits_m", "grid_points"):
        config["plot"][key] = metadata["config"]["plot"][key]
    return data, config
