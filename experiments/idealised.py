"""Main-paper DOPPIO/LATTE validation and illustrative ESP sampling schematic.

Experimental design and reporting only: estimators live in root functions.py.
All numerical inputs/outputs are SI. Plotting consumes saved data without fitting.
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
from scipy.interpolate import RegularGridInterpolator

import functions as esp
from reporting import _json_safe
from experiments.outputs import sha256, markdown_table

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("doppio_idealised", "latte_idealised", "schematic")
ERRORS = ("centre_error_m", "vorticity_error_s_1", "Omega_error_s_1", "aspect_ratio_error", "Rc_error_m")


def positive(value, label, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be positive and finite")
    if integer and not isinstance(value,int):
        raise ValueError(f"{label} must be an integer")


def bounds(value, label):
    if len(value) != 2 or not np.isfinite(value).all() or value[0] >= value[1]:
        raise ValueError(f"{label} needs two increasing finite limits")


def vector(value, label):
    if np.shape(value) != (2,) or not np.isfinite(value).all():
        raise ValueError(f"{label} needs two finite coordinates")


def grid_axis(spec):
    bounds(spec["limits_m"], "Grid")
    positive(spec["step_m"], "Grid step")
    lo,hi = spec["limits_m"]
    n = (hi-lo)/spec["step_m"]
    if not np.isclose(n,round(n),rtol=0,atol=1e-8):
        raise ValueError("Grid span must be an integral number of steps")
    return np.linspace(lo,hi,int(round(n))+1)


def truth_parameters(config):
    t = config["truth"]
    Q = np.asarray(t["Q_input"],float)
    if not esp.is_elliptical_Q(Q,require_det_one=False):
        raise ValueError("Truth Q_input must be symmetric positive definite")
    Q = Q/np.sqrt(np.linalg.det(Q))
    omega = -2*t["psi0_m2_s"]/t["rc_m"]**2
    return Q,omega


def cross_coordinates(spec):
    vector(spec["intersection_m"], "Cross intersection")
    half,step = spec["half_length_m"],spec["step_m"]
    positive(half,"Transect half length");positive(step,"Transect step")
    n = half/step
    if not np.isclose(n,round(n),rtol=0,atol=1e-8) or n < 2:
        raise ValueError("Each cross arm needs >=5 samples, including one exact shared intersection")
    d = np.arange(-int(round(n)),int(round(n))+1)*step
    xc,yc = spec["intersection_m"]
    return (xc+d,np.full(d.shape,yc),np.full(d.shape,xc),yc+d)


def load_config(path, name):
    c = json.loads(Path(path).read_text())
    if c.get("schema_version") != 1 or c.get("experiment") != name or name not in NAMES:
        raise ValueError(f"Expected schema_version=1, experiment={name}")
    t = c["truth"]
    vector([t["xc_m"],t["yc_m"]],"Truth centre")
    positive(t["rc_m"],"True Rc")
    if not np.isfinite(t["psi0_m2_s"]) or t["psi0_m2_s"] == 0:
        raise ValueError("True psi0 must be finite and nonzero")
    truth_parameters(c)
    p = c["plot"]
    bounds(p["limits_km"],"Display limits")
    for key in ("dpi","quiver_scale"):
        positive(p[key],key,integer=key == "dpi")
    if not p["formats"] or not set(p["formats"]) <= {"pdf","png"}:
        raise ValueError("Output formats must be pdf/png")
    if len(p["colors"]) != (4 if name == "latte_idealised" else 3):
        raise ValueError("Invalid number of figure colors")
    if name != "schematic":
        bounds(p["grid_limits_m"],"Display grid")
        positive(p["grid_points"],"Grid points",integer=True)
        if p["grid_points"] < 2:
            raise ValueError("Display grid needs >=2 points")
        positive(p["quiver_stride"],"Quiver stride",integer=True)
        positive(p["outer_radius_limit_km"],"Outer display radius")
        rc_max = c["inversion"]["outer_Rc_max_m"]
        if rc_max is not None:
            positive(rc_max,"Outer radius bound")
    if name == "doppio_idealised":
        if [s["name"] for s in c["cases"]] != ["EP1","EP2","EP3","EL1","EL2","EL3"]:
            raise ValueError("Expected EP1–EP3 and EL1–EL3 cases in order")
        for s in c["cases"]:
            cross_coordinates(s)
        bounds(p["velocity_limits_m_s"],"Velocity limits")
        if set(c["comparison_tolerances"]) != set(ERRORS):
            raise ValueError("Comparison requires tolerances for all five metrics")
        if not all(np.isfinite(v) and v >= 0 for v in c["comparison_tolerances"].values()):
            raise ValueError("Comparison tolerances must be nonnegative and finite")
    else:
        grid_axis(c["sampling_grid"])
        s = c["sampling"]
        if name == "latte_idealised":
            bounds(s["ship"]["y_limits_m"],"Ship latitude range")
            for key in ("count","y_scale_m"):
                positive(s["ship"][key],key,integer=key == "count")
            if s["ship"]["count"] < 3:
                raise ValueError("Ship sample needs at least three points")
            vector([s["ship"]["x_amplitude_m"],s["ship"]["x_offset_m"]],"Ship shape")
            vector(s["drifter"]["initial_m"],"Drifter initial point")
            positive(s["drifter"]["dt_s"],"Drifter step seconds")
            positive(s["drifter"]["steps"],"Drifter steps",integer=True)
            if len(s["transects"]) != 2:
                raise ValueError("Expected two oblique transects")
            for transect in s["transects"]:
                vector(transect["start_m"],"Transect start");vector(transect["end_m"],"Transect end")
                positive(transect["count"],"Transect count",integer=True)
            bounds(s["grid"]["limits_m"],"Sample grid")
            positive(s["grid"]["step_m"],"Sample grid step")
            if s["grid"]["endpoint"] is not False:
                raise ValueError("Sample grid uses an explicitly excluded upper endpoint")
        else:
            if not isinstance(s["seed"],int) or s["seed"] < 0:
                raise ValueError("Seed must be a nonnegative integer")
            for key in ("core_count","outer_count"):
                positive(s[key],key,integer=True)
            if not 0 < s["core_radius_m"] < s["outer_radius_m"] <= s["sample_radius_m"]:
                raise ValueError("Expected 0 < core radius < outer radius <= sampling radius")
            vector(s["initial_guess_m"],"Initial guess")
            if not np.allclose(truth_parameters(c)[0],np.eye(2)):
                raise ValueError("This circular schematic requires Q=I")
    return c


def velocity(x,y,config):
    Q,omega = truth_parameters(config)
    t = config["truth"]
    return esp.model_uv_at_xy(x,y,t["xc_m"],t["yc_m"],Q,omega,t["rc_m"])


def make_field(config, sampling=False):
    if sampling:
        axis = grid_axis(config["sampling_grid"])
    else:
        axis = np.linspace(*config["plot"]["grid_limits_m"],config["plot"]["grid_points"])
    x,y = np.meshgrid(axis,axis,indexing="ij")
    u,v = velocity(x,y,config)
    return dict(x_m=axis,y_m=axis,u_m_s=u,v_m_s=v)


def observe_cross(spec, config):
    x1,y1,x2,y2 = cross_coordinates(spec)
    u1,v1 = velocity(x1,y1,config);u2,v2 = velocity(x2,y2,config)
    arms = [np.array([x1,y1,u1,v1]),np.array([x2,y2,u2,v2])]
    # Keep both complete arms for DOPPIO; give each location one weight in outer/LATTE fits.
    shared = np.all(np.column_stack([x1,y1]) == spec["intersection_m"],axis=1)
    unique = np.concatenate([arms[0][:,~shared],arms[1]],axis=1)
    return arms,unique


def evaluate_doppio(arms, details):
    profiles = {}
    for i,arm in enumerate(arms):
        for key,value in zip(("x","y","u","v"),arm):
            profiles[f"{key}{i+1}"] = value
    if details.get("status") != "optimized":
        for key in ("polynomial_u1","polynomial_v1","polynomial_u2","polynomial_v2",
                    "tangent_u1","tangent_v1","tangent_u2","tangent_v2"):
            profiles[key] = np.full(arms[0].shape[1],np.nan)
        return profiles
    for letter,component,arm_index,coord,root_key in (
            ("A","u",0,0,"x_root"),("B","v",0,0,"x_root"),
            ("C","u",1,1,"y_root"),("D","v",1,1,"y_root")):
        p = details[letter]
        z = arms[arm_index][coord]-details["coordinate_origin"][coord]
        root = details[root_key]
        derivative = np.polynomial.polynomial.polyval(root,np.polynomial.polynomial.polyder(p))
        profiles[f"polynomial_{component}{arm_index+1}"] = np.polynomial.polynomial.polyval(z,p)
        profiles[f"tangent_{component}{arm_index+1}"] = np.polynomial.polynomial.polyval(root,p)+derivative*(z-root)
    return profiles


def fit_case(name, method, observations, config, arms=None):
    x,y,u,v = observations
    inner,outer = {},{}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if method == "DOPPIO":
            xc,yc,w,Q,omega = esp.doppio(*arms[0],*arms[1],fit_details=inner)
        else:
            xc,yc,w,Q,omega,_ = esp.latte(x,y,u,v,fit_details=inner)
        rc,psi,final_omega = esp.out_core_param_fit(x,y,u,v,xc,yc,Q,Omega0=omega,
            Rc_max=config["inversion"]["outer_Rc_max_m"],fit_details=outer)
    t = config["truth"];Qt,ot = truth_parameters(config)
    row = dict(case=name,method=method,count=len(x),xc_m=xc,yc_m=yc,Q11=Q[0,0],Q12=Q[0,1],Q22=Q[1,1],
        inner_Omega_s_1=omega,inner_vorticity_s_1=w,Omega_s_1=final_omega,
        vorticity_s_1=final_omega*np.trace(Q),Rc_m=rc,psi0_m2_s=psi,aspect_ratio=esp.axis_ratio_from_Q(Q),
        inner_status=inner.get("status","unavailable"),outer_status=outer["status"],
        status="optimized" if inner.get("status") == outer["status"] == "optimized" else "unavailable",
        warnings="; ".join(str(a.message) for a in caught))
    row.update(x_error_m=abs(xc-t["xc_m"]),y_error_m=abs(yc-t["yc_m"]),
        centre_error_m=np.hypot(xc-t["xc_m"],yc-t["yc_m"]),
        Omega_error_s_1=abs(final_omega-ot),vorticity_error_s_1=abs(row["vorticity_s_1"]-ot*np.trace(Qt)),
        aspect_ratio_error=abs(row["aspect_ratio"]-esp.axis_ratio_from_Q(Qt)),
        Rc_error_m=abs(rc-t["rc_m"]),peak_radius_error_m=abs(rc-t["rc_m"])/np.sqrt(2))
    curve = dict(x_m=x,y_m=y,u_m_s=u,v_m_s=v,inner=inner,outer=outer)
    if arms is not None:
        curve["profiles"] = evaluate_doppio(arms,inner)
    return row,curve


def compare_methods(parameters, tolerances):
    rows = []
    for case in parameters.case.unique():
        pair = parameters.loc[parameters.case == case].set_index("method")
        for metric in ERRORS:
            lat,dop = pair.loc["LATTE",metric],pair.loc["DOPPIO",metric]
            valid = pair.status.eq("optimized").all() and np.isfinite([lat,dop]).all()
            reason = ("fit_unavailable" if not valid else "zero_reference_error" if lat == 0 else
                      "below_difference_tolerance" if abs(lat-dop) < tolerances[metric] else "included")
            rows.append(dict(case=case,metric=metric,latte_error=lat,doppio_error=dop,
                difference_tolerance=tolerances[metric],status=reason,
                improvement_pct=100*(lat-dop)/lat if reason == "included" else np.nan))
    return pd.DataFrame(rows)


def sampling_rows(name, observations, **extras):
    return [dict(case=name,index=i,x_m=x,y_m=y,u_m_s=u,v_m_s=v,**extras)
            for i,(x,y,u,v) in enumerate(zip(*observations))]


def run_doppio(config):
    rows,samples,curves = [],[],{}
    for spec in config["cases"]:
        arms,observations = observe_cross(spec,config)
        samples.extend(sampling_rows(spec["name"],observations))
        for method in ("DOPPIO","LATTE"):
            row,curve = fit_case(spec["name"],method,observations,config,arms if method == "DOPPIO" else None)
            rows.append(row);curves[f"{spec['name']}_{method}"] = curve
    parameters = pd.DataFrame(rows)
    return dict(parameters=parameters,sampling=pd.DataFrame(samples),curves=curves,field=make_field(config),
        comparison=compare_methods(parameters,config["comparison_tolerances"]))


def latte_samples(config):
    # Preserve bilinear sampling of the original 1-km grid, rather than silently
    # replacing off-grid observations with analytic velocities.
    field = make_field(config,sampling=True)
    axes = (field["x_m"],field["y_m"])
    fu,fv = [RegularGridInterpolator(axes,field[k],bounds_error=True) for k in ("u_m_s","v_m_s")]
    def at(x,y):
        points = np.column_stack([x,y])
        return np.asarray([x,y,fu(points),fv(points)])
    s = config["sampling"]
    ship = s["ship"]
    y = np.linspace(*ship["y_limits_m"],ship["count"])
    x = ship["x_amplitude_m"]*np.sin(y/ship["y_scale_m"])+ship["x_offset_m"]
    observations = {"Ex1":at(x,y)}
    point = np.asarray(s["drifter"]["initial_m"],float)
    trajectory = []
    for _ in range(s["drifter"]["steps"]):
        # Forward Euler, excluding initial state, exactly as in the original design.
        vector_uv = np.array([fu(point[None,:])[0],fv(point[None,:])[0]])
        point = point+s["drifter"]["dt_s"]*vector_uv
        if np.any(point < axes[0][0]) or np.any(point > axes[0][-1]):
            raise ValueError("Drifter left the interpolation grid; enlarge the configured domain")
        trajectory.append(point.copy())
    path = np.asarray(trajectory)
    observations["Ex2"] = at(path[:,0],path[:,1])
    transects = [np.linspace(t["start_m"],t["end_m"],t["count"]) for t in s["transects"]]
    xy = np.concatenate(transects)
    observations["Ex3"] = at(xy[:,0],xy[:,1])
    grid = s["grid"]
    axis = np.arange(*grid["limits_m"],grid["step_m"])
    x,y = np.meshgrid(axis,axis)
    observations["Ex4"] = at(x.ravel(),y.ravel())
    return observations


def run_latte(config):
    rows,samples,curves = [],[],{}
    for name,observations in latte_samples(config).items():
        row,curve = fit_case(name,"LATTE",observations,config)
        rows.append(row);curves[name] = curve
        samples.extend(sampling_rows(name,observations))
    return dict(parameters=pd.DataFrame(rows),sampling=pd.DataFrame(samples),curves=curves,field=make_field(config))


def run_schematic(config):
    s,t = config["sampling"],config["truth"]
    rng = np.random.default_rng(s["seed"])
    theta = 2*np.pi*rng.random(s["core_count"])
    radius = s["core_radius_m"]*np.sqrt(rng.random(s["core_count"]))
    theta_out = 2*np.pi*rng.random(s["outer_count"])
    radius_out = s["core_radius_m"]+(s["sample_radius_m"]-s["core_radius_m"])*rng.random(s["outer_count"])**2
    x = t["xc_m"]+np.r_[radius*np.cos(theta),radius_out*np.cos(theta_out)]
    y = t["yc_m"]+np.r_[radius*np.sin(theta),radius_out*np.sin(theta_out)]
    # Snap to the original regular field grid. Distinct draws can share a point;
    # retain them for this illustrative figure and report the unique count.
    axis = grid_axis(config["sampling_grid"])
    if min(x.min(),y.min()) < axis[0] or max(x.max(),y.max()) > axis[-1]:
        raise ValueError("Schematic samples exceed the sampling grid")
    x,y = [axis[np.abs(axis[:,None]-a).argmin(axis=0)] for a in (x,y)]
    u,v = velocity(x,y,config)
    frame = pd.DataFrame(sampling_rows("illustration",[x,y,u,v]))
    radius = np.hypot(x-t["xc_m"],y-t["yc_m"])
    frame["initial_mask"] = np.hypot(x-s["initial_guess_m"][0],y-s["initial_guess_m"][1]) < s["core_radius_m"]
    frame["core_mask"] = radius < s["core_radius_m"]
    frame["outer_mask"] = (radius < s["outer_radius_m"]) & ~frame.core_mask
    return dict(sampling=frame)


def run_experiment(config):
    return {"doppio_idealised":run_doppio,"latte_idealised":run_latte,"schematic":run_schematic}[config["experiment"]](config)


def manuscript_table(parameters):
    columns = ["case","method","xc_m","yc_m","vorticity_s_1","Omega_s_1","aspect_ratio","Rc_m","status"]
    frame = parameters[columns].copy()
    for name in ("xc_m","yc_m","Rc_m"):
        frame[name.replace("_m","_km")] = frame.pop(name)/1000
    for name in ("vorticity_s_1","Omega_s_1"):
        frame[name] = frame[name].map(lambda x:f"{x:.8e}")
    return frame


def write_report(data,config,directory):
    name = config["experiment"];t = config["truth"];Q,omega = truth_parameters(config)
    lines = [f"# {name} — manuscript results", "", "## Truth and design", "",
        f"Gaussian truth: centre=({t['xc_m']:g},{t['yc_m']:g}) m; Rc={t['rc_m']/1000:g} km; psi0={t['psi0_m2_s']:g} m²/s.",
        f"Q={Q.tolist()}; aspect ratio={esp.axis_ratio_from_Q(Q):.12g}; Omega={omega:.12g} s⁻¹; centre vorticity={omega*np.trace(Q):.12g} s⁻¹.",
        "All numerical inputs/fits and scientific CSVs use SI. Q is normalised to unit determinant. Figure coordinates display km. Full effective configuration is appended below.", ""]
    if name == "schematic":
        f = data["sampling"]
        lines += ["Figure: ESP_schematic.png (also exported as PDF).", "",
            "Illustration only: known true centre and circular masks demonstrate the workflow. No inversion, fitted centre or parameter-recovery result is claimed.",
            f"Seed={config['sampling']['seed']}; {len(f)} samples, {len(f[['x_m','y_m']].drop_duplicates())} unique locations. Selected counts: initial={int(f.initial_mask.sum())}, inner={int(f.core_mask.sum())}, outer annulus={int(f.outer_mask.sum())}.",
            "The earlier illustration used unseeded random samples. This is a new fixed, reproducible realisation of the same sampling design, snapped to the 1-km grid. Duplicate snapped locations are retained.", ""]
    else:
        p = data["parameters"]
        table = manuscript_table(p)
        table.to_csv(directory/"manuscript_table.csv",index=False)
        lines += ["## Parameters", "",markdown_table(table), "",
            "Vorticity is final outer-fit Omega times trace(Q). Preliminary inner rates are saved separately; plots use these same stored fits, with no second plotting fit.",
            f"Accepted full fits: {int(p.status.eq('optimized').sum())}/{len(p)}. Failed attempts remain in the CSVs with NaN/null and explicit inner/outer status; no error-based rejection.", "",
            "## Absolute errors", "",
            markdown_table(p[["case","method","x_error_m","y_error_m",*ERRORS,"peak_radius_error_m"]].assign(
                vorticity_error_s_1=p.vorticity_error_s_1.map(lambda x:f"{x:.8e}"),
                Omega_error_s_1=p.Omega_error_s_1.map(lambda x:f"{x:.8e}"))), ""]
        for method,frame in p.groupby("method",sort=False):
            lines += [f"{method}: max coordinate error={max(frame.x_error_m.max(),frame.y_error_m.max())/1000:.9g} km; max Euclidean centre error={frame.centre_error_m.max()/1000:.9g} km; max Rc error={frame.Rc_error_m.max()/1000:.9g} km; max peak-radius error={frame.peak_radius_error_m.max()/1000:.9g} km; max aspect-ratio error={frame.aspect_ratio_error.max():.9g}.", ""]
        if name == "doppio_idealised":
            comparison = data["comparison"]
            display = comparison.pivot(index="case",columns="metric",values="improvement_pct").reindex(p.case.unique())
            display.loc["Mean"] = display.mean()
            display.reset_index().to_csv(directory/"comparison_table.csv",index=False)
            lines += ["Figures: DOPPIO.png and DOPPIO_outercore_fits.png; DOPPIO parameter table and DOPPIO-versus-LATTE improvement table.", "",
                "## Paired DOPPIO versus LATTE", "",
                "Both estimators receive the same Gaussian vectors at the same cross. DOPPIO receives both full arms; LATTE and both outer fits receive their unique union, counting the shared intersection once. Each estimator's own centre and Q are used for its outer fit.",
                "Improvement (%) = 100 × (LATTE absolute error − DOPPIO absolute error) / LATTE absolute error. Positive means DOPPIO has smaller error. Exclude unavailable paired fits, zero LATTE error, or absolute error differences strictly below the configured metric tolerance. Raw errors and every exclusion reason are retained in comparison.csv.",
                "Tolerance units match each metric's column; defaults are 1000 m (centre/Rc), 1e−7 s⁻¹ (rates), 0.005 (aspect ratio). Means use full-precision included percentages, before display rounding; each metric mean can have a different number of cases.", "",
                markdown_table(display.reset_index()), "",
                f"Overall mean across included case/metric entries: {comparison.improvement_pct.mean():.9g}%; included={comparison.improvement_pct.notna().sum()}/{len(comparison)}.", "",
                "## Manuscript checks", "",
                "EL1–EL3 intersections are (10,10) km, as calculated, not the (15,15) km stated in the current manuscript. EP intersections are (10,10), (−25,25), (−30,−30) km; half lengths 30 km. EL half lengths are 20/30/40 km, all sampled every 10 km.",
                "The corrected shared DOPPIO root constraints and SI fitting can change old results. The earlier parameter table mixed inner vorticity with outer Omega; this export keeps them separate and reports final rates consistently. The old comparison also rounded before averaging and added a denominator epsilon; this export averages full precision and marks zero-reference percentages unavailable.", ""]
        else:
            lines += ["Figure: LATTE.png; LATTE parameter table.", "",
                "## Sampling and manuscript checks", "",
                "Ex1: sinusoidal ship transect (15 samples). Ex2: exaggerated forward-Euler drifter trajectory, excluding its initial point (30 samples). Ex3: two oblique transects (10 points each). Ex4: 5×5 grid, −50…30 km in each coordinate (upper 50-km endpoint excluded). Counts refer to defaults; effective settings below govern changed runs.",
                "Velocities are bilinearly interpolated from the configured 1-km truth grid, preserving off-grid sampling rather than substituting analytic velocities. The drifter's original update used km positions and m/s velocity with dt=6; its equivalent SI timestep is 6000 s (100 min), not 6 s or 6 h. Default trajectory duration=180000 s (50 h); this deliberately exaggerated Euler path is not a high-accuracy particle integrator.",
                "All synthetic observations are passed to the inner estimator; no observational core-selection radius is introduced. The corrected outer fitter is unlimited unless configured. Recheck existing manuscript values, including its Euclidean centre-error claim, against these results.", ""]
    lines += ["## Effective scientific configuration", "", "```json",json.dumps(config,indent=2),"```", "",
        "Unrounded results, sampling coordinates and diagnostics are saved alongside this report. run_metadata.json records source/data hashes, Git state and versions. Plot-only rendering preserves this report and original scientific files.", ""]
    (directory/"manuscript_summary.md").write_text("\n".join(lines))


def output_directory(config,output_root=None):
    root = Path(output_root or config["output_directory"])
    return (root if root.is_absolute() else ROOT/root)/config["experiment"]


def save_results(data,config,config_path,output_root=None):
    directory = output_directory(config,output_root)
    directory.mkdir(parents=True,exist_ok=True)
    files = []
    for key,value in data.items():
        if isinstance(value,pd.DataFrame):
            path = directory/f"{key}.csv";value.to_csv(path,index=False);files.append(path)
        elif key == "field":
            path = directory/"field.npz";np.savez_compressed(path,**value);files.append(path)
        elif key == "curves":
            path = directory/"curves.json"
            path.write_text(json.dumps(_json_safe(value),allow_nan=False));files.append(path)
    write_report(data,config,directory)
    files.extend(directory/f for f in ("manuscript_table.csv","comparison_table.csv","manuscript_summary.md") if (directory/f).is_file())
    def git(*args):
        result = subprocess.run(["git",*args],cwd=ROOT,text=True,capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None
    metadata = dict(schema_version=1,experiment=config["experiment"],utc=datetime.now(timezone.utc).isoformat(),
        config=config,config_path=str(Path(config_path).resolve()),config_sha256=sha256(config_path),
        output_root_override=None if output_root is None else str(output_root),
        git_commit=git("rev-parse","HEAD"),git_status=git("status","--short"),python=platform.python_version(),
        versions={name:importlib.metadata.version(name) for name in ("numpy","scipy","pandas","matplotlib")},
        source_sha256={name:sha256(ROOT/name) for name in ("functions.py","reporting.py","experiments/idealised.py","experiments/outputs.py","experiments/run.py")},
        data_sha256={p.name:sha256(p) for p in files})
    (directory/"run_metadata.json").write_text(json.dumps(_json_safe(metadata),indent=2,allow_nan=False)+"\n")
    return directory


def load_saved(directory,current_config):
    metadata = json.loads((directory/"run_metadata.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("experiment") != current_config["experiment"]:
        raise ValueError("Unsupported saved idealised experiment")
    data = {}
    for name,digest in metadata["data_sha256"].items():
        path = directory/name
        if sha256(path) != digest:
            raise ValueError(f"Saved data changed: {name}")
        if name in ("parameters.csv","sampling.csv","comparison.csv"):
            data[path.stem] = pd.read_csv(path)
        elif name == "curves.json":
            data["curves"] = json.loads(path.read_text())
        elif name == "field.npz":
            with np.load(path,allow_pickle=False) as field:
                data["field"] = {key:field[key].copy() for key in field.files}
    config = deepcopy(metadata["config"])
    config["plot"] = deepcopy(current_config["plot"])
    for key in ("grid_points","grid_limits_m"):
        if key in metadata["config"]["plot"]:
            config["plot"][key] = metadata["config"]["plot"][key]
    return data,config
