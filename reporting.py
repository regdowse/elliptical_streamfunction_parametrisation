"""Configuration and results reporting for real-data ESP application notebooks.

All values come from the current run. Configuration utilities select explicitly
requested dates/windows; reporting/export never fits models or assumes agreement
with published manuscript numbers.
"""
import hashlib
import importlib.metadata
from pathlib import Path
import platform as python_platform
import contextlib
import io
import json
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def repo_path(root, path):
    """Resolve configured paths relative to the repository, never the kernel cwd."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(root) / path


def utc_timestamp(value):
    """Timezone-naive representation of UTC, matching notebook time coordinates."""
    t = pd.Timestamp(value)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def load_application_config(path, application, root):
    """Read an application JSON configuration with explicit units in key names."""
    config = json.loads(Path(path).read_text())
    if config.get("schema_version") != 1 or config.get("application") != application:
        raise ValueError(f"Expected schema_version=1 and application={application!r} in {path}")
    for section in ["inputs", "coordinates", "analysis", "plot", "outer_fit", "output_dir"]:
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
    for name, file in config["inputs"].items():
        if not repo_path(root, file).is_file():
            raise FileNotFoundError(f"Configured {name} input does not exist: {repo_path(root, file)}")
    for section in ["analysis", "plot", "outer_fit"]:
        for key, value in config[section].items():
            if key.endswith(("radius_m", "half_length_m", "half_window_hours", "step_m", "limit_m_s")) or key in ["quiver_spacing", "quiver_scale", "maxfev"]:
                if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{section}.{key} must be a positive finite number")
    for axis in ["x_limits_km", "y_limits_km"]:
        limits = config["plot"][axis]
        if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
            raise ValueError(f"plot.{axis} must contain increasing finite limits")
    for axis in ["longitude", "latitude"]:
        value = config["coordinates"].get(axis)
        if not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"coordinates.{axis} must be a finite number")
    if "reconstruction" in config:
        points = config["reconstruction"]["grid_points"]
        if not isinstance(points, int) or isinstance(points, bool) or points < 2:
            raise ValueError("reconstruction.grid_points must be an integer >=2")
        for axis in ["x_limits_m", "y_limits_m"]:
            lo, hi = config["reconstruction"][axis]
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError(f"reconstruction.{axis} must be increasing and finite")
    if "depth_m" in config["analysis"] and config["analysis"]["depth_m"] <= 0:
        raise ValueError("analysis.depth_m must be positive")
    for key in ["diagnostics", "plot_data"]:
        if key in config["plot"] and not isinstance(config["plot"][key], bool):
            raise ValueError(f"plot.{key} must be true or false")
    if "source" in config["plot"] and config["plot"]["source"] not in ["multi", "sadcp", "sat"]:
        raise ValueError("plot.source must be multi, sadcp or sat")
    for key in ["quiver_spacing", "diagnostic_step_index"]:
        value = config["plot"].get(key, 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"plot.{key} must be a nonnegative integer")
    hour = config["analysis"].get("daily_analysis_hour_utc", 6)
    if not isinstance(hour, int) or not 0 <= hour <= 23:
        raise ValueError("daily_analysis_hour_utc must be an integer from 0 to 23")
    return config


def select_snapshot_time(times, requested, tolerance_hours=1):
    """Select a configured date within a stated tolerance; return the actual time."""
    times = np.asarray(times, dtype="datetime64[ns]")
    if not times.size or np.isnat(times).any():
        raise ValueError("No valid fitted timestamps are available for the snapshot")
    if not np.isfinite(tolerance_hours) or tolerance_hours < 0:
        raise ValueError("Snapshot tolerance must be finite and nonnegative")
    target = utc_timestamp(requested).to_datetime64()
    if np.isnat(target):
        raise ValueError("Snapshot date must be a valid UTC timestamp")
    i = np.argmin(np.abs(times-target))
    if abs(times[i]-target) / np.timedelta64(1, "h") > tolerance_hours:
        raise ValueError(f"No fitted snapshot within {tolerance_hours} hours of {requested}")
    return times[i]


def select_analysis_window(data, start=None, end=None):
    """Apply inclusive UTC bounds to an xarray dataset's time coordinate."""
    start = None if start is None else utc_timestamp(start)
    end = None if end is None else utc_timestamp(end)
    if start is not None and end is not None and start > end:
        raise ValueError("Analysis start must not follow analysis end")
    selected = data.sel(time=slice(start, end))
    if not selected.sizes.get("time", 0):
        raise ValueError("Configured analysis window contains no observations")
    return selected


@contextlib.contextmanager
def capture_summary():
    """Keep the existing printed summary visible while collecting its export."""
    buffer = io.StringIO()
    destination = sys.stdout

    class Tee:
        def write(self, value):
            destination.write(value)
            return buffer.write(value)

        def flush(self):
            destination.flush()

    with contextlib.redirect_stdout(Tee()):
        yield buffer


def _json_safe(value):
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        return None if pd.isna(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return str(value) if isinstance(value, Path) else value


def export_application_results(root, config_path, config, results, summary_text,
                               notebook, tables=None, run_info=None):
    """Save the current run's summary, SI parameters and strict-JSON provenance.

    The configured output directory is replaced file-by-file on rerun. Use a
    different output_dir to retain another run. No fits or figure cleanup occur.
    """
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    if json.loads(config_path.read_text()) != config:
        raise RuntimeError("Configuration changed after loading; restart and run all cells before export")
    output = repo_path(root, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    parameters = results.copy()
    if "Q" in parameters:
        matrices = [np.asarray(q, float) for q in parameters.pop("Q")]
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            parameters[f"q{i+1}{j+1}"] = [q[i, j] for q in matrices]
    if "time" in parameters:
        parameters["time"] = pd.to_datetime(parameters.time, utc=True).map(lambda t: t.isoformat())
    files = ["functions.py", "reporting.py", notebook, *config["inputs"].values()]
    hashes = {str(path): hashlib.sha256(repo_path(root, path).read_bytes()).hexdigest() for path in files}
    hashes[str(config_path.relative_to(root)) if config_path.is_relative_to(root) else str(config_path)] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        changes = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL).splitlines()
        git = {"commit": commit, "dirty": bool(changes), "changes": changes}
    except (OSError, subprocess.CalledProcessError):
        git = {"commit": None, "dirty": None, "changes": None}
    metadata = {"schema_version": 1, "application": config["application"],
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "configuration": config, "git": git, "sha256": hashes,
                "environment": {"python": python_platform.python_version(), **{
                    p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scipy", "xarray", "matplotlib", "netCDF4"]}},
                "parameter_units": {"time": "UTC", "xc": "m", "yc": "m", "Rc": "m", "Omega": "s^-1", "w": "s^-1",
                                    "Omega_core": "s^-1", "w_core": "s^-1", "psi0": "m^2/s", "alpha": "1", "qij": "1",
                                    "R2": "velocity reconstruction", "r2_latte": "linear inner-core fit"},
                "run_info": run_info or {},
                "tables": {key: value.to_dict("records") for key, value in (tables or {}).items()},
                "result_rows": len(parameters)}
    headings = ["DATA AND SETTINGS", "NUMERICAL RESULTS", "TRACKING COVERAGE AND FINITE-VALUE RANGES",
                "GEOMETRY CHECK", "OBSERVATION COUNTS", "OUTER-FIT DIAGNOSTICS", "FIGURE AND INTERPRETATION NOTES", "REPRODUCTION INFORMATION", "MANUSCRIPT-ROUNDED VALUES", "PLOTTED SNAPSHOT:"]
    report = [f"# {config['application']} — manuscript results", "",
              "Current run only; compare with the manuscript. Dates are UTC. Parameters are exported in SI units to `parameters.csv`.", "",
              f"Generated: {metadata['generated_at_utc']}. Git revision: `{git['commit']}`; working tree dirty: `{git['dirty']}`.", "",
              "Configuration, versions, hashes, counts and fit diagnostics are in `run_metadata.json`.", ""]
    block = []

    def flush():
        if any(line.strip() for line in block):
            report.extend(["```text", *block, "```", ""])
        block.clear()

    for line in summary_text.splitlines():
        if line and set(line) == {"="}:
            continue
        if line.startswith("MANUSCRIPT RESULTS SUMMARY") or any(line.startswith(h) for h in headings):
            flush()
            report.extend(["## " + line, ""])
        else:
            block.append(line)
    flush()
    report.extend(["## Configuration used", "", "```json", json.dumps(config, indent=2, allow_nan=False), "```", ""])
    parameters.to_csv(output / "parameters.csv", index=False, float_format="%.17g")
    (output / "manuscript_summary.md").write_text("\n".join(report) + "\n")
    (output / "run_metadata.json").write_text(json.dumps(_json_safe(metadata), indent=2, allow_nan=False) + "\n")
    print(f"Results exported to {output}")
    return output


def platform_counts(platform, mask, x, y, u, v):
    """Count finite velocity/position pairs passed to a fitting stage."""
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(u) & np.isfinite(v)
    selected = np.asarray(platform)[np.asarray(mask) & finite]
    return {str(p): int(np.count_nonzero(selected == p)) for p in np.unique(platform)}


def fit_record(time, fit, source="combined"):
    """Compact diagnostics from an existing outer fit, never a second fit."""
    return {"time": time, "source": source,
            "outer_fit_n": len(fit.get("rho", [])),
            "outer_fit_status": fit.get("status", "unavailable"),
            "R2_tangential": fit.get("R2", np.nan),
            "Rc_standard_error_m": fit.get("Rc_standard_error", np.nan)}


def evaluation_counts(data):
    """Count finite pairs actually available to the velocity R² evaluation."""
    keys = (["time"] if "time" in data else []) + (["source"] if "source" in data else [])
    if data.empty:
        return pd.DataFrame(columns=keys + ["velocity_R2_n"])
    finite = np.isfinite(data[["xi", "yi", "ui", "vi"]].to_numpy(float)).all(axis=1)
    counted = data[keys].copy()
    counted["velocity_R2_n"] = finite.astype(int)
    return (counted.groupby(keys, as_index=False)["velocity_R2_n"].sum() if keys
            else pd.DataFrame({"velocity_R2_n": [int(finite.sum())]}))


def _table(data, precision=9):
    print(data.to_string(index=False, float_format=lambda x: f"{x:.{precision}g}", na_rep="unavailable"))


def _number(value, spec):
    return format(float(value), spec) if np.isfinite(value) else "unavailable"


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return type(value)(_plain(v) for v in value)
    return value


def print_manuscript_summary(title, results, settings, counts=None, outer_records=None,
                             snapshot_time=None, notes=(), input_files=(), root=None):
    """Print settings, manuscript units/rounding, numerical results and provenance.

    `results` contains stored SI parameters. Time-series extrema are computed
    over finite values only; unavailable full-field results remain unavailable.
    All displayed dates are UTC labels. `counts` describes stages explicitly.
    """
    print("\n" + "=" * 88 + "\nMANUSCRIPT RESULTS SUMMARY — " + title + "\n" + "=" * 88)
    print("Current run; compare with the manuscript after any analysis change. All times UTC.")
    print("\nDATA AND SETTINGS")
    for label, value in settings.items():
        print(f"  {label}: {_plain(value)}")
    data = results.copy()
    if data.empty:
        print("\nNo accepted fits; parameter summaries unavailable.")
    else:
        data["time"] = pd.to_datetime(data.time)
        if "source" not in data:
            data["source"] = "combined"
        cols = {"xc": ("x0_km", 1e3), "yc": ("y0_km", 1e3),
                "w": ("omega_s-1", 1), "Omega": ("Omega_s-1", 1),
                "alpha": ("alpha", 1), "Rc": ("Rc_km", 1e3),
                "psi0": ("psi0_m2_s-1", 1), "R2": ("R2_velocity", 1)}
        table = data[["source", "time"]].copy()
        for old, (new, divisor) in cols.items():
            table[new] = pd.to_numeric(data[old], errors="coerce") / divisor if old in data else np.nan
        if "r2_latte" in data:
            table["R2_linear_core"] = data.r2_latte.to_numpy()
        if "Omega_core" in data:
            table["Omega_core_s-1"] = data.Omega_core.to_numpy()
            table["omega_core_fit_s-1"] = data.w_core.to_numpy()
        if "fit_status" in data:
            table["fit_status"] = data.fit_status.to_numpy()
        if snapshot_time is not None:
            selected = table[table.time == pd.Timestamp(snapshot_time)]
            print(f"\nPLOTTED SNAPSHOT: {snapshot_time} UTC")
            if selected.empty:
                print("  No fitted row at this exact plotted time.")
        else:
            selected = table if len(table) <= 3 else table.iloc[[0, -1]]
        print("\nMANUSCRIPT-ROUNDED VALUES (final refined Omega; omega = Omega tr(Q))")
        for _, row in selected.iterrows():
            print(f"  {row.source}, {row.time} UTC: (x0,y0)=({_number(row.x0_km, '.1f')}, {_number(row.y0_km, '.1f')}) km; "
                  f"omega={_number(row['omega_s-1'], '.1e')} s^-1; Omega={_number(row['Omega_s-1'], '.1e')} s^-1; "
                  f"alpha={_number(row.alpha, '.2f')}; Rc={_number(row.Rc_km, '.1f')} km; "
                  f"psi0={_number(row['psi0_m2_s-1'], '.6g')} m^2/s; R²_velocity={_number(row.R2_velocity, '.2f')}")
        print("\nNUMERICAL RESULTS (additional precision, not physical measurement accuracy)")
        _table(table)
        print("\nTRACKING COVERAGE AND FINITE-VALUE RANGES")
        for source, group in table.groupby("source", sort=False):
            print(f"  {source}: {len(group)} fitted timesteps; {group.time.min()} to {group.time.max()} UTC")
            if len(group) > 1:
                ranges = []
                for column in table.columns[2:]:
                    if column == "fit_status":
                        continue
                    valid = group[np.isfinite(group[column])]
                    if valid.empty:
                        continue
                    lo, hi = valid[column].idxmin(), valid[column].idxmax()
                    ranges.append({"quantity": column, "min": valid.loc[lo, column],
                                   "at_min_UTC": valid.loc[lo, "time"], "median": valid[column].median(),
                                   "max": valid.loc[hi, column], "at_max_UTC": valid.loc[hi, "time"],
                                   "finite_n": len(valid)})
                if ranges:
                    _table(pd.DataFrame(ranges))
        print("\nGEOMETRY CHECK")
        if "Q" in data:
            valid_q = []
            for q in data.Q:
                q = np.asarray(q, dtype=float)
                valid_q.append(q.shape == (2, 2) and np.isfinite(q).all()
                               and np.allclose(q, q.T) and np.linalg.eigvalsh(q).min() > 0
                               and np.isclose(np.linalg.det(q), 1, rtol=1e-6, atol=1e-8))
            print(f"  Symmetric positive-definite, determinant-one Q: {sum(valid_q)}/{len(valid_q)} rows.")
    if counts is not None and len(counts):
        print("\nOBSERVATION COUNTS (outer-fit count includes retained inner-core observations)")
        _table(pd.DataFrame(counts))
    if outer_records:
        outer = pd.DataFrame(outer_records)
        print("\nOUTER-FIT DIAGNOSTICS (R2_tangential differs from manuscript velocity R²)")
        _table(outer)
        print("  Status counts:", outer.outer_fit_status.value_counts().to_dict())
        print("  Failed fits return unavailable parameters (NaN); initial guesses are never substituted.")
        if (outer.outer_fit_status != "optimized").any():
            print("  Failed outer-fit rows retain valid inner geometry where available, but have no Gaussian reconstruction.")
    print("\nFIGURE AND INTERPRETATION NOTES")
    print("  x0/y0 are relative to the stated local origin; Rc is the Gaussian characteristic radius.")
    print("  Peak normalized tangential speed occurs at rho=Rc/sqrt(2), not at Rc.")
    print("  R2_velocity evaluates reconstructed u/v on the notebook's evaluation rows;")
    print("  R2_tangential evaluates the normalized tangential fit on its filtered rows.")
    for note in notes:
        print("  " + note)
    print("\nREPRODUCTION INFORMATION")
    packages = ["numpy", "pandas", "scipy", "xarray", "matplotlib", "netCDF4"]
    print("  Python", python_platform.python_version(), ";", ", ".join(f"{p}={importlib.metadata.version(p)}" for p in packages))
    for path in dict.fromkeys(["functions.py", "reporting.py", *input_files]):
        p = repo_path(root or Path.cwd(), path)
        print(f"  SHA256 {path}: {hashlib.sha256(p.read_bytes()).hexdigest()}" if p.is_file() else f"  File unavailable: {path}")
