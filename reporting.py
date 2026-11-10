"""Read-only reporting for the four real-data ESP application notebooks.

All values come from the current run. This module does not fit models, change
selections, read a manuscript, or assume agreement with published numbers.
"""
import hashlib
import importlib.metadata
from pathlib import Path
import platform as python_platform

import numpy as np
import pandas as pd


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
            "R2_tangential": fit.get("R2", np.nan)}


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
                             snapshot_time=None, notes=(), input_files=()):
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
        print("  Fallback status means returned guesses, not a successful optimization.")
        if (outer.outer_fit_status != "optimized").any():
            print("  Parameter tables/ranges include those returned fallback rows; consult their status before citing them.")
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
        p = Path(path)
        print(f"  SHA256 {p}: {hashlib.sha256(p.read_bytes()).hexdigest()}" if p.is_file() else f"  File unavailable: {p}")
