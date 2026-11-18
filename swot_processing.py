"""SWOT selection and LATTE analysis; no plotting or archived dependencies."""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import xarray as xr
from functions import (latte, is_elliptical_Q, refine_latte, out_core_param_fit, model_uv_at_xy,
                       vector_R2, axis_ratio_from_Q)


def load_config(path):
    return json.loads(Path(path).read_text())


def _merge(base, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def application_config(config, eddy, snapshot):
    """Shared defaults < eddy settings < per-combination overrides."""
    settings = deepcopy({k: v for k, v in config.items()
                         if k not in ['eddies', 'snapshots', 'overrides']})
    _merge(settings, config['eddies'][eddy])
    _merge(settings, config['snapshots'][snapshot])
    _merge(settings, config.get('overrides', {}).get(eddy, {}).get(snapshot, {}))
    settings['label'] = (f"{config['eddies'][eddy]['label']} — {snapshot} — "
                         f"cycle {settings['cycle']}, pass {settings['pass']}")
    return settings


def load_observations(root, config):
    f = config['fields']
    c = config['coordinates']
    s = config['selection']
    stride = s['pixel_stride']
    if not isinstance(stride, int) or stride < 1:
        raise ValueError('pixel_stride must be a positive integer')
    with xr.open_dataset(Path(root) / config['input_file']) as raw:
        provenance = dict(raw.attrs)
        ds = raw[list(dict.fromkeys(f.values()))].isel(
            num_lines=slice(None, None, stride), num_pixels=slice(None, None, stride)).load()
    for key in ['u', 'v']:
        if ds[f[key]].attrs.get('units') not in ['m/s', 'm s-1']:
            raise ValueError('Expected velocities in m/s')
    if ds[f['longitude']].attrs.get('units') != 'degrees_east' or ds[f['latitude']].attrs.get('units') != 'degrees_north':
        raise ValueError('Expected longitude/latitude in degrees')
    lon, lat = ds[f['longitude']].values, ds[f['latitude']].values
    x = (lon-c['origin_longitude_deg'])*c['metres_per_degree']*np.cos(np.deg2rad(c['origin_latitude_deg']))
    y = (lat-c['origin_latitude_deg'])*c['metres_per_degree']
    u, v = ds[f['u']].values, ds[f['v']].values
    xmin, xmax, ymin, ymax = np.asarray(s['extent_km'])*1000
    region = (x >= xmin)&(x <= xmax)&(y >= ymin)&(y <= ymax)
    quality = np.isin(ds[f['quality']].values, s['quality_values'])
    valid = region & quality & np.isfinite(x)&np.isfinite(y)&np.isfinite(u)&np.isfinite(v)
    xi, yi, ui, vi = [a[valid] for a in [x, y, u, v]]
    local_times = ds[f['time']].values[np.any(valid, axis=1)]
    return dict(xi=xi, yi=yi, ui=ui, vi=vi, times=local_times,
                provenance=provenance, source=str(Path(root) / config['input_file']))


def fit_snapshot(data, config, inner_radius, outer_radius):
    xi, yi, ui, vi = (data[k] for k in ["xi", "yi", "ui", "vi"])
    c = config["coordinates"]
    if not (0 < inner_radius < outer_radius):
        raise ValueError('Require 0 < inner_radius < outer_radius')
    opts = config['latte']
    refine = opts.get('refine_inner', True)
    if not isinstance(refine, bool):
        raise ValueError('refine_inner must be a JSON boolean')
    inner = {}
    if refine:
        result = refine_latte(xi, yi, ui, vi, center=tuple(opts['initial_center_m']),
                              Q_pre=np.asarray(opts['initial_Q'], float),
                              rho_core=inner_radius, fit_details=inner)
        used_mask = inner['refined_mask']
    else:
        centre = np.asarray(opts['initial_center_m'], float)
        seed_Q = np.asarray(opts['initial_Q'], float)
        if centre.shape != (2,) or not np.isfinite(centre).all() or not is_elliptical_Q(seed_Q):
            raise ValueError('A finite centre and valid initial ellipse are required')
        dx, dy = xi-centre[0], yi-centre[1]
        selected = (seed_Q[0,0]*dx**2 + 2*seed_Q[0,1]*dx*dy + seed_Q[1,1]*dy**2) <= inner_radius**2
        stage = {}
        result = latte(xi[selected], yi[selected], ui[selected], vi[selected], fit_details=stage)
        inner.update(status=stage['status'], initial_mask=selected,
                     refined_mask=np.zeros(xi.size, dtype=bool))
        used_mask = selected
    inner['used_mask'] = used_mask
    inner['mode'] = 'refined' if refine else 'single-stage'
    xc, yc, w_core, Q, omega_core, inner_r2 = result
    record = dict(inner_radius_km=inner_radius/1000, outer_radius_km=outer_radius/1000,
                  inner_status=inner['status'], outer_status='not_attempted',
                  inner_mode=inner['mode'],
                  initial_inner_samples=int(inner['initial_mask'].sum()),
                  refined_inner_samples=int(used_mask.sum()) if refine else None,
                  xc_m=xc, yc_m=yc, inner_vorticity_s_1=w_core, inner_R2=inner_r2)
    outer = {}
    outer_mask = np.zeros(xi.size, dtype=bool)
    if inner['status'] == 'optimized':
        dx, dy = xi-xc, yi-yc
        rho2 = Q[0,0]*dx**2+2*Q[0,1]*dx*dy+Q[1,1]*dy**2
        outer_mask = rho2 <= outer_radius**2
        Rc, psi0, omega = out_core_param_fit(
            xi[outer_mask], yi[outer_mask], ui[outer_mask], vi[outer_mask], xc, yc, Q,
            Omega0=omega_core, core_mask=used_mask[outer_mask],
            fit_details=outer, **config['outer_fit'])
        uh, vh = model_uv_at_xy(xi[outer_mask], yi[outer_mask], xc, yc, Q, omega, Rc)
        record.update(outer_status=outer['status'], Rc_km=Rc/1000, psi0_m2_s=psi0,
                      Omega_s_1=omega, axis_ratio=axis_ratio_from_Q(Q),
                      centre_longitude=c['origin_longitude_deg']+xc/(c['metres_per_degree']*np.cos(np.deg2rad(c['origin_latitude_deg']))),
                      centre_latitude=c['origin_latitude_deg']+yc/c['metres_per_degree'],
                      cyclonic=bool(omega*np.sin(np.deg2rad(c['origin_latitude_deg'])) > 0) if np.isfinite(omega) else None,
                      inner_samples=int(used_mask.sum()), outer_candidates=int(outer_mask.sum()),
                      radial_fit_samples=len(outer.get('observation_indices', [])),
                      vector_R2=vector_R2(ui[outer_mask], vi[outer_mask], uh, vh),
                      vector_RMSE_m_s=float(np.sqrt(np.mean((ui[outer_mask]-uh)**2+(vi[outer_mask]-vh)**2))))
    return record, Q, inner, outer, outer_mask



def analyse(root, config, eddy, snapshot):
    """Fit this eddy/pass independently, without advancing any tracking state."""
    settings = application_config(config, eddy, snapshot)
    data = load_observations(root, settings)
    record, Q, inner, outer, mask = fit_snapshot(
        data, settings, settings['latte']['inner_radius_m'], settings['latte']['outer_radius_m'])
    return dict(label=settings['label'], config=settings, data=data, record=record,
                Q=Q, inner=inner, outer=outer, outer_mask=mask)


def sensitivity_table(result):
    """Optional selection diagnostic, not part of the main eight-case display."""
    settings = result['config']
    return pd.DataFrame([fit_snapshot(result['data'], settings, ri, ro)[0]
                         for ri in settings['sensitivity']['inner_radii_m']
                         for ro in settings['sensitivity']['outer_radii_m'] if ri < ro])


def summary_markdown(result):
    rec, Q, data = result['record'], result['Q'], result['data']
    text = f"### {result['label']} — summary\n\n"
    if data['times'].size:
        text += f"Local observations (UTC): {data['times'].min()} to {data['times'].max()}.\n\n"
    text += (f"Input vectors: {data['xi'].size}; inner status: `{rec['inner_status']}`; "
             f"radial status: `{rec['outer_status']}`.\n\n")
    c = result['config']['coordinates']
    mode_label = 'single-stage inner fit (refinement disabled)' if rec['inner_mode'] == 'single-stage' else 'two-stage refined inner fit'
    text += f'**Method:** {mode_label}.\n\n'
    refined_count = rec['refined_inner_samples'] if rec['refined_inner_samples'] is not None else 'not applied'
    text += (f"**Starting origin (lon, lat):** {c['origin_longitude_deg']:.5f}°E, "
             f"{c['origin_latitude_deg']:.5f}°N (signed latitude).\n\n"
             f"**Selection thresholds (elliptical radius):** inner {rec['inner_radius_km']:g} km; "
             f"outer {rec['outer_radius_km']:g} km. These are prescribed, not fitted.\n\n"
             f"**Inner samples:** initial {rec['initial_inner_samples']}; refined {refined_count}.\n\n")
    if rec['inner_status'] != 'optimized':
        stage = 'Refined' if rec['inner_status'].startswith('refined_') else 'Initial'
        text += f'{stage} inner fit failed; no outer selection was applied.\n\n'
    if rec['outer_status'] != 'optimized':
        if rec['inner_status'] == 'optimized':
            text += (f"Inner-fit centre: {rec['centre_longitude']:.4f}°E, "
                     f"{rec['centre_latitude']:.4f}°N (signed latitude). "
                     f"Inner vorticity: {rec['inner_vorticity_s_1']:.3e} s⁻¹.\n\n")
        return text + 'Full reconstruction unavailable; no valid full-model parameters reported.'
    omega, Rc = rec['Omega_s_1'], rec['Rc_km']
    maximum_speed = abs(omega) * Rc*1000/np.sqrt(2)*np.exp(-0.5)*np.sqrt(np.linalg.eigvalsh(Q).max())
    direction = 'clockwise' if omega < 0 else 'anticlockwise'
    kind = 'cyclonic' if rec['cyclonic'] else 'anticyclonic'
    return text + f"""- **Centre (lon, lat):** {rec['centre_longitude']:.4f}°E, {abs(rec['centre_latitude']):.4f}°{'S' if rec['centre_latitude'] < 0 else 'N'}.
- **Centre (x, y):** ({rec['xc_m']/1000:.2f}, {rec['yc_m']/1000:.2f}) km east/north of this eddy's configured origin.
- **Size:** Gaussian characteristic radius **Rc = {Rc:.2f} km**, not an observed boundary.
- **Maximum fitted tangential speed:** {maximum_speed:.3f} m/s (full elliptical model).
- **Rotation:** {direction}, {kind}.
- **Aspect ratio (major/minor):** {rec['axis_ratio']:.3f}.
- **Angular-velocity parameter Ω:** {omega:.3e} s⁻¹.
- **Centre vorticity Ω tr(Q):** {omega*np.trace(Q):.3e} s⁻¹.
- **Fit diagnostics:** inner R² = {rec['inner_R2']:.3f}; full vector R² = {rec['vector_R2']:.3f}; vector RMSE = {rec['vector_RMSE_m_s']:.3f} m/s.
- **Samples:** inner {rec['inner_samples']}, outer candidates {rec['outer_candidates']}, retained radial samples {rec['radial_fit_samples']}.

Provisional fit: assess frontal-flow residuals, incomplete coverage and selection sensitivity before interpreting it physically.
"""
