"""Figures for independent SWOT/LATTE applications."""
import numpy as np
import matplotlib.pyplot as plt
from functions import model_uv_at_xy, plot_outer_core_fit


def plot_application(result):
    """Observed and reconstructed speed at valid observed locations only."""
    data, settings, rec = result['data'], result['config'], result['record']
    x, y, u, v = (data[k] for k in ['xi', 'yi', 'ui', 'vi'])
    opts = settings['plot']
    step = opts['quiver_stride']
    if not isinstance(step, int) or step < 1:
        raise ValueError('quiver_stride must be a positive integer')
    success = rec['outer_status'] == 'optimized'
    uh, vh = (model_uv_at_xy(x, y, rec['xc_m'], rec['yc_m'], result['Q'],
                            rec['Omega_s_1'], rec['Rc_km']*1000) if success
              else (np.full_like(u, np.nan), np.full_like(v, np.nan)))
    speeds = np.concatenate([np.hypot(u,v), np.hypot(uh,vh)])
    finite = speeds[np.isfinite(speeds)]
    vmax = float(finite.max()) if finite.size else 1.0
    fig, axes = plt.subplots(1, 2, figsize=opts['figure_size'], constrained_layout=True)
    fig.suptitle(result['label'])
    for i, (ax, U, V, title) in enumerate(zip(axes, [u,uh], [v,vh],
                                  ['Observed geostrophic flow', 'LATTE reconstruction'])):
        if i == 0 and not x.size:
            ax.text(.5, .5, 'No valid observed vectors in configured region',
                    ha='center', va='center', transform=ax.transAxes)
        if i == 0 or success:
            h = ax.scatter(x/1000, y/1000, c=np.hypot(U,V), s=opts['scatter_size'],
                           cmap=opts['cmap'], vmin=0, vmax=vmax)
            ax.quiver(x[::step]/1000, y[::step]/1000, U[::step], V[::step],
                      scale=opts['quiver_scale'])
        else:
            ax.text(.5, .5, f"Reconstruction unavailable\n{rec['inner_status']} / {rec['outer_status']}",
                    ha='center', va='center', transform=ax.transAxes)
        seed = settings['latte']['initial_center_m']
        if opts.get('show_starting_centre', True):
            ax.scatter(seed[0]/1000, seed[1]/1000, marker='+',
                       c=opts.get('starting_centre_color', 'magenta'), label='Starting centre')
        if rec['inner_status'] == 'optimized':
            ax.scatter(rec['xc_m']/1000, rec['yc_m']/1000, marker='x', c='red', label='Centre')
            values, vectors = np.linalg.eigh(result['Q'])
            theta = np.linspace(0, 2*np.pi, 200)
            refined = rec['inner_mode'] == 'refined'
            selection_centre = result['inner']['preliminary_center'] if refined else seed
            selection_Q = result['inner']['preliminary_Q'] if refined else np.asarray(settings['latte']['initial_Q'])
            boundaries = [
                (settings['latte']['inner_radius_m'], ':', 'red',
                 'Refined inner selection' if refined else 'Initial inner selection (single-stage)',
                 selection_centre, selection_Q),
                (settings['latte']['outer_radius_m'], '--', 'red', 'Outer selection',
                 (rec['xc_m'], rec['yc_m']), result['Q'])]
            if not opts.get('show_selection_boundaries', True):
                boundaries = []
            if success:
                boundaries.append((rec['Rc_km']*1000/np.sqrt(2), '-', 'darkorange',
                                   'Max tangential velocity',
                                   (rec['xc_m'], rec['yc_m']), result['Q']))
            for radius, style, colour, label, centre, geometry in boundaries:
                values, vectors = np.linalg.eigh(geometry)
                ellipse = vectors @ (radius/np.sqrt(values)[:,None]*np.array([np.cos(theta),np.sin(theta)]))
                ax.plot((ellipse[0]+centre[0])/1000, (ellipse[1]+centre[1])/1000,
                        style, color=colour, label=label)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc='upper right')
        ax.set(title=title, xlabel='East (km)', ylabel='North (km)',
               aspect='equal', xlim=settings['selection']['extent_km'][:2],
               ylim=settings['selection']['extent_km'][2:])
    if finite.size:
        fig.colorbar(h, ax=list(axes), label='Speed (m/s)')
    return fig


def plot_radial_diagnostic(result):
    """Optional radial diagnostic, without independence-based uncertainty bands."""
    fig, ax = plt.subplots(figsize=(6,5), constrained_layout=True)
    if result['outer'].get('status') == 'optimized':
        plot_outer_core_fit(result['outer'], ax=ax, km_flag=True, ci_flag=False,
                            pred_flag=False, n_curve=result['config']['plot']['radial_curve_points'])
    else:
        ax.text(.5,.5,'Radial fit unavailable',ha='center',transform=ax.transAxes)
    ax.set_title(result['label'])
    return fig


def show_application(result):
    """Display the figure followed immediately by the rendered Markdown summary."""
    from IPython.display import display, Markdown
    from swot_processing import summary_markdown
    fig = plot_application(result)
    display(fig)
    plt.close(fig)
    display(Markdown(summary_markdown(result)))
