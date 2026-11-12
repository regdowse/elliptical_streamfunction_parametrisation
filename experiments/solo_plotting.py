"""SOLO manuscript figures drawn exclusively from saved observations and fits."""
from datetime import datetime, timezone
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from experiments.outputs import sha256
from manuscript_figures import save_manuscript_figure


def field_axes(ax, data, config):
    field = data["field"]
    x,y = field["x_m"]/1000,field["y_m"]/1000
    u,v = field["u_m_s"],field["v_m_s"]
    image = ax.pcolormesh(x,y,np.hypot(u,v).T,cmap="Blues_r",shading="auto",rasterized=True)
    s = config["plot"]["quiver_stride"]
    ax.quiver(x[::s],y[::s],u[::s,::s].T,v[::s,::s].T,
        scale=config["plot"]["quiver_scale"],alpha=.22,width=.002)
    truth = config["truth"]
    theta = np.linspace(0,2*np.pi,361)
    radius = truth["rc_m"]/np.sqrt(2)/1000
    ax.plot(truth["xc_m"]/1000+radius*np.cos(theta),truth["yc_m"]/1000+radius*np.sin(theta),"k--",lw=1)
    ax.axhline(truth["yc_m"]/1000,color="k",ls="--",lw=.5,alpha=.5)
    ax.axvline(truth["xc_m"]/1000,color="k",ls="--",lw=.5,alpha=.5)
    ax.set(xlim=config["plot"]["limits_km"],ylim=config["plot"]["limits_km"],xlabel="x (km)",ylabel="y (km)")
    ax.set_aspect("equal")
    return image


def plot_cases(data, config):
    """Original six-panel SOLO layout, drawing stored inner-core linearisations."""
    from matplotlib.patches import Rectangle
    import matplotlib.gridspec as gridspec
    field = data["field"]
    x,y = field["x_m"]/1000,field["y_m"]/1000
    X,Y = np.meshgrid(x,y,indexing="ij")
    u,v = field["u_m_s"],field["v_m_s"]
    truth = config["truth"]
    Rc = truth["rc_m"]/1000
    rho2 = (X-truth["xc_m"]/1000)**2+(Y-truth["yc_m"]/1000)**2
    df_res = data["parameters"].set_index("case").copy()
    df_res["xc"],df_res["yc"],df_res["Rc"] = df_res.xc_m/1000,df_res.yc_m/1000,df_res.Rc_m/1000
    df_res["q11"],df_res["q12"],df_res["q22"] = 1.,0.,1.
    dic_tests = {}
    for name in df_res.index:
        curve = data["curves"][name]
        x1,y1 = np.asarray(curve["x_m"])/1000,np.asarray(curve["y_m"])/1000
        dic_tests[name] = dict(name=name,x1=x1,y1=y1,l=x1-x1[0],
            u1=np.asarray(curve["u_m_s"]),v1=np.asarray(curve["v_m_s"]),
            UU=np.asarray(curve["tangent_u"]),VV=np.asarray(curve["tangent_v"]))
    def extract_number(name):
        return int(name[-1])
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(2, 6, figure=fig, width_ratios=[1, 1, 2, 1, 1, 2])
    ax_main_left = fig.add_subplot(gs[:, :2])
    pc = ax_main_left.pcolor(x, y, np.hypot(u, v), cmap='Blues_r', zorder=0)
    plt.colorbar(pc, ax=ax_main_left, orientation='horizontal', location='top', label='Current speed (ms$^{-1}$)')
    ax_main_left.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
    ax_main_left.axvline(0, color='black', linewidth=0.8, linestyle='--', zorder=0)
    spacing = 2 * 5
    scale = 40
    width = 0.002
    alpha = 1
    quiv = ax_main_left.quiver(x[::spacing], y[::spacing], u[::spacing, ::spacing].T, v[::spacing, ::spacing].T, alpha=0.2, scale=scale, width=width, zorder=1)
    ax_main_left.contour(X, Y, rho2, levels=[Rc ** 2 / 2], colors='k', linestyles='--')
    colors = ['limegreen', 'r', 'b']
    color = ['g', 'r', 'b']
    color_fits = [(0, 0.9, 0, 0.8), (1, 0, 0, 0.6), (0, 0, 1, 0.6)]
    i = 0
    for dic in dic_tests.values():
        if 'P' in dic['name']:
            (x1, y1, l, u1, v1) = (dic['x1'], dic['y1'], dic['l'], dic['u1'], dic['v1'])
            data = df_res.loc[dic['name']]
            ax_main_left.text(x1[0] - 20, y1[0], dic['name'], color=color[i], zorder=3, fontsize=18, fontweight='bold')
            ax_main_left.quiver(x1, y1, u1, v1, scale=scale, width=width * 2, color=color[i], zorder=3)
            ax_main_left.scatter(data.xc, data.yc, color=colors[i], marker='x', s=70, zorder=10 - extract_number(dic['name']))
            (dx_ell, dy_ell) = (X - data.xc, Y - data.yc)
            rho2_ell = data.q11 * dx_ell ** 2 + 2 * data.q12 * dx_ell * dy_ell + data.q22 * dy_ell ** 2
            ax_main_left.contour(X, Y, rho2_ell, levels=[data.Rc ** 2 / 2], colors=color[i], linestyles='--')
            ax_main_left.text(10, -100 + 10 * i, f'$(x_0,y_0)=({np.abs(data.xc):.2f},{data.yc:.2f})$', color=color[i], zorder=10)
            i += 1
    (x_pos, y_pos) = (0.75, 0.1)
    box_width = 0.45
    box_height = 0.16
    rect = Rectangle((x_pos - box_width / 2, y_pos - box_height / 2), box_width, box_height, transform=ax_main_left.transAxes, facecolor='white', edgecolor='black', zorder=9)
    ax_main_left.add_patch(rect)
    ax_main_left.axis('equal')
    ax_main_left.set_xlim(-100, 100)
    ax_main_left.set_ylim(-100, 100)
    ax_main_left.set_xlabel('x (km)')
    ax_main_left.set_ylabel('y (km)')
    ax_main_left.text(-0.1, 1.05, 'a)', transform=ax_main_left.transAxes, fontsize=12, fontweight='bold')
    ax2_left = fig.add_subplot(gs[0, 2])
    ax3_left = fig.add_subplot(gs[1, 2])
    j = 0
    for dic in dic_tests.values():
        if 'P' in dic['name']:
            (x1, y1, l, u1, v1, UU, VV) = (dic['x1'], dic['y1'], dic['l'], dic['u1'], dic['v1'], dic['UU'], dic['VV'])
            data = df_res.loc[dic['name']]
            ax2_left.plot(x1, v1, color=color_fits[j], linewidth=3, zorder=-extract_number(dic['name']) - 3)
            ax2_left.plot(x1, VV, color=color[j], linestyle='--', linewidth=2, zorder=-extract_number(dic['name']))
            ax3_left.plot(x1, u1, color=color_fits[j], linewidth=3, zorder=-extract_number(dic['name']) - 3)
            ax3_left.plot(x1, UU, color=color[j], linestyle='--', linewidth=2, zorder=-extract_number(dic['name']))
            j += 1
    ax2_left.axvline(x1[len(x1) // 2], color='black', linewidth=0.8, linestyle='--', zorder=2)
    ax2_left.axhline(y=0, color='k', zorder=-1, linewidth=0.5, alpha=0.5)
    ax2_left.set_ylabel('$V$ (ms$^{-1}$)')
    ax2_left.set_xlabel('x (km)')
    ax2_left.text(-0.1, 1.05, 'b)', transform=ax2_left.transAxes, fontsize=12, fontweight='bold')
    ax3_left.axvline(x1[len(x1) // 2], color='black', linewidth=0.8, linestyle='--', zorder=2)
    ax3_left.axhline(y=0, color='k', zorder=-1, linewidth=0.5, alpha=0.5)
    ax3_left.set_ylabel('$U$ (ms$^{-1}$)')
    ax3_left.set_xlabel('x (km)')
    ax3_left.text(-0.1, 1.05, 'c)', transform=ax3_left.transAxes, fontsize=12, fontweight='bold')
    ax_main_right = fig.add_subplot(gs[:, 3:5])
    pc = ax_main_right.pcolor(x, y, np.hypot(u, v), cmap='Blues_r')
    plt.colorbar(pc, ax=ax_main_right, orientation='horizontal', location='top', label='Current speed (ms$^{-1}$)')
    ax_main_right.axhline(0, color='black', linewidth=0.8, linestyle='--', zorder=2)
    ax_main_right.axvline(0, color='black', linewidth=0.8, linestyle='--', zorder=2)
    quiv = ax_main_right.quiver(x[::spacing], y[::spacing], u[::spacing, ::spacing].T, v[::spacing, ::spacing].T, alpha=0.2, scale=scale, width=width)
    ax_main_right.contour(X, Y, rho2, levels=[Rc ** 2 / 2], colors='k', linestyles='--')
    i = 0
    for dic in dic_tests.values():
        if 'L' in dic['name']:
            (x1, y1, l, u1, v1) = (dic['x1'], dic['y1'], dic['l'], dic['u1'], dic['v1'])
            data = df_res.loc[dic['name']]
            ax_main_right.quiver(x1, y1, u1, v1, scale=scale, width=width * 2, color=color[i], zorder=5 - i)
            ax_main_right.scatter(data.xc, data.yc, color=colors[i], marker='x', s=70, zorder=10 - extract_number(dic['name']))
            ax_main_right.text(10, -100 + 10 * i, f'$(x_0,y_0)=({np.abs(data.xc):.2f},{data.yc:.2f})$', color=color[i], zorder=10)
            (dx_ell, dy_ell) = (X - data.xc, Y - data.yc)
            rho2_ell = data.q11 * dx_ell ** 2 + 2 * data.q12 * dx_ell * dy_ell + data.q22 * dy_ell ** 2
            ax_main_right.contour(X, Y, rho2_ell, levels=[data.Rc ** 2 / 2], colors=color[i], linestyles='--')
            ax_main_right.text(x1[0] + (-i * 2 + 10), y1[0] + 20, dic['name'], color=color[i], zorder=3, fontsize=18, fontweight='bold')
            i += 1
    (x_pos, y_pos) = (0.75, 0.1)
    box_width = 0.45
    box_height = 0.16
    rect = Rectangle((x_pos - box_width / 2, y_pos - box_height / 2), box_width, box_height, transform=ax_main_right.transAxes, facecolor='white', edgecolor='black', zorder=9)
    ax_main_right.add_patch(rect)
    ax_main_right.axis('equal')
    ax_main_right.set_xlim(-100, 100)
    ax_main_right.set_ylim(-100, 100)
    ax_main_right.set_xlabel('x (km)')
    ax_main_right.set_ylabel('y (km)')
    ax_main_right.text(-0.1, 1.05, 'd)', transform=ax_main_right.transAxes, fontsize=12, fontweight='bold')
    ax2_right = fig.add_subplot(gs[0, 5])
    ax3_right = fig.add_subplot(gs[1, 5])
    j = 0
    for dic in dic_tests.values():
        if 'L' in dic['name']:
            (x1, y1, l, u1, v1, UU, VV) = (dic['x1'], dic['y1'], dic['l'], dic['u1'], dic['v1'], dic['UU'], dic['VV'])
            data = df_res.loc[dic['name']]
            ax2_right.plot(x1, v1, color=color_fits[j], linewidth=3, zorder=-extract_number(dic['name']))
            ax2_right.plot(x1, VV, color=color_fits[j], linestyle='--', linewidth=2, zorder=3 - extract_number(dic['name']))
            ax3_right.plot(x1, u1, color=color_fits[j], linewidth=3, zorder=-extract_number(dic['name']))
            ax3_right.plot(x1, UU, color=color_fits[j], linestyle='--', linewidth=2, zorder=3 - extract_number(dic['name']))
            j += 1
    ax2_right.axvline(x1[len(x1) // 2], color='black', linewidth=0.8, linestyle='--', zorder=2)
    ax2_right.axhline(y=0, color='k', zorder=-1, linewidth=0.5, alpha=0.5)
    ax2_right.set_ylabel('$V$ (ms$^{-1}$)')
    ax2_right.text(-0.1, 1.05, 'e)', transform=ax2_right.transAxes, fontsize=12, fontweight='bold')
    ax2_right.set_xlabel('x (km)')
    ax3_right.axvline(x1[len(x1) // 2], color='black', linewidth=0.8, linestyle='--', zorder=2)
    ax3_right.axhline(y=0, color='k', zorder=-1, linewidth=0.5, alpha=0.5)
    ax3_right.set_ylabel('$U$ (ms$^{-1}$)')
    ax3_right.set_xlabel('x (km)')
    ax3_right.text(-0.1, 1.05, 'f)', transform=ax3_right.transAxes, fontsize=12, fontweight='bold')
    ax3_right.set_xlabel('x (km)')
    axs = [ax2_left, ax3_left, ax2_right, ax3_right]
    (ymin, ymax) = (np.inf, -np.inf)
    for ax in axs:
        ymin = min(ymin, ax.get_ylim()[0])
        ymax = max(ymax, ax.get_ylim()[1])
    (ymin, ymax) = (-3, 3)
    for ax in axs:
        ax.set_ylim(ymin, ymax)
    plt.tight_layout()
    return fig


def plot_outer(data, config):
    """Legacy outer-fit styling applied to the current stored fit parameters."""
    fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(10, 4),
                            sharey=True, sharex=True)
    radius = np.linspace(0, config["plot"]["outer_radius_limit_km"]*1000, 400)
    colors = ['g', 'r', 'b']*2
    for i, (ax, spec, color) in enumerate(zip(axs.flat, config["cases"], colors)):
        name = spec["name"]
        fit = data["curves"][name]["outer"]
        rho, vt = np.asarray(fit["rho"], float), np.asarray(fit["vt"], float)
        ax.scatter(rho/1000, np.abs(vt), s=40, label='Observed', marker='x',
                   zorder=2, color=color)
        if fit["status"] == "optimized":
            speed = abs(fit["Omega"])*radius*np.exp(-radius**2/fit["Rc"]**2)
            # Preserve the original convention: fade beyond the outermost datum.
            mask = radius <= rho.max()
            ax.plot(radius[mask]/1000, speed[mask], label='Fit', lw=2,
                    color='c', zorder=1)
            ax.plot(radius[~mask]/1000, speed[~mask], label='Fit', lw=2,
                    color='c', zorder=1, alpha=.3)
            ax.axvline(fit["Rc"]/np.sqrt(2)/1000, ls='--',
                       label=r'$\rho_{\max}$', lw=2, color=color)
        else:
            ax.text(.03, .95, fit["status"], transform=ax.transAxes, va='top', fontsize=8)
        ax.set_title(name)
        ax.set_xlabel(r'$\rho$ (km)')
        ax.axvline(config["truth"]["rc_m"]/np.sqrt(2)/1000, color='k',
                   zorder=-1, alpha=.5)
        ax.set_ylabel(r'$|v_t^\star|$ (ms$^{-1}$)' if i in (0, 3) else '')
    return fig


def plot_comparison(data, config):
    """Original conceptual illustration layout, using saved model inputs/guides."""
    from matplotlib.gridspec import GridSpec
    field = data["field"]
    X,Y = np.meshgrid(field["x_m"]/1000,field["y_m"]/1000,indexing="ij")
    u,v = field["u_m_s"],field["v_m_s"]
    curve = data["curves"]["comparison"]
    x1,y1 = np.asarray(curve["x_m"])/1000,np.asarray(curve["y_m"])/1000
    u1,v1 = np.asarray(curve["u_m_s"]),np.asarray(curve["v_m_s"])
    rows = data["comparison"].set_index("method")
    x0,y0 = rows.loc["SOLO",["xc_m","yc_m"]].to_numpy()/1000
    x01,y01 = rows.loc["MOCA",["xc_m","yc_m"]].to_numpy()/1000
    scale,width,spac = 30,0.002,10
    fig = plt.figure(figsize=(14, 4))
    gs = GridSpec(1, 4, width_ratios=[0.05, 1, 1, 1], figure=fig)
    cax = fig.add_subplot(gs[0, 0])
    axs = [fig.add_subplot(gs[0, i]) for i in range(1, 4)]
    im = axs[0].pcolor(X, Y, np.hypot(u, v), cmap='Blues_r')
    fig.colorbar(im, cax=cax, location='left', label='Current speed (ms$^{-1}$)')
    axs[0].quiver(X[::spac, ::spac], Y[::spac, ::spac], u[::spac, ::spac], v[::spac, ::spac], scale=scale, width=width, alpha=0.2)
    axs[0].quiver(x1[::spac], y1[::spac], u1[::spac], v1[::spac], color='m', scale=scale, width=width * 3)
    axs[0].scatter(x0, y0, marker='x', color='limegreen', s=100, label='SOLO')
    axs[0].scatter(x01, y01, marker='x', color='r', s=100, label='MOCA')
    axs[0].set_xlabel('x (km)')
    axs[0].set_ylabel('y (km)')
    axs[0].axis('equal')
    axs[0].legend(loc='lower right')
    axs[0].set_xlim(-100, 100)
    axs[0].set_ylim(-100, 100)
    axs[0].text(-0.2, 1.1, 'a)', transform=axs[0].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    axs[1].plot(x1, u1, 'm', lw=2)
    axs[1].plot(x1, np.asarray(curve['moca_u']), ls='--', color='r', lw=2)
    axs[1].plot(x1, np.asarray(curve["illustration_u"]), ls='--', color='g', lw=2)
    axs[1].axvline(x=0, ls='--', color='k', alpha=0.5)
    axs[1].axhline(y=0, lw=0.5, color='k', alpha=0.5)
    axs[1].set_xlabel('x (km)')
    axs[1].set_ylabel('$U$ (ms$^{-1}$)')
    axs[1].text(-0.1, 1.1, 'b)', transform=axs[1].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    axs[2].plot(x1, v1, 'm', lw=2)
    axs[2].plot(x1, np.asarray(curve['moca_v']), ls='--', color='r', lw=2)
    axs[2].plot(x1, np.asarray(curve["illustration_v"]), ls='--', color='g', lw=2)
    axs[2].axvline(x=0, ls='--', color='k', alpha=0.5)
    axs[2].axhline(y=0, lw=0.5, color='k', alpha=0.5)
    axs[2].set_ylim(-3, 3)
    axs[2].set_xlabel('x (km)')
    axs[2].set_ylabel('$V$ (ms$^{-1}$)')
    axs[2].text(-0.1, 1.1, 'c)', transform=axs[2].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    plt.tight_layout()
    return fig


def plot_deformation(data, config):
    """Legacy deformation layout, with current centre errors and requested label."""
    frame = data["deformation"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    colors = plt.cm.cool(np.linspace(0, 1, len(frame)))
    axs[1].plot(frame.aspect_ratio, frame.centre_error_m/1000, lw=.5,
                color='k', alpha=.5)
    for ar, err, color in zip(frame.aspect_ratio, frame.centre_error_m/1000, colors):
        axs[1].plot(ar, err, marker='x', color=color)
    axs[1].set_xlabel('Aspect ratio')
    axs[1].set_ylabel('Center error (km)')

    limits = np.asarray(config["plot"]["grid_limits_m"])/1000
    axis = np.linspace(*limits, config["plot"]["grid_points"])
    X, Y = np.meshgrid(axis, axis, indexing='ij')
    truth = config["truth"]
    dx, dy = X-truth["xc_m"]/1000, Y-truth["yc_m"]/1000
    for row, color in zip(frame.itertuples(), colors):
        rho2 = row.Q11*dx**2 + 2*row.Q12*dx*dy + row.Q22*dy**2
        axs[0].contour(X, Y, rho2, levels=[(truth["rc_m"]/1000)**2/2],
                       colors=[color], linewidths=1.5)
    for row, color in zip(frame.itertuples(), colors):
        axs[0].plot(row.xc_m/1000, row.yc_m/1000, marker='x', color=color)

    spec = config["deformation"]["transect"]
    count = int(round(2*spec["half_length_m"]/spec["step_m"]))+1
    x1 = np.linspace(spec["x_mid_m"]-spec["half_length_m"],
                     spec["x_mid_m"]+spec["half_length_m"], count)/1000
    y1 = np.full_like(x1, spec["y_m"]/1000)
    axs[0].plot(x1, y1, color='g', lw=2)
    axs[0].axis('equal')
    axs[0].set_xlim(*config["plot"]["limits_km"])
    axs[0].set_ylim(*config["plot"]["limits_km"])
    axs[0].set_xlabel('x (km)')
    axs[0].set_ylabel('y (km)')
    for ax, label in zip(axs, ('a)', 'b)')):
        ax.text(-.1, 1.05, label, transform=ax.transAxes, fontsize=12, fontweight='bold')
    axs[0].axvline(truth["xc_m"]/1000, color='black', linewidth=.8,
                   linestyle='--', zorder=-1, alpha=.5)
    axs[0].axhline(truth["yc_m"]/1000, color='black', linewidth=.8,
                   linestyle='--', zorder=-1, alpha=.5)
    plt.tight_layout()
    return fig


def render(data, config, directory):
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42})
    figures = {"SOLO":plot_cases,"SOLO_outercore_fits":plot_outer,
               "improved_MOCA":plot_comparison,"SOLO_AR_err":plot_deformation}
    paths = []
    for name,plotter in figures.items():
        fig = plotter(data,config)
        try:
            paths.append(save_manuscript_figure(fig,name,directory,dpi=config["plot"]["dpi"]))
        finally:
            plt.close(fig)
    metadata = dict(utc=datetime.now(timezone.utc).isoformat(),plot={**config["plot"],"formats":["png"]},
        source_sha256=sha256(__file__),run_metadata_sha256=sha256(directory/"run_metadata.json"),
        figures_sha256={os.path.relpath(p,directory):sha256(p) for p in paths})
    (directory/"plot_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return paths
