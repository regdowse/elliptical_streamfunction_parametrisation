"""Render saved DOPPIO/LATTE validations and the illustrative ESP schematic."""
from datetime import datetime, timezone
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.outputs import sha256
from manuscript_figures import save_manuscript_figure


def ellipse(ax,xc,yc,Q,radius,**style):
    Q = np.asarray(Q,float)
    if Q.shape != (2,2) or not np.isfinite([xc,yc,radius]).all() or not np.isfinite(Q).all() or radius <= 0:
        return
    values,vectors = np.linalg.eigh(Q)
    if np.any(values <= 0):
        return
    theta = np.linspace(0,2*np.pi,361)
    xy = vectors@np.diag(radius/np.sqrt(values))@np.array([np.cos(theta),np.sin(theta)])
    xy += np.array([xc,yc])[:,None]
    ax.plot(*(xy/1000),**style)


def background(ax,data,config):
    f = data["field"];p = config["plot"];t = config["truth"]
    x,y = f["x_m"]/1000,f["y_m"]/1000
    image = ax.pcolormesh(x,y,np.hypot(f["u_m_s"],f["v_m_s"]).T,cmap="Blues_r",shading="auto",rasterized=True)
    s = p["quiver_stride"]
    ax.quiver(x[::s],y[::s],f["u_m_s"][::s,::s].T,f["v_m_s"][::s,::s].T,
        alpha=.22,scale=p["quiver_scale"],width=.002)
    Q = np.asarray(t["Q_input"],float);Q /= np.sqrt(np.linalg.det(Q))
    ellipse(ax,t["xc_m"],t["yc_m"],Q,t["rc_m"]/np.sqrt(2),color="k",ls="--",lw=1)
    ax.set(xlabel="x (km)",ylabel="y (km)",xlim=p["limits_km"],ylim=p["limits_km"])
    ax.set_aspect("equal")
    return image


def fit_ellipse(ax,row,color):
    Q = np.array([[row.Q11,row.Q12],[row.Q12,row.Q22]])
    ellipse(ax,row.xc_m,row.yc_m,Q,row.Rc_m/np.sqrt(2),color=color,ls="--",lw=1.5)
    ax.plot(row.xc_m/1000,row.yc_m/1000,"x",color=color,ms=8)


def outer_profile(ax,curve,config,color):
    fit = curve["outer"]
    rho,vt = np.asarray(fit["rho"],float),np.asarray(fit["vt"],float)
    ax.scatter(rho/1000,np.abs(vt),color=color,marker="x",s=25)
    if fit["status"] == "optimized":
        radius = np.linspace(0,config["plot"]["outer_radius_limit_km"]*1000,500)
        speed = abs(fit["Omega"])*radius*np.exp(-radius**2/fit["Rc"]**2)
        ax.plot(radius/1000,speed,color=color,alpha=.3,ls="--")
        inside = (radius >= rho.min()) & (radius <= rho.max())
        ax.plot(radius[inside]/1000,speed[inside],color=color,lw=2)
        ax.axvline(fit["Rc"]/np.sqrt(2)/1000,color=color,ls="--",lw=1)
    else:
        ax.text(.03,.93,fit["status"],transform=ax.transAxes,va="top",fontsize=8)
    ax.axvline(config["truth"]["rc_m"]/np.sqrt(2)/1000,color="k",lw=1,alpha=.5)
    ax.set(xlabel="ρ (km)",ylabel="|vₜ*| (m s⁻¹)",xlim=(0,config["plot"]["outer_radius_limit_km"]))


def plot_doppio(data,config):
    """Manuscript Figure 7: legacy layout using the current saved DOPPIO fits."""
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    field = data["field"]
    x,y = field["x_m"]/1000,field["y_m"]/1000
    X,Y = np.meshgrid(x,y,indexing="ij")
    u,v = field["u_m_s"],field["v_m_s"]
    truth = config["truth"]
    Q = np.asarray(truth["Q_input"],float)
    Q = Q/np.sqrt(np.linalg.det(Q))
    dx,dy = X-truth["xc_m"]/1000,Y-truth["yc_m"]/1000
    rho2 = Q[0,0]*dx**2+2*Q[0,1]*dx*dy+Q[1,1]*dy**2
    Rc = truth["rc_m"]/1000
    df_res = data["parameters"].query("method == 'DOPPIO'").set_index("case").copy()
    df_res.index = [name[1:] for name in df_res.index]
    df_res["xc"],df_res["yc"],df_res["Rc"] = df_res.xc_m/1000,df_res.yc_m/1000,df_res.Rc_m/1000
    df_res = df_res.rename(columns={"Q11":"q11","Q12":"q12","Q22":"q22"})
    dic_tests = {}
    for spec in config["cases"]:
        name = spec["name"]
        profiles = data["curves"][name+"_DOPPIO"]["profiles"]
        curve = {key:np.asarray(profiles[key])/(1000 if key[0] in ('x','y') else 1)
                 for key in ('x1','y1','u1','v1','x2','y2','u2','v2')}
        curve.update(name=name[1:],yu1=np.asarray(profiles["tangent_u1"]),
                     yv1=np.asarray(profiles["tangent_v1"]),
                     yu2=np.asarray(profiles["tangent_u2"]),
                     yv2=np.asarray(profiles["tangent_v2"]))
        dic_tests[name[1:]] = curve
    def extract_number(name):
        return int(name[-1])
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(2, 5, figure=fig)
    ax_main = fig.add_subplot(gs[:, :2])
    cmap = 'Blues_r'
    (spacing, scale, width) = (10, 40, 0.002)
    pc = ax_main.pcolor(x, y, np.hypot(u, v).T, cmap=cmap)
    plt.colorbar(pc, ax=ax_main, orientation='horizontal', location='top', label='Current speed (ms$^{-1}$)')
    ax_main.quiver(x[::spacing], y[::spacing], u[::spacing, ::spacing].T, v[::spacing, ::spacing].T, alpha=0.2, scale=scale, width=width)
    ax_main.contour(X, Y, rho2, levels=[Rc ** 2 / 2], colors='k', linestyles='--')
    colours = ['r', 'g', 'b']
    color = ['r', 'g', 'b']
    i = 0
    for dic in dic_tests.values():
        if 'P' in dic['name']:
            (x1, y1, u1, v1, x2, y2, u2, v2) = (dic[k] for k in ['x1', 'y1', 'u1', 'v1', 'x2', 'y2', 'u2', 'v2'])
            data = df_res.loc[dic['name']]
            ax_main.plot(x1, y1, 'k', zorder=2)
            ax_main.plot(x2, y2, 'k', zorder=2)
            ax_main.quiver(x1, y1, u1, v1, scale=scale, width=width * 2, color=colours[i], zorder=3)
            ax_main.quiver(x2, y2, u2, v2, scale=scale, width=width * 2, color=colours[i], zorder=3)
            ax_main.scatter(data.xc, data.yc, marker='x', s=60, color=colours[i], zorder=20 - extract_number(dic['name']))
            (dx_ell, dy_ell) = (X - data.xc, Y - data.yc)
            rho2_ell = data.q11 * dx_ell ** 2 + 2 * data.q12 * dx_ell * dy_ell + data.q22 * dy_ell ** 2
            ax_main.contour(X, Y, rho2_ell, levels=[data.Rc ** 2 / 2], colors=colours[i], linestyles='--', zorder=3)
            ax_main.text(33, -74 - 10 * i, f'$(x_0,y_0)=({data.xc:.2f},{data.yc:.2f})$', color='darkgreen' if '2' in dic['name'] else colours[i], zorder=11)
            i += 1
    (x_pos, y_pos) = (0.8, 0.09)
    box_width = 0.37
    box_height = 0.16
    rect = Rectangle((x_pos - box_width / 2, y_pos - box_height / 2), box_width, box_height, transform=ax_main.transAxes, facecolor='white', edgecolor='black', zorder=10)
    ax_main.add_patch(rect)
    ax_main.text(-160, 140, '(a)', fontsize=14, fontweight='bold')
    ax_main.axis('equal')
    ax_main.set_xlim(-100, 100)
    ax_main.set_ylim(-100, 100)
    ax_main.set_xlabel('x (km)')
    ax_main.set_ylabel('y (km)')
    axs = []
    tags = ['(b)', '(c)', '(d)', '(e)', '(f)', '(g)']
    for (d, dic) in enumerate(dic_tests.values()):
        (x1, y1, u1, v1, x2, y2, u2, v2) = (dic[k] for k in ['x1', 'y1', 'u1', 'v1', 'x2', 'y2', 'u2', 'v2'])
        (yu1, yv1, yu2, yv2) = (dic[k] for k in ['yu1', 'yv1', 'yu2', 'yv2'])
        data = df_res.loc[dic['name']]
        ax2 = fig.add_subplot(gs[d // 3, 2 + d % 3])
        ax2.plot(x1, u1, color='brown', label='$U_1$', alpha=0.6)
        ax2.plot(x1, yu1, color='brown', alpha=0.2)
        ax2.plot(x1, v1, 'k', label='$V_1$', alpha=0.6)
        ax2.plot(x1, yv1, 'k', alpha=0.2)
        ax2.plot(y2, u2, color='olive', label='$U_2$', alpha=0.6)
        ax2.plot(y2, yu2, color='olive', alpha=0.2)
        ax2.plot(y2, v2, 'm', label='$V_2$', alpha=0.6)
        ax2.plot(y2, yv2, 'm', alpha=0.2)
        ax2.axhline(y=0, color='k', zorder=-1, linewidth=0.5, alpha=0.5)
        ax2.axvline(x=0, color='k', zorder=-1, linewidth=0.5, alpha=0.5)
        ax2.set_ylabel('(ms$^{-1}$)')
        ax2.set_xlabel('x and y (km)')
        ax2.set_title('E' + dic['name'])
        if dic['name'] == 'P2':
            ax2.legend(loc='lower center', ncol=2)
        axs.append(ax2)
        ax2.text(-0.1, 1.05, tags[d], fontsize=12, fontweight='bold', transform=ax2.transAxes)
    ymin = -4
    ymax = 4
    for ax in axs:
        ax.set_ylim(ymin, ymax)
    ax_main.text(45, 45, 'EP1', color=color[0], zorder=3, fontsize=18, fontweight='bold')
    ax_main.text(-85, 75, 'EP2', color=color[1], zorder=3, fontsize=18, fontweight='bold')
    ax_main.text(-100, -55, 'EP3', color=color[2], zorder=3, fontsize=18, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_doppio_outer(data,config):
    """Legacy DOPPIO outer-fit appearance, using current stored fit results."""
    fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(10, 4),
                            sharey=True, sharex=True)
    radius = np.linspace(0, config["plot"]["outer_radius_limit_km"]*1000, 400)
    # Match Figure 7: test 1 red, test 2 green, test 3 blue.
    for i, (ax, spec, color) in enumerate(zip(axs.flat, config["cases"], ['r', 'g', 'b']*2)):
        fit = data["curves"][spec["name"]+"_DOPPIO"]["outer"]
        rho, vt = np.asarray(fit["rho"], float), np.asarray(fit["vt"], float)
        ax.scatter(rho/1000, np.abs(vt), s=40, marker='x', zorder=2, color=color)
        if fit["status"] == "optimized":
            speed = abs(fit["Omega"])*radius*np.exp(-radius**2/fit["Rc"]**2)
            mask = radius <= rho.max()
            ax.plot(radius[mask]/1000, speed[mask], lw=2, color='c', zorder=1)
            ax.plot(radius[~mask]/1000, speed[~mask], lw=2, color='c', zorder=1, alpha=.3)
            ax.axvline(fit["Rc"]/np.sqrt(2)/1000, ls='--', color=color, label='', lw=2)
        else:
            ax.text(.03, .93, fit["status"], transform=ax.transAxes, va='top', fontsize=8)
        ax.set_title(spec["name"])
        ax.set_xlabel(r'$\rho$ (km)')
        ax.set_ylabel(r'$|v_t^\star|$ (ms$^{-1}$)' if i in (0, 3) else '')
        ax.axvline(config["truth"]["rc_m"]/np.sqrt(2)/1000, color='k', zorder=-1, alpha=.5)
    return fig


def _latte_legacy_profile(ax, curve, config, color):
    """Legacy profile artists, evaluated from stored parameters without fitting."""
    from matplotlib.colors import to_rgb
    fit = curve["outer"]
    rho,vt = np.asarray(fit["rho"],float),np.asarray(fit["vt"],float)
    ax.scatter(rho/1000,np.abs(vt),s=50,zorder=1,
               color=tuple(.8*c for c in to_rgb(color)),marker='x')
    if fit["status"] == "optimized":
        radius = np.linspace(0,config["plot"]["outer_radius_limit_km"]*1000,400)
        speed = abs(fit["Omega"])*radius*np.exp(-radius**2/fit["Rc"]**2)
        mask = radius <= rho.max()
        ax.plot(radius[mask]/1000,speed[mask],lw=3,zorder=0,color=color)
        ax.plot(radius[~mask]/1000,speed[~mask],lw=3,zorder=0,color=color,alpha=.3)
        ax.axvline(fit["Rc"]/np.sqrt(2)/1000,ls='--',label=r'$\rho_{\max}$',lw=2,color=color)
    else:
        ax.text(.03,.93,fit["status"],transform=ax.transAxes,va='top',fontsize=8)


def plot_latte(data,config):
    """Manuscript Figure 11: original layout, with current stored LATTE results."""
    import matplotlib.gridspec as gridspec
    field = data["field"]
    x,y = field["x_m"]/1000,field["y_m"]/1000
    X,Y = np.meshgrid(x,y,indexing='ij')
    u,v = field["u_m_s"],field["v_m_s"]
    saved_curves = data["curves"]
    truth = config["truth"]
    Q = np.asarray(truth["Q_input"],float)
    Q = Q/np.sqrt(np.linalg.det(Q))
    dx,dy = X-truth["xc_m"]/1000,Y-truth["yc_m"]/1000
    rho2 = Q[0,0]*dx**2+2*Q[0,1]*dx*dy+Q[1,1]*dy**2
    df_res = data["parameters"].copy()
    df_res['xc'],df_res['yc'],df_res['Rc'] = df_res.xc_m/1000,df_res.yc_m/1000,df_res.Rc_m/1000
    df_res = df_res.rename(columns={'Q11':'q11','Q12':'q12','Q22':'q22'})
    control = df_res.iloc[0].copy()
    control['xc'],control['yc'],control['Rc'] = truth['xc_m']/1000,truth['yc_m']/1000,truth['rc_m']/1000
    control['q11'],control['q12'],control['q22'] = Q[0,0],Q[0,1],Q[1,1]
    df_res.loc[-1] = control
    df_res = df_res.sort_index().reset_index(drop=True)
    arrays = {name:[np.asarray(curve[key])/(1000 if key in ('x_m','y_m') else 1)
                   for key in ('x_m','y_m','u_m_s','v_m_s')]
              for name,curve in saved_curves.items()}
    x_squ,y_squ,u_squ,v_squ = arrays['Ex1']
    x_drif,y_drif,u_drif,v_drif = arrays['Ex2']
    x_l,y_l,u_l,v_l = arrays['Ex3']
    split = config['sampling']['transects'][0]['count']
    x_l1,y_l1,u_l1,v_l1 = [a[:split] for a in arrays['Ex3']]
    x_l2,y_l2,u_l2,v_l2 = [a[split:] for a in arrays['Ex3']]
    x_p,y_p,u_p,v_p = arrays['Ex4']
    grid_side = len(np.unique(x_p))
    clr = 'b'
    scat_scale = 1
    cmap = 'Blues_r'
    (spacing, scale, width) = (10, 40, 0.004)
    wid = 2
    lw = 3
    tab10 = plt.get_cmap('tab10').colors
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0, hspace=0)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax in axs:
        pc = ax.pcolor(x, y, np.hypot(u, v).T, cmap=cmap)
        ax.quiver(x[::spacing], y[::spacing], u[::spacing, ::spacing].T, v[::spacing, ::spacing].T, alpha=0.3, scale=scale, width=width, zorder=1)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-75, 75)
        ax.set_ylim(-75, 75)
        ax.set_xlabel('x (km)', fontsize=12)
        ax.set_ylabel('y (km)', fontsize=12)
    axs[0].plot(x_squ, y_squ, 'c', linewidth=0.5, zorder=2)
    axs[0].quiver(x_squ, y_squ, u_squ, v_squ, scale=scale * scat_scale, width=width * wid, color=tab10[1], zorder=3)
    axs[0].contour(X, Y, rho2, levels=[df_res.iloc[0].Rc ** 2 / 2], colors='k', linestyles='--')
    data = df_res.iloc[1]
    (dx_ell, dy_ell) = (X - data.xc, Y - data.yc)
    rho2_ell = data.q11 * dx_ell ** 2 + 2 * data.q12 * dx_ell * dy_ell + data.q22 * dy_ell ** 2
    axs[0].contour(X, Y, rho2_ell, levels=[data.Rc ** 2 / 2], colors=[tab10[1]], linewidths=[lw], linestyles='--')
    axs[0].scatter(data.xc, data.yc, marker='x', s=150, color=tab10[1], zorder=10, linewidths=3)
    axs[0].set_title('Ex1: S-ADCP Data')
    axs[0].text(-0.1, 1.05, 'a)', transform=axs[0].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    axs[1].plot(x_drif, y_drif, 'c', linewidth=0.5, zorder=2)
    axs[1].quiver(x_drif, y_drif, u_drif, v_drif, scale=scale * scat_scale, width=width * wid, color=tab10[2], zorder=3)
    axs[1].contour(X, Y, rho2, levels=[df_res.iloc[0].Rc ** 2 / 2], colors='k', linestyles='--')
    data = df_res.iloc[2]
    (dx_ell, dy_ell) = (X - data.xc, Y - data.yc)
    rho2_ell = data.q11 * dx_ell ** 2 + 2 * data.q12 * dx_ell * dy_ell + data.q22 * dy_ell ** 2
    axs[1].contour(X, Y, rho2_ell, levels=[data.Rc ** 2 / 2], colors=[tab10[2]], linewidths=[lw], linestyles='--')
    axs[1].scatter(data.xc, data.yc, marker='x', s=150, color=tab10[2], zorder=10, linewidths=3)
    axs[1].set_title('Ex2: Drifter Data')
    axs[1].text(-0.1, 1.05, 'b)', transform=axs[1].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    axs[2].plot(x_l1, y_l1, 'c', linewidth=0.5, zorder=2)
    axs[2].plot(x_l2, y_l2, 'c', linewidth=0.5, zorder=2)
    axs[2].quiver(x_l1, y_l1, u_l1, v_l1, scale=scale * scat_scale, width=width * wid, color=tab10[3], zorder=3)
    axs[2].quiver(x_l2, y_l2, u_l2, v_l2, scale=scale * scat_scale, width=width * wid, color=tab10[3], zorder=3)
    axs[2].contour(X, Y, rho2, levels=[df_res.iloc[0].Rc ** 2 / 2], colors='k', linestyles='--')
    data = df_res.iloc[3]
    (dx_ell, dy_ell) = (X - data.xc, Y - data.yc)
    rho2_ell = data.q11 * dx_ell ** 2 + 2 * data.q12 * dx_ell * dy_ell + data.q22 * dy_ell ** 2
    axs[2].contour(X, Y, rho2_ell, levels=[data.Rc ** 2 / 2], colors=[tab10[3]], linewidths=[lw], linestyles='--')
    axs[2].scatter(data.xc, data.yc, marker='x', s=150, color=tab10[3], zorder=10, linewidths=3)
    axs[2].set_title('Ex3: Transect Data')
    axs[2].text(-0.1, 1.05, 'c)', transform=axs[2].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    for i in range(0, len(x_p), grid_side):
        axs[3].plot([x_p[i], x_p[i + grid_side - 1]], [y_p[i], y_p[i + grid_side - 1]], 'c', linewidth=0.5, zorder=2)
    for i in range(grid_side):
        axs[3].plot([x_p[i], x_p[i + grid_side * (grid_side - 1)]], [y_p[i], y_p[i + grid_side * (grid_side - 1)]], 'c', linewidth=0.5, zorder=2)
    axs[3].quiver(x_p, y_p, u_p, v_p, scale=scale * scat_scale, width=width * wid, color=tab10[6], zorder=3)
    axs[3].contour(X, Y, rho2, levels=[df_res.iloc[0].Rc ** 2 / 2], colors='k', linestyles='--')
    data = df_res.iloc[4]
    (dx_ell, dy_ell) = (X - data.xc, Y - data.yc)
    rho2_ell = data.q11 * dx_ell ** 2 + 2 * data.q12 * dx_ell * dy_ell + data.q22 * dy_ell ** 2
    axs[3].contour(X, Y, rho2_ell, levels=[data.Rc ** 2 / 2], colors=[tab10[6]], linewidths=[lw], linestyles='--')
    axs[3].scatter(data.xc, data.yc, marker='x', s=150, color=tab10[6], zorder=10, linewidths=3)
    axs[3].set_title('Ex4: Gridded Data')
    axs[3].text(-0.1, 1.05, 'd)', transform=axs[3].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    cax = fig.add_axes([-0.04, 0.15, 0.015, 0.7])
    cb = fig.colorbar(pc, cax=cax, orientation='vertical')
    cb.set_label('Current speed (ms$^{-1}$)', fontsize=12)
    right = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[:, 2], hspace=0.25)
    ax_fits = [fig.add_subplot(right[i, 0]) for i in range(4)]
    rho_plot_max = 100
    lw = 3
    Rc = df_res.iloc[0].Rc
    _latte_legacy_profile(ax_fits[0], saved_curves['Ex1'], config, tab10[1])
    ax_fits[0].axvline(Rc / np.sqrt(2), color='k', zorder=-1, alpha=0.5)
    ax_fits[0].set_title('Ex1')
    ax_fits[0].set_xlabel('')
    ax_fits[0].set_ylabel('$|v_t^\\star|$ (ms$^{-1}$)', fontsize=11)
    ax_fits[0].text(-0.1, 1.1, 'e)', transform=ax_fits[0].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    _latte_legacy_profile(ax_fits[1], saved_curves['Ex2'], config, tab10[2])
    ax_fits[1].axvline(Rc / np.sqrt(2), color='k', zorder=-1, alpha=0.5)
    ax_fits[1].set_title('Ex2')
    ax_fits[1].set_xlabel('')
    ax_fits[1].set_ylabel('$|v_t^\\star|$ (ms$^{-1}$)', fontsize=11)
    ax_fits[1].text(-0.1, 1.1, 'f)', transform=ax_fits[1].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    _latte_legacy_profile(ax_fits[2], saved_curves['Ex3'], config, tab10[3])
    ax_fits[2].axvline(Rc / np.sqrt(2), color='k', zorder=-1, alpha=0.5)
    ax_fits[2].set_title('Ex3')
    ax_fits[2].set_xlabel('')
    ax_fits[2].set_ylabel('$|v_t^\\star|$ (ms$^{-1}$)', fontsize=11)
    ax_fits[2].text(-0.1, 1.1, 'g)', transform=ax_fits[2].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    _latte_legacy_profile(ax_fits[3], saved_curves['Ex4'], config, tab10[6])
    ax_fits[3].axvline(Rc / np.sqrt(2), color='k', zorder=-1, alpha=0.5)
    ax_fits[3].set_title('Ex4')
    ax_fits[3].set_xlabel('$\\rho$ (km)', fontsize=11)
    ax_fits[3].set_ylabel('$|v_t^\\star|$ (ms$^{-1}$)', fontsize=11)
    ax_fits[3].text(-0.1, 1.1, 'h)', transform=ax_fits[3].transAxes, fontsize=13, fontweight='bold', va='top', ha='left')
    return fig


def plot_schematic(data,config):
    """Legacy schematic styling, applied to saved reproducible observations."""
    frame = data["sampling"]
    sampling,truth,plot = config["sampling"],config["truth"],config["plot"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True,
                            sharey=True, constrained_layout=True)
    xi, yi = frame.x_m.to_numpy()/1000, frame.y_m.to_numpy()/1000
    ui, vi = frame.u_m_s.to_numpy(), frame.v_m_s.to_numpy()
    init_mask, core_mask, out_mask = [frame[k].to_numpy(bool)
        for k in ("initial_mask", "core_mask", "outer_mask")]
    other_mask_3 = ~(core_mask | out_mask)
    xc, yc = truth["xc_m"]/1000, truth["yc_m"]/1000
    x0, y0 = np.asarray(sampling["initial_guess_m"])/1000
    core_r, outer_r = sampling["core_radius_m"]/1000, sampling["outer_radius_m"]/1000
    lim = max(abs(value) for value in plot["limits_km"])
    init_color, core_color, outer_color = plot["colors"]
    faint_color = "0.75"
    theta = np.linspace(0, 2*np.pi, 400)
    s, scale, width = 90, plot["quiver_scale"], 0.006
    for ax in axs:
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axs[0].set_title('1. Initial guess', fontsize=13, weight='bold')
    axs[0].add_patch(plt.Circle((x0, y0), core_r, facecolor=init_color, edgecolor='none', alpha=0.18, zorder=0))
    axs[0].plot(x0 + core_r * np.cos(theta), y0 + core_r * np.sin(theta), color=init_color, lw=3, zorder=2)
    axs[0].quiver(xi[~init_mask], yi[~init_mask], ui[~init_mask], vi[~init_mask], scale=scale, color=faint_color, alpha=0.35, width=width, zorder=2)
    axs[0].quiver(xi[init_mask], yi[init_mask], ui[init_mask], vi[init_mask], scale=scale, color=init_color, alpha=0.95, width=width, zorder=3)
    axs[0].scatter(x0, y0, marker='X', color=init_color, s=s * 1.4, edgecolor='k', linewidths=0.8, zorder=5)
    axs[0].text(x0 + 25, y0 + 25, 'initial guess', color=init_color, fontsize=12, weight='bold')
    axs[1].set_title('2. Fit inner-core', fontsize=13, weight='bold')
    axs[1].quiver(xi[~core_mask], yi[~core_mask], ui[~core_mask], vi[~core_mask], scale=scale, color=faint_color, alpha=0.3, width=width, zorder=2)
    axs[1].quiver(xi[core_mask], yi[core_mask], ui[core_mask], vi[core_mask], scale=scale, color=core_color, alpha=0.95, width=width, zorder=4)
    axs[1].add_patch(plt.Circle((xc, yc), core_r, facecolor=core_color, edgecolor='none', alpha=0.12, zorder=0))
    axs[1].plot(xc + core_r * np.cos(theta), yc + core_r * np.sin(theta), color=core_color, lw=2.5, alpha=0.9, zorder=1)
    axs[1].scatter(x0, y0, marker='X', color=init_color, s=s * 1.2, edgecolor='k', linewidths=0.8, zorder=5)
    axs[1].scatter(xc, yc, marker='X', color=core_color, s=s * 1.4, edgecolor='k', linewidths=0.8, zorder=6)
    axs[1].text(xc + 32, yc - 32, 'find $\\mathbf{x_0, y_0, Q, \\Omega}$', color=core_color, fontsize=12, weight='bold')
    axs[2].set_title('3. Fit outer-core', fontsize=13, weight='bold')
    axs[2].quiver(xi[other_mask_3], yi[other_mask_3], ui[other_mask_3], vi[other_mask_3], scale=scale, color=faint_color, alpha=0.25, width=width, zorder=2)
    axs[2].quiver(xi[core_mask], yi[core_mask], ui[core_mask], vi[core_mask], scale=scale, color=core_color, alpha=0.95, width=width, zorder=3)
    axs[2].quiver(xi[out_mask], yi[out_mask], ui[out_mask], vi[out_mask], scale=scale, color=outer_color, alpha=0.95, width=width, zorder=4)
    axs[2].scatter(xc, yc, marker='X', color=core_color, s=s * 1.4, edgecolor='k', linewidths=0.8, zorder=5)
    axs[2].add_patch(plt.Circle((xc, yc), core_r, facecolor=core_color, edgecolor='none', alpha=0.12, zorder=0))
    axs[2].plot(xc + core_r * np.cos(theta), yc + core_r * np.sin(theta), color=core_color, lw=2.5, alpha=0.9, zorder=1)
    axs[2].add_patch(plt.Circle((xc, yc), outer_r, facecolor=outer_color, edgecolor='none', alpha=0.08, zorder=0))
    axs[2].plot(xc + outer_r * np.cos(theta), yc + outer_r * np.sin(theta), color=outer_color, lw=2.5, alpha=0.9, zorder=1)
    axs[2].text(xc - outer_r + 10, yc + outer_r - 10, 'find $\\mathbf{R_c, \\psi_0}$', color=outer_color, ha='center', fontsize=12, weight='bold')
    return fig


def render(data,config,directory):
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42})
    figures = {"doppio_idealised":{"DOPPIO":plot_doppio,"DOPPIO_outercore_fits":plot_doppio_outer},
               "latte_idealised":{"LATTE":plot_latte},"schematic":{"ESP_schematic":plot_schematic}}[config["experiment"]]
    paths = []
    for name,plotter in figures.items():
        fig = plotter(data,config)
        try:
            paths.append(save_manuscript_figure(fig,name,directory,dpi=config["plot"]["dpi"],
                         **({"bbox_inches":"tight"} if name == "LATTE" else {})))
        finally:
            plt.close(fig)
    metadata = dict(utc=datetime.now(timezone.utc).isoformat(),plot={**config["plot"],"formats":["png"]},source_sha256=sha256(__file__),
        run_metadata_sha256=sha256(directory/"run_metadata.json"),
        figures_sha256={os.path.relpath(path,directory):sha256(path) for path in paths})
    (directory/"plot_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return paths
