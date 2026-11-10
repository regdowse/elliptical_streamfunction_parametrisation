"""Render saved DOPPIO/LATTE validations and the illustrative ESP schematic."""
from datetime import datetime, timezone
import json

import matplotlib.pyplot as plt
import numpy as np

from experiments.outputs import sha256


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
    fig = plt.figure(figsize=(16,6),layout="constrained")
    gs = fig.add_gridspec(2,5)
    ax = fig.add_subplot(gs[:,:2])
    image = background(ax,data,config)
    fig.colorbar(image,ax=ax,location="top",label="Current speed (m s⁻¹)",shrink=.8)
    ax.set_title("a) Proximity tests",loc="left")
    rows = data["parameters"].query("method == 'DOPPIO'").set_index("case")
    for spec,color in zip(config["cases"][:3],config["plot"]["colors"]):
        name = spec["name"];curve = data["curves"][name+"_DOPPIO"]["profiles"];row = rows.loc[name]
        for i in (1,2):
            x,y,u,v = [np.asarray(curve[f"{k}{i}"],float) for k in ("x","y","u","v")]
            ax.plot(x/1000,y/1000,color="k",lw=.5)
            ax.quiver(x/1000,y/1000,u,v,color=color,scale=config["plot"]["quiver_scale"],width=.004)
        fit_ellipse(ax,row,color)
        ax.plot([],[],color=color,label=f"{name}: ({row.xc_m/1000:.2f}, {row.yc_m/1000:.2f}) km")
        ax.text((spec["intersection_m"][0]-spec["half_length_m"])/1000,
            spec["intersection_m"][1]/1000+10,name,color=color,weight="bold")
    ax.legend(loc="lower right",fontsize=8)
    for j,spec in enumerate(config["cases"]):
        profile = fig.add_subplot(gs[j//3,2+j%3])
        curve = data["curves"][spec["name"]+"_DOPPIO"]["profiles"]
        for arm,component,color in ((1,"u","brown"),(1,"v","black"),(2,"u","olive"),(2,"v","magenta")):
            coordinate = np.asarray(curve["x1" if arm == 1 else "y2"],float)/1000
            profile.plot(coordinate,curve[f"{component}{arm}"],"o",color=color,ms=2.5,alpha=.55)
            profile.plot(coordinate,curve[f"polynomial_{component}{arm}"],color=color,lw=1.3,label=f"{component.upper()}{arm}")
            profile.plot(coordinate,curve[f"tangent_{component}{arm}"],color=color,ls="--",alpha=.35,lw=1)
        profile.axhline(0,color="k",lw=.5,alpha=.4)
        profile.set(xlabel="x or y (km)",ylabel="Velocity (m s⁻¹)",ylim=config["plot"]["velocity_limits_m_s"],
            title=f"{chr(98+j)}) {spec['name']}")
        if j == 1:
            profile.legend(ncol=2,fontsize=8)
    return fig


def plot_doppio_outer(data,config):
    fig,axs = plt.subplots(2,3,figsize=(10,5),sharex=True,sharey=True,layout="constrained")
    for ax,spec,color in zip(axs.flat,config["cases"],config["plot"]["colors"]*2):
        outer_profile(ax,data["curves"][spec["name"]+"_DOPPIO"],config,color)
        ax.set_title(spec["name"])
    return fig


def plot_latte(data,config):
    fig = plt.figure(figsize=(16,10),layout="constrained")
    gs = fig.add_gridspec(4,3,width_ratios=[1,1,.95])
    rows = data["parameters"].set_index("case")
    titles = ["Ship transect","Drifter trajectory","Oblique transects","Gridded observations"]
    maps = []
    for j,(name,color,title) in enumerate(zip(("Ex1","Ex2","Ex3","Ex4"),config["plot"]["colors"],titles)):
        ax = fig.add_subplot(gs[2*(j//2):2*(j//2)+2,j%2]);maps.append(ax)
        image = background(ax,data,config)
        curve = data["curves"][name];row = rows.loc[name]
        x,y,u,v = [np.asarray(curve[k],float) for k in ("x_m","y_m","u_m_s","v_m_s")]
        if name == "Ex3":
            first_count = config["sampling"]["transects"][0]["count"]
            for sl in (slice(None,first_count),slice(first_count,None)):
                ax.plot(x[sl]/1000,y[sl]/1000,color="cyan",lw=.6)
        elif name != "Ex4":
            ax.plot(x/1000,y/1000,color="cyan",lw=.6)
        ax.quiver(x/1000,y/1000,u,v,color=color,scale=config["plot"]["quiver_scale"],width=.006)
        fit_ellipse(ax,row,color)
        ax.set_title(f"{chr(97+j)}) {name}: {title}",loc="left",fontsize=11)
        profile = fig.add_subplot(gs[j,2])
        outer_profile(profile,curve,config,color)
        profile.set_title(f"{chr(101+j)}) {name}",loc="left",fontsize=11)
    fig.colorbar(image,ax=maps,location="left",label="Current speed (m s⁻¹)",shrink=.8,fraction=.035)
    return fig


def plot_schematic(data,config):
    frame = data["sampling"]
    s,t,p = config["sampling"],config["truth"],config["plot"]
    fig,axs = plt.subplots(1,3,figsize=(13,4.2),sharex=True,sharey=True,layout="constrained")
    x,y,u,v = [frame[key].to_numpy()/ (1000 if key in ("x_m","y_m") else 1) for key in ("x_m","y_m","u_m_s","v_m_s")]
    initial,core,outer = [frame[k].to_numpy(bool) for k in ("initial_mask","core_mask","outer_mask")]
    orange,magenta,green = p["colors"]
    centre = np.array([t["xc_m"],t["yc_m"]])/1000
    guess = np.asarray(s["initial_guess_m"])/1000
    def ring(ax,point,radius,color):
        ax.add_patch(plt.Circle(point,radius/1000,facecolor=color,alpha=.12,edgecolor="none"))
        ax.add_patch(plt.Circle(point,radius/1000,fill=False,edgecolor=color,lw=2))
    def vectors(ax,mask,color,alpha=1):
        ax.quiver(x[mask],y[mask],u[mask],v[mask],color=color,alpha=alpha,scale=p["quiver_scale"],width=.006)
    for j,ax in enumerate(axs):
        selected = initial if j == 0 else core if j == 1 else core|outer
        vectors(ax,~selected,".75",.35)
        if j == 0:
            vectors(ax,initial,orange)
            ring(ax,guess,s["core_radius_m"],orange)
        else:
            vectors(ax,core,magenta)
            ring(ax,centre,s["core_radius_m"],magenta)
            if j == 2:
                vectors(ax,outer,green)
                ring(ax,centre,s["outer_radius_m"],green)
            ax.scatter(*centre,marker="X",color=magenta,s=100,edgecolor="k",lw=.7,zorder=5)
        if j < 2:
            ax.scatter(*guess,marker="X",color=orange,s=90,edgecolor="k",lw=.7,zorder=4)
        ax.set_aspect("equal")
        ax.set(xlim=p["limits_km"],ylim=p["limits_km"],xticks=[],yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax,title in zip(axs,("1. Initial guess","2. Fit inner core","3. Fit outer core")):
        ax.set_title(title,weight="bold")
    axs[0].text(.55,.65,"initial guess",color=orange,transform=axs[0].transAxes,weight="bold")
    axs[1].text(.54,.34,"find x₀, y₀, Q, Ω",color=magenta,transform=axs[1].transAxes,weight="bold")
    axs[2].text(.12,.85,"find Rc, ψ₀",color=green,transform=axs[2].transAxes,weight="bold")
    return fig


def render(data,config,directory):
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42})
    figures = {"doppio_idealised":{"DOPPIO":plot_doppio,"DOPPIO_outercore_fits":plot_doppio_outer},
               "latte_idealised":{"LATTE":plot_latte},"schematic":{"ESP_schematic":plot_schematic}}[config["experiment"]]
    output = directory/"figures";output.mkdir(exist_ok=True)
    paths = []
    for name,plotter in figures.items():
        fig = plotter(data,config)
        try:
            for extension in config["plot"]["formats"]:
                path = output/f"{name}.{extension}"
                fig.savefig(path,dpi=config["plot"]["dpi"])
                paths.append(path)
        finally:
            plt.close(fig)
    metadata = dict(utc=datetime.now(timezone.utc).isoformat(),plot=config["plot"],source_sha256=sha256(__file__),
        run_metadata_sha256=sha256(directory/"run_metadata.json"),
        figures_sha256={str(path.relative_to(directory)):sha256(path) for path in paths})
    (directory/"plot_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return paths
