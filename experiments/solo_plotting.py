"""SOLO manuscript figures drawn exclusively from saved observations and fits."""
from datetime import datetime, timezone
import json

import numpy as np
import matplotlib.pyplot as plt

from experiments.outputs import sha256


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
    fig = plt.figure(figsize=(18,6),layout="constrained")
    gs = fig.add_gridspec(2,6,width_ratios=[1,1,2,1,1,2])
    rows = data["parameters"].set_index("case")
    theta = np.linspace(0,2*np.pi,361)
    for column,prefix,letters in [(0,"P","abc"),(3,"L","def")]:
        ax = fig.add_subplot(gs[:,column:column+2])
        image = field_axes(ax,data,config)
        fig.colorbar(image,ax=ax,location="top",label="Current speed (m s⁻¹)",shrink=.85)
        profiles = [fig.add_subplot(gs[j,column+2]) for j in range(2)]
        ax.set_title(f"{letters[0]}) {'Proximity' if prefix == 'P' else 'Length'} tests",loc="left")
        for i,color in enumerate(config["plot"]["colors"],1):
            name = f"{prefix}{i}"
            curve,row = data["curves"][name],rows.loc[name]
            x,y,u,v = [np.asarray(curve[k],float) for k in ("x_m","y_m","u_m_s","v_m_s")]
            ax.quiver(x/1000,y/1000,u,v,color=color,scale=config["plot"]["quiver_scale"],width=.004)
            ax.text(max(x[0]/1000,config["plot"]["limits_km"][0]+2),y[0]/1000+8,name,color=color,weight="bold")
            ax.plot(row.xc_m/1000,row.yc_m/1000,"x",color=color,ms=8,
                label=f"{name}: ({row.xc_m/1000:.2f}, {row.yc_m/1000:.2f}) km")
            if np.isfinite(row.Rc_m):
                r = row.Rc_m/np.sqrt(2)/1000
                ax.plot(row.xc_m/1000+r*np.cos(theta),row.yc_m/1000+r*np.sin(theta),ls="--",color=color,lw=1)
            for profile,observed,polynomial in zip(profiles,("v_m_s","u_m_s"),("polynomial_v","polynomial_u")):
                profile.plot(x/1000,curve[observed],color=color,lw=2,alpha=.55,label=name)
                profile.plot(x/1000,curve[polynomial],color=color,ls="--",lw=1.5)
        ax.legend(loc="lower right",fontsize=7)
        for j,profile in enumerate(profiles):
            profile.axhline(0,color="k",lw=.5,alpha=.5)
            profile.axvline(config["truth"]["xc_m"]/1000,color="k",lw=.5,ls="--",alpha=.5)
            profile.set(xlabel="x (km)",ylabel=("V" if j == 0 else "U")+" (m s⁻¹)",
                ylim=config["plot"]["velocity_limits_m_s"])
            profile.set_title(f"{letters[j+1]}) Observations and polynomial fits",loc="left",fontsize=10)
            profile.legend(fontsize=8)
    return fig


def plot_outer(data, config):
    fig,axs = plt.subplots(2,3,figsize=(10,5),sharex=True,sharey=True,layout="constrained")
    radius = np.linspace(0,config["plot"]["outer_radius_limit_km"]*1000,500)
    for ax,(name,curve),color in zip(axs.flat,[(k,v) for k,v in data["curves"].items() if k != "comparison"],config["plot"]["colors"]*2):
        fit = curve["outer"]
        rho,vt = np.asarray(fit["rho"],float),np.asarray(fit["vt"],float)
        ax.scatter(rho/1000,np.abs(vt),s=17,color=color)
        if fit["status"] == "optimized":
            speed = abs(fit["Omega"])*radius*np.exp(-radius**2/fit["Rc"]**2)
            sampled = (radius >= rho.min()) & (radius <= rho.max())
            ax.plot(radius/1000,speed,color=color,ls="--",alpha=.3)
            ax.plot(radius[sampled]/1000,speed[sampled],color=color)
            ax.axvline(fit["Rc"]/np.sqrt(2)/1000,color=color,ls="--",lw=1)
        else:
            ax.text(.03,.95,fit["status"],transform=ax.transAxes,va="top",fontsize=8)
        ax.axvline(config["truth"]["rc_m"]/np.sqrt(2)/1000,color="k",alpha=.5,lw=1)
        ax.set(title=name,xlabel="ρ (km)",xlim=(0,radius[-1]/1000))
    for ax in axs[:,0]:
        ax.set_ylabel("|vₜ*| (m s⁻¹)")
    return fig


def plot_comparison(data, config):
    fig,axs = plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
    image = field_axes(axs[0],data,config)
    fig.colorbar(image,ax=axs[0],location="top",label="Current speed (m s⁻¹)")
    curve = data["curves"]["comparison"]
    x,y,u,v = [np.asarray(curve[k],float) for k in ("x_m","y_m","u_m_s","v_m_s")]
    stride = config["plot"]["quiver_stride"]
    axs[0].quiver(x[::stride]/1000,y[::stride]/1000,u[::stride],v[::stride],color="m",
        scale=config["plot"]["quiver_scale"],width=.005)
    for row,color in zip(data["comparison"].itertuples(),["green","red"]):
        axs[0].plot(row.xc_m/1000,row.yc_m/1000,"x",color=color,ms=9,label=row.method)
    axs[0].legend(loc="lower right")
    for ax,component in zip(axs[1:],("u","v")):
        ax.plot(x/1000,curve[f"{component}_m_s"],color="m",lw=2,label="Observations")
        ax.plot(x/1000,curve[f"moca_{component}"],color="red",ls="--",label="MOCA regression")
        ax.plot(x/1000,curve[f"tangent_{component}"],color="green",ls="--",label="SOLO tangent")
        ax.axhline(0,color="k",lw=.5,alpha=.5)
        ax.set(xlabel="x (km)",ylabel=component.upper()+" (m s⁻¹)",ylim=config["plot"]["velocity_limits_m_s"])
        ax.legend(fontsize=8)
    for ax,letter in zip(axs,"abc"):
        ax.set_title(f"{letter})",loc="left")
    return fig


def plot_deformation(data, config):
    frame = data["deformation"]
    fig,axs = plt.subplots(1,2,figsize=(10,4),layout="constrained")
    theta = np.linspace(0,2*np.pi,361)
    unit = np.array([np.cos(theta),np.sin(theta)])
    colors = plt.cm.cool(np.linspace(0,1,len(frame)))
    radius = config["truth"]["rc_m"]/np.sqrt(2)
    centre = np.array([config["truth"]["xc_m"],config["truth"]["yc_m"]])[:,None]
    for row,color in zip(frame.itertuples(),colors):
        Q = np.array([[row.Q11,row.Q12],[row.Q12,row.Q22]])
        eigen,vectors = np.linalg.eigh(Q)
        ellipse = centre+vectors@np.diag(radius/np.sqrt(eigen))@unit
        axs[0].plot(* (ellipse/1000),color=color,lw=1)
        axs[0].plot(row.xc_m/1000,row.yc_m/1000,"x",color=color)
        axs[1].plot(row.aspect_ratio,row.centre_error_m/1000,"x",color=color)
    spec = config["deformation"]["transect"]
    axs[0].plot(np.array([spec["x_mid_m"]-spec["half_length_m"],spec["x_mid_m"]+spec["half_length_m"]])/1000,
        [spec["y_m"]/1000]*2,color="green",lw=2)
    axs[0].set(xlabel="x (km)",ylabel="y (km)",xlim=config["plot"]["limits_km"],ylim=config["plot"]["limits_km"])
    axs[0].set_aspect("equal")
    axs[1].plot(frame.aspect_ratio,frame.centre_error_m/1000,color="k",lw=.5,alpha=.5)
    axs[1].set(xlabel="Aspect ratio",ylabel="Centre error (km)")
    for ax,letter in zip(axs,"ab"):
        ax.set_title(f"{letter})",loc="left")
    return fig


def render(data, config, directory):
    plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42})
    figures = {"SOLO":plot_cases,"SOLO_outercore_fits":plot_outer,
               "Improved_MOCA":plot_comparison,"SOLO_AR_Err":plot_deformation}
    output = directory/"figures"
    output.mkdir(exist_ok=True)
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
    metadata = dict(utc=datetime.now(timezone.utc).isoformat(),plot=config["plot"],
        source_sha256=sha256(__file__),run_metadata_sha256=sha256(directory/"run_metadata.json"),
        figures_sha256={str(p.relative_to(directory)):sha256(p) for p in paths})
    (directory/"plot_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return paths
