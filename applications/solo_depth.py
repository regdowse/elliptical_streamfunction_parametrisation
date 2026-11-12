"""Fixed-window SOLO depth tracking and plots of stored fits. SI throughout.

No estimator is implemented here: all inversions use the root ESP functions.
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import functions as esp


def fit_depths(dataset, analysis, outer_kwargs):
    """One shallow-to-deep pass; rejected inner fits stop, outer failures do not."""
    start = analysis['start_depth_m']
    if start not in dataset.z.values:
        raise ValueError(f'Start depth {start:g} m is not a measured bin')
    for key in ('core_radius_m', 'outer_radius_m', 'search_half_width_m', 'max_centre_step_m'):
        if not np.isfinite(analysis[key]) or analysis[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    minimum = analysis['min_points']
    if not isinstance(minimum, int) or minimum < 4:
        raise ValueError('min_points must be an integer >=4')
    if analysis['expected_rotation_sign'] not in (-1, 1):
        raise ValueError('expected_rotation_sign must be -1 or 1')
    surface = dataset.sel(z=start)
    projection, slope = esp.project_sadcp_to_transect(surface.x.values, surface.y.values,
                                                    surface.u.values, surface.v.values)
    if projection.empty or not np.isfinite(slope):
        raise ValueError('Cannot establish transect frame at start depth')
    origin = projection.iloc[0][['x', 'y']].to_numpy(float)
    tangent = np.array([1., slope])/np.sqrt(1+slope*slope)
    normal = np.array([-tangent[1], tangent[0]])
    xy = np.column_stack([dataset.x.values, dataset.y.values])
    along = (xy-origin) @ tangent
    previous = np.asarray(analysis['initial_center_m'], float)
    if previous.shape != (2,) or not np.isfinite(previous).all():
        raise ValueError('initial_center_m needs two finite coordinates')
    samples, fits, records = {}, {}, []
    # Prepare real observations even at depths below a subsequently stopped track.
    for z in sorted(float(z) for z in dataset.z.values if z >= start):
        level = dataset.sel(z=z)
        sample = pd.DataFrame(dict(raw_index=dataset.raw_index.values, x_m=xy[:,0], y_m=xy[:,1],
            u=level.u.values, v=level.v.values, along_m=along))
        sample['finite'] = np.isfinite(sample[['x_m','y_m','u','v']]).all(axis=1)
        for key in ('search','inner','outer_candidate','outer_used'):
            sample[key] = False
        samples[z] = sample
    for z, sample in samples.items():
        finite = sample.finite.to_numpy()
        search = finite & (abs(along-(previous-origin) @ tangent) <= analysis['search_half_width_m'])
        sample['search'] = search
        inner, outer = {}, {}
        fit = dict(inner=inner, outer=outer, origin=origin, tangent=tangent, normal=normal)
        fits[z] = fit
        row = dict(depth_m=z, time=pd.Timestamp(dataset.time.mean().values),
            xc=np.nan, yc=np.nan, Q=np.full((2,2), np.nan), alpha=1.,
            Omega_core=np.nan, w_core=np.nan, Omega=np.nan, w=np.nan,
            Rc=np.nan, psi0=np.nan, candidate_Rc_m=np.nan, R2=np.nan,
            n_finite=int(finite.sum()), n_search=int(search.sum()), n_core=0,
            n_outer_candidates=0, n_outer_used=0, inner_valid=False, outer_valid=False,
            peak_bracketed=False, centre_step_m=np.nan, status='unattempted',
            fit_status='not_attempted', warnings='')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                if search.sum() < minimum:
                    raise ValueError('insufficient_search_points')
                indices = np.flatnonzero(search)
                indices = indices[np.argsort(along[indices])]
                uv = sample[['u','v']].to_numpy()
                l, vt, vn = along[indices], (uv @ tangent)[indices], (uv @ normal)[indices]
                fit.update(input_indices=indices, l=l, VT=vt, VN=vn)
                l0,r0,w,Q,omega = esp.solo(l,vt,vn, core_thresh=analysis['core_radius_m'],
                                         selection_details=inner)
                if 'core_mask' in inner:
                    sample.loc[indices,'inner'] = inner['core_mask']
                    row['n_core'] = int(sample.inner.sum())
                if not np.isfinite([l0,r0,omega]).all():
                    raise ValueError('nonfinite_inner_fit')
                if not l.min() < l0 < l.max():
                    raise ValueError('crossing_outside_observations')
                if omega*analysis['expected_rotation_sign'] <= 0:
                    raise ValueError('rotation_reversal')
                if row['n_core'] < minimum:
                    raise ValueError('insufficient_core_points')
                centre = origin+l0*tangent+r0*normal
                row['centre_step_m'] = float(np.linalg.norm(centre-previous))
                if row['centre_step_m'] > analysis['max_centre_step_m']:
                    raise ValueError('centre_jump')
                row.update(inner_valid=True, xc=centre[0], yc=centre[1], Q=Q,
                           Omega_core=omega, w_core=w, status='inner_only')
                previous = centre.copy()
                use = finite & (np.linalg.norm(xy-centre,axis=1) <= analysis['outer_radius_m'])
                sample['outer_candidate'] = use
                row['n_outer_candidates'] = int(use.sum())
                if use.sum() < minimum:
                    row['status'] = 'insufficient_outer_points'
                else:
                    rc,psi,final_omega = esp.out_core_param_fit(*xy[use].T,
                        sample.u.to_numpy()[use], sample.v.to_numpy()[use], *centre,Q,
                        Omega0=omega, core_mask=sample.inner.to_numpy()[use],
                        fit_details=outer, **outer_kwargs)
                    retained = np.asarray(outer.get('observation_indices',[]),int)
                    sample.loc[np.flatnonzero(use)[retained],'outer_used'] = True
                    rho = np.asarray(outer['rho'])
                    bracket = bool(len(rho) and np.isfinite(rc) and rho.min() < rc/np.sqrt(2) < rho.max())
                    row.update(candidate_Rc_m=rc, peak_bracketed=bracket,
                        n_outer_used=len(retained), fit_status=outer['status'], status=outer['status'])
                    if outer['status'] == 'optimized':
                        if analysis['require_peak_bracket'] and not bracket:
                            row['status'] = 'peak_not_bracketed'
                        else:
                            row.update(outer_valid=True, Rc=rc, psi0=psi, Omega=final_omega,
                                       w=final_omega*np.trace(Q), status='accepted')
                            uh,vh = esp.model_uv_at_xy(*xy[finite].T,*centre,Q,final_omega,rc)
                            row['R2'] = esp.vector_R2(sample.u[finite],sample.v[finite],uh,vh)
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                row['status'] = str(error)
            row['warnings'] = '; '.join(str(w.message) for w in caught)
        records.append(row)
        if not row['inner_valid']:
            break
    results = pd.DataFrame(records)
    if results.empty or not results.iloc[0].inner_valid:
        raise ValueError('No accepted centre at the starting depth: '+str(results.status.tolist()))
    reference = results.iloc[0]
    results['east_offset_km'] = (results.xc-reference.xc)/1000
    results['north_offset_km'] = (results.yc-reference.yc)/1000
    return results,samples,fits


def select_depth(samples, requested):
    levels = np.array(list(samples))
    if not np.isfinite(requested) or requested < levels.min() or requested > levels.max():
        raise ValueError(f'Requested depth must be within {levels.min():g}–{levels.max():g} m')
    return float(levels[np.argmin(abs(levels-requested))])


def diagnostic(results, fits, depth):
    rows = results[results.depth_m == depth]
    if rows.empty or not rows.iloc[0].inner_valid:
        raise ValueError(f'No accepted inner fit at diagnostic depth {depth:g} m')
    # Match solo(..., plot=True) using its stored coefficients and selected data.
    # The outer fit remains in the subsequent reconstruction/outer-fit figure.
    stored = fits[depth]
    p = stored['inner']; mask=np.asarray(p['core_mask'],bool)
    l = stored['l'][mask]
    lfit = np.linspace(l.min(),l.max(),500)
    dl = lfit-p['x0']
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for ax,values,curve,tangent,label in zip(axes,(stored['VN'],stored['VT']),
            (p['C']*dl+p['D']*dl**3,p['A']+p['B']*dl**2),
            (p['C']*dl,np.full_like(lfit,p['A'])),('v_normal','v_tangential')):
        ax.scatter(l/1000,values[mask])
        ax.plot(lfit/1000,curve)
        ax.plot(lfit/1000,tangent,color='tab:orange')
        ax.axvline(p['x0']/1000,linestyle='--')
        ax.set(xlabel=r'Along track distance (km)',ylabel=label,title=f'{label} fit')
    axes[0].axhline(0,linewidth=.8)
    fig.tight_layout()
    return fig


def profiles(results):
    fig,axes = plt.subplots(1,4,figsize=(14,5),sharey=True,layout='constrained')
    for ax,key,label,scale in zip(axes,['east_offset_km','north_offset_km','Omega','Rc'],
            ['x offset (km)','y offset (km)','Ω (s⁻¹)','Rc (km)'],[1,1,1,1000]):
        ax.plot(results[key]/scale,results.depth_m,'.-')
        ax.set_xlabel(label);ax.grid(alpha=.2)
    axes[0].set_ylabel('Depth (m)');axes[0].invert_yaxis()
    return fig


def horizontal_maps(results,samples,settings):
    depths=[select_depth(samples,z) for z in settings['horizontal_depths_m']]
    if len(depths)!=8 or len(set(depths))!=8:
        raise ValueError('Horizontal maps need eight distinct measured depths')
    axis=np.linspace(*settings['x_limits_km'],90)
    yaxis=np.linspace(*settings['y_limits_km'],90)
    xx,yy=np.meshgrid(axis*1000,yaxis*1000)
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    norm=Normalize(0,settings['speed_max_m_s']);theta=np.linspace(0,2*np.pi,180)
    for ax,z in zip(axes.flat,depths):
        sample=samples[z]; rows=results[results.depth_m==z];row=None if rows.empty else rows.iloc[0]
        if row is not None and row.outer_valid:
            u,v=esp.model_uv_at_xy(xx,yy,row.xc,row.yc,row.Q,row.Omega,row.Rc)
            ax.pcolormesh(axis,yaxis,np.hypot(u,v),norm=norm,cmap='Blues',shading='auto')
            ax.quiver(xx[::9,::9]/1000,yy[::9,::9]/1000,u[::9,::9],v[::9,::9],color='grey',alpha=.6,scale=12)
            radius=row.Rc/np.sqrt(2)/1000
            ax.plot(row.xc/1000+radius*np.cos(theta),row.yc/1000+radius*np.sin(theta),'r--',lw=1)
        for mask,color in [(sample.finite,'black'),(sample.finite & sample.outer_used,'limegreen'),
                           (sample.finite & sample.inner,'magenta')]:
            d=sample.loc[mask];ax.quiver(d.x_m/1000,d.y_m/1000,d.u,d.v,scale=12,color=color,width=.005)
        if row is not None and row.inner_valid:
            ax.scatter(row.xc/1000,row.yc/1000,marker='x',color='r')
        status='not attempted: track stopped' if row is None else row.status.replace('_',' ')
        ax.text(.02,.02,status,transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
        ax.text(.02,.94,f'Inner: {sample.inner.sum()}; outer used: {sample.outer_used.sum()}',transform=ax.transAxes,fontsize=8)
        ax.set(title=f'{z:g} m',xlabel='x (km)',ylabel='y (km)',
               xlim=settings['x_limits_km'],ylim=settings['y_limits_km'],aspect='equal')
    fig.colorbar(ScalarMappable(norm=norm,cmap='Blues'),ax=list(axes.flat),label=r'Current speed (m$^{-1}$)',shrink=.8)
    # fig.suptitle('SOLO · magenta: inner · green: outer (inner colour takes priority) · black: other\ngrey: reconstructed · red: centre / fitted peak')
    return fig,depths


def _section_peak_intersections(xc,yc,Q,Rc,x_plane,y_plane):
    """Intersect rho²=Rc²/2 with the fixed planes; no crossing gives NaNs.

    These are peak-contour intersections, not extrema of u/v along the section.
    Coordinates are SI; centred quadratics avoid large-origin cancellation.
    """
    q11,q12,q22=Q[0,0],Q[0,1],Q[1,1]
    dy=y_plane-yc
    discriminant_x=(q12*dy)**2-q11*(q22*dy*dy-Rc*Rc/2)
    x_roots=(xc+(-q12*dy+np.array([-1.,1.])*np.sqrt(discriminant_x))/q11
             if discriminant_x>=0 else np.full(2,np.nan))
    dx=x_plane-xc
    discriminant_y=(q12*dx)**2-q22*(q11*dx*dx-Rc*Rc/2)
    y_roots=(yc+(-q12*dx+np.array([-1.,1.])*np.sqrt(discriminant_y))/q22
             if discriminant_y>=0 else np.full(2,np.nan))
    return x_roots,y_roots


def _gaussian_vorticity(x,y,xc,yc,Q,Omega,Rc):
    """Analytic -Laplacian(psi) for the stored Gaussian streamfunction, in s^-1."""
    dx,dy=np.asarray(x)-xc,np.asarray(y)-yc
    qx=Q[0,0]*dx+Q[0,1]*dy
    qy=Q[1,0]*dx+Q[1,1]*dy
    rho2=dx*qx+dy*qy
    return Omega*np.exp(-rho2/Rc**2)*(np.trace(Q)-2*(qx*qx+qy*qy)/Rc**2)


def vertical_sections(results,settings):
    """Signed velocity and analytic vorticity through the fixed 37-m centre.

    Model evaluation and contour intersections only, without fitting or filling
    unsupported depth rows. The orange centreline is projected onto each plane.
    """
    reference=results.loc[(results.depth_m==37)&results.inner_valid]
    if reference.empty:
        raise ValueError('Vertical sections require an accepted 37-m centre')
    x_plane,y_plane=reference.iloc[0][['xc','yc']].to_numpy(float)
    step=settings['section_contour_step_m_s']
    if not np.isfinite(step) or step<=0:
        raise ValueError('section_contour_step_m_s must be positive and finite')
    x=np.linspace(*settings['x_limits_km'],201)*1000
    y=np.linspace(*settings['y_limits_km'],201)*1000
    depths=results.depth_m.to_numpy(float)
    zonal=np.full((len(depths),len(x)),np.nan)
    meridional=np.full((len(depths),len(y)),np.nan)
    zonal_vorticity=np.full_like(zonal,np.nan)
    meridional_vorticity=np.full_like(meridional,np.nan)
    x_roots=np.full((len(depths),2),np.nan)
    y_roots=np.full((len(depths),2),np.nan)
    for i,row in enumerate(results.itertuples()):
        if not row.outer_valid:
            continue
        _,zonal[i]=esp.model_uv_at_xy(x,np.full_like(x,y_plane),row.xc,row.yc,row.Q,row.Omega,row.Rc)
        meridional[i],_=esp.model_uv_at_xy(np.full_like(y,x_plane),y,row.xc,row.yc,row.Q,row.Omega,row.Rc)
        zonal_vorticity[i]=_gaussian_vorticity(x,np.full_like(x,y_plane),row.xc,row.yc,row.Q,row.Omega,row.Rc)
        meridional_vorticity[i]=_gaussian_vorticity(np.full_like(y,x_plane),y,row.xc,row.yc,row.Q,row.Omega,row.Rc)
        x_roots[i],y_roots[i]=_section_peak_intersections(row.xc,row.yc,row.Q,row.Rc,x_plane,y_plane)
    limit=settings['section_velocity_max_m_s']
    if limit is None:
        finite=np.r_[zonal[np.isfinite(zonal)],meridional[np.isfinite(meridional)]]
        limit=max(step,np.ceil(np.max(np.abs(finite))/step)*step) if len(finite) else step
    if not np.isfinite(limit) or limit<=0:
        raise ValueError('section_velocity_max_m_s must be null or positive and finite')
    count=max(1,int(np.ceil(limit/step)))
    levels=np.linspace(-limit,limit,2*count+1)
    vorticity_step=settings['section_vorticity_contour_step_s_1']
    if not np.isfinite(vorticity_step) or vorticity_step<=0:
        raise ValueError('section_vorticity_contour_step_s_1 must be positive and finite')
    vorticity_limit=settings['section_vorticity_max_s_1']
    if vorticity_limit is None:
        finite=np.r_[zonal_vorticity[np.isfinite(zonal_vorticity)],
                     meridional_vorticity[np.isfinite(meridional_vorticity)]]
        vorticity_limit=(max(vorticity_step,np.ceil(np.max(np.abs(finite))/vorticity_step)*vorticity_step)
                         if len(finite) else vorticity_step)
    if not np.isfinite(vorticity_limit) or vorticity_limit<=0:
        raise ValueError('section_vorticity_max_s_1 must be null or positive and finite')
    vorticity_count=max(1,int(np.ceil(vorticity_limit/vorticity_step)))
    vorticity_levels=np.linspace(-vorticity_limit,vorticity_limit,2*vorticity_count+1)
    fig,axes=plt.subplots(1,4,figsize=(20,6),sharey=True,layout='constrained')
    for ax,coordinate,values,roots,centre,label,component,panel_levels in zip(axes,
            (x,y,x,y),(zonal,meridional,zonal_vorticity,meridional_vorticity),
            (x_roots,y_roots,x_roots,y_roots),(results.xc,results.yc,results.xc,results.yc),
            ('x (km)','y (km)','x (km)','y (km)'),
            (r'v (ms$^{-1}$)',r'u (ms$^{-1}$)',r'$\zeta$ (s$^{-1}$)',r'$\zeta$ (s$^{-1}$)'),
            (levels,levels,vorticity_levels,vorticity_levels)):
        if len(depths)>=2 and np.isfinite(values).any():
            image=ax.contourf(coordinate/1000,depths,np.ma.masked_invalid(values),levels=panel_levels,
                             cmap='RdBu_r',extend='both',corner_mask=False)
            fig.colorbar(image,ax=ax,location='top',label=component)
        else:
            ax.text(.5,.5,'No supported depth reconstruction',ha='center',transform=ax.transAxes)
        ax.plot(centre.where(results.inner_valid)/1000,depths,color='orange',lw=2.5,label='Projected centreline')
        for branch in range(2):
            ax.plot(roots[:,branch]/1000,depths,'r--',lw=2.5,
                    label='Peak-contour intersection' if branch==0 else None)
        ax.set(xlabel=label,xlim=(coordinate[0]/1000,coordinate[-1]/1000))
    axes[0].set_ylabel('Depth (m)')
    axes[0].set_ylim(depths.max(),depths.min())
    # axes[0].legend(loc='lower left',fontsize=8)
    # fig.suptitle('Fixed section planes through the 37-m centre')
    # Saved display data make plane locations, signed components and actual
    # contour intersections inspectable without another estimator call.
    data=dict(depth_m=depths,x_m=x,y_m=y,zonal_v_m_s=zonal,meridional_u_m_s=meridional,
              zonal_vorticity_s_1=zonal_vorticity,meridional_vorticity_s_1=meridional_vorticity,
              vorticity_limit_s_1=vorticity_limit,vorticity_levels_s_1=vorticity_levels,
              x_peak_roots_m=x_roots,y_peak_roots_m=y_roots,
              x_plane_m=x_plane,y_plane_m=y_plane,velocity_limit_m_s=limit)
    return fig,data


def manuscript_panels(results,section_data,*,figure=None,subplot_spec=None,include_track=True):
    """Draw companion panels, optionally including the bird's-eye trajectory."""
    accepted=results.loc[results.inner_valid].sort_values('depth_m')
    if accepted.empty:
        raise ValueError('Companion panels require accepted inner centres')
    shallow,deep=accepted.depth_m.iloc[[0,-1]].to_numpy(float)
    within=(section_data['depth_m']>=shallow)&(section_data['depth_m']<=deep)
    depths=section_data['depth_m'][within]
    if len(depths)<2:
        raise ValueError('Companion panels need at least two measured depth levels')
    fig=figure if figure is not None else plt.figure(figsize=(7,13),layout='constrained')
    grid_factory=fig.add_gridspec if subplot_spec is None else subplot_spec.subgridspec
    heights=[1.05,1,1] if include_track else [1.05,1]
    grid=grid_factory(len(heights),2,width_ratios=[1,.045],height_ratios=heights)
    section_ax=fig.add_subplot(grid[0,0]);section_cax=fig.add_subplot(grid[0,1])
    values=section_data['zonal_vorticity_s_1'][within]
    # Reuse the original contour intervals as well as its fixed-plane values.
    levels=section_data['vorticity_levels_s_1']
    if np.isfinite(values).any():
        image=section_ax.contourf(section_data['x_m']/1000,depths,
            np.ma.masked_invalid(values),levels=levels,cmap='RdBu_r',
            extend='both',corner_mask=False)
        fig.colorbar(image,cax=section_cax,label=r'Relative vorticity $\omega$ (s$^{-1}$)')
    else:
        section_ax.text(.5,.5,'No supported reconstruction',ha='center',transform=section_ax.transAxes)
        section_cax.set_visible(False)
    for branch in range(2):
        section_ax.plot(section_data['x_peak_roots_m'][within,branch]/1000,depths,'r--',lw=2.5)
    section_ax.plot(accepted.xc/1000,accepted.depth_m,color='orange',lw=2.5)
    section_ax.set(xlabel='x (km)',ylabel='Depth (m)',ylim=(deep,shallow),
        xlim=(section_data['x_m'][0]/1000,section_data['x_m'][-1]/1000),)
        # title=f"Zonal vorticity · y={section_data['y_plane_m']/1000:.2f} km")
    # section_ax.set_ylim(None, 0)

    rc_ax=fig.add_subplot(grid[1,0]);omega_ax=rc_ax.twiny()
    radius_color='saddlebrown';omega_color='dodgerblue'
    tracked=results.loc[(results.depth_m>=shallow)&(results.depth_m<=deep)]
    rc_ax.plot(tracked.Rc/1000,tracked.depth_m,'.-',color=radius_color,lw=2.5,ms=5)
    omega_ax.plot(tracked.Omega,tracked.depth_m,'.-',color=omega_color,lw=2.5,ms=5)
    rc_ax.set_xlabel(r'Characteristic radius $\mathbf{R_c}$ (km)', fontweight='bold')
    rc_ax.set(ylabel='Depth (m)',ylim=(deep,shallow))
    omega_ax.set_xlabel(r'Angular velocity $\mathbf{\Omega}$ (s$\mathbf{^{-1}}$)',color=omega_color, fontweight='bold')
    rc_ax.xaxis.label.set_color(radius_color)
    rc_ax.tick_params(axis='x',colors=radius_color)
    omega_ax.tick_params(axis='x',colors=omega_color)
    omega_ax.xaxis.get_offset_text().set_color(omega_color)
    rc_ax.spines['bottom'].set_color(radius_color)
    omega_ax.spines['top'].set_color(omega_color)
    rc_ax.set_ylim(None, 0)

    if not include_track:
        for ax,label in ((section_ax,'b)'),(rc_ax,'c)')):
            ax.text(-.10,1.03,label,transform=ax.transAxes,fontweight='bold',fontsize=12)
        return fig

    track_ax=fig.add_subplot(grid[2,0]);track_cax=fig.add_subplot(grid[2,1])
    # Retain NaN breaks rather than connecting across any rejected inner bins.
    # Draw deep segments first so shallower segments overlie them at crossings.
    bird_track=tracked.iloc[::-1]
    xy=bird_track[['east_offset_km','north_offset_km']].to_numpy(float)
    z=bird_track.depth_m.to_numpy(float)
    depth_norm=Normalize(shallow,deep)
    depth_cmap=plt.get_cmap('cool_r')
    depth_span=max(deep-shallow,1.)
    # Give every segment AND sampling dot its own depth-based priority. A
    # separate all-dot layer would put deep dots on top of shallow segments.
    for i in range(len(z)):
        if i and np.isfinite(xy[i-1:i+1]).all():
            segment_depth=(z[i-1]+z[i])/2
            track_ax.plot(xy[i-1:i+1,0],xy[i-1:i+1,1],
                color=depth_cmap(depth_norm(segment_depth)),lw=2.5,
                zorder=2+(deep-segment_depth)/depth_span)
        # if np.isfinite(xy[i]).all():
        #     track_ax.scatter(xy[i,0],xy[i,1],color=depth_cmap(depth_norm(z[i])),
        #         marker='.',s=30,zorder=2+(deep-z[i])/depth_span)
    points=ScalarMappable(norm=depth_norm,cmap=depth_cmap)
    # track_ax.scatter(*xy[-1],marker='o',s=65,facecolors='none',edgecolors='black',zorder=4,label=f'{shallow:g} m')
    # track_ax.scatter(*xy[0],marker='x',s=50,color='black',zorder=4,label=f'{deep:g} m')
    track_ax.autoscale_view();track_ax.margins(.12)
    track_ax.set(xlabel=r'$x_c$ offset (km)',
                 ylabel=r'$y_c$ offset (km)')
    track_ax.set_aspect('equal',adjustable='box')
    # track_ax.legend(loc='best',fontsize=8)
    colorbar=fig.colorbar(points,cax=track_cax,label='Depth (m)')
    colorbar.ax.invert_yaxis()
    # Follow the actual equal-aspect plot height, not the full GridSpec row.
    from matplotlib.transforms import Bbox
    def depth_colorbar_position(bar_ax,renderer):
        plot_box=track_ax.get_position()
        bar_box=bar_ax.get_position(original=True)
        height=.95*plot_box.height
        return Bbox.from_bounds(bar_box.x0,plot_box.y0+(plot_box.height-height)/2,
                                bar_box.width,height)
    track_cax.set_axes_locator(depth_colorbar_position)
    for ax,label in ((section_ax,'b)'),(rc_ax,'c)'),(track_ax,'d)')):
        ax.text(-.10,1.03,label,transform=ax.transAxes,fontweight='bold',fontsize=12)
    return fig


def slices_3d(results,samples,settings,*,figure=None,subplot_spec=None,include_colorbar=True):
    supported=results[results.outer_valid]
    if supported.empty:
        raise ValueError('No supported reconstruction available for 3D slices')
    requested=settings['slice_depths_m']
    if requested is None:
        indices=np.unique(np.linspace(0,len(supported)-1,min(3,len(supported))).astype(int))
        depths=supported.iloc[indices].depth_m.tolist()
    else:
        if len(requested)!=3:
            raise ValueError('Specify three slice depths, or null for automatic selection')
        depths=[select_depth(samples,z) for z in requested]
        if len(set(depths))!=3:
            raise ValueError('Slice depths must select distinct measured bins')
        if not all(z in supported.depth_m.values for z in depths):
            raise ValueError('An explicitly selected 3D slice lacks a supported outer fit')
    # All settings below are display-only; this function never fits parameters.
    positive = ['slice_speed_max_m_s','slice_contour_step_m_s',
                'slice_observed_quiver_scale','slice_model_quiver_scale',
                'slice_quiver_width','slice_model_quiver_spacing',
                'slice_centreline_width','slice_grid_points']
    for key in positive:
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    for key in ('slice_grid_points','slice_model_quiver_spacing'):
        if not isinstance(settings[key],int):
            raise ValueError(f'{key} must be an integer')
    if settings['slice_grid_points'] < 3:
        raise ValueError('slice_grid_points must be >=3')
    if not np.isfinite([settings['slice_elevation_deg'],settings['slice_azimuth_deg']]).all():
        raise ValueError('Slice viewing angles must be finite')
    if settings['slice_contour_step_m_s'] > settings['slice_speed_max_m_s']:
        raise ValueError('Slice contour step must not exceed the speed maximum')
    axis=np.linspace(*settings['x_limits_km'],settings['slice_grid_points'])
    yaxis=np.linspace(*settings['y_limits_km'],settings['slice_grid_points'])
    xx,yy=np.meshgrid(axis,yaxis)
    levels=np.arange(0,settings['slice_speed_max_m_s']+settings['slice_contour_step_m_s']*.5,
                     settings['slice_contour_step_m_s'])
    if levels[-1] < settings['slice_speed_max_m_s']:
        levels=np.r_[levels,settings['slice_speed_max_m_s']]
    norm=Normalize(0,settings['slice_speed_max_m_s'])
    fig=figure if figure is not None else plt.figure(figsize=(11,10),layout='constrained')
    ax=fig.add_subplot(111 if subplot_spec is None else subplot_spec,
                       projection='3d',computed_zorder=False)
    theta=np.linspace(0,2*np.pi,180)
    span=settings['x_limits_km'][1]-settings['x_limits_km'][0]
    # Match 2D quiver convention: bigger scale -> shorter arrows.
    observed_length=span/settings['slice_observed_quiver_scale']
    model_length=span/settings['slice_model_quiver_scale']
    # Keep the standalone arrow weight when composing into a wider figure.
    linewidth=settings['slice_quiver_width']*11*72
    stride=settings['slice_model_quiver_spacing']
    for layer_index,z in enumerate(sorted(depths,reverse=True)):
        layer_order=10*layer_index
        row=supported[supported.depth_m==z].iloc[0];sample=samples[z]
        u,v=esp.model_uv_at_xy(xx*1000,yy*1000,row.xc,row.yc,row.Q,row.Omega,row.Rc)
        # Opaque contourf layers: same palette and discrete fills as panel a).
        layer=ax.contourf(xx,yy,np.hypot(u,v),zdir='z',offset=z,
                         levels=levels,cmap='Blues_r',norm=norm,extend='max',zorder=layer_order)
        layer.set_edgecolor('face')
        layer.set_linewidth(0)
        layer.set_antialiased(False)
        ax.quiver(xx[::stride,::stride],yy[::stride,::stride],
            np.full_like(xx[::stride,::stride],z),u[::stride,::stride],v[::stride,::stride],
            np.zeros_like(u[::stride,::stride]),color='grey',alpha=.7,
            length=model_length,arrow_length_ratio=.2,linewidth=linewidth*.6,zorder=layer_order+2)
        for mask,color in [(sample.finite,'black'),(sample.finite & sample.outer_used,'green'),
                           (sample.finite & sample.inner,'magenta')]:
            d=sample.loc[mask]
            ax.quiver(d.x_m/1000,d.y_m/1000,np.full(len(d),z),d.u,d.v,np.zeros(len(d)),
                      color=color,length=observed_length,arrow_length_ratio=.2,
                      linewidth=linewidth,zorder=layer_order+3)
        radius=row.Rc/np.sqrt(2)/1000
        ax.plot(row.xc/1000+radius*np.cos(theta),row.yc/1000+radius*np.sin(theta),
                np.full_like(theta,z),'r--',lw=2,zorder=layer_order+4, alpha=.7)
        if z == min(depths):
            # ax.scatter(row.xc/1000,row.yc/1000,z,marker='x',color='r',s=50,zorder=layer_order+5)
            ax.scatter(row.xc/1000,row.yc/1000,z,marker='.',color='gold',s=80,zorder=layer_order+5)
        # Place the glyphs in the horizontal plane, rather than facing the camera.
        from matplotlib.textpath import TextPath
        from matplotlib.patches import PathPatch
        from mpl_toolkits.mplot3d.art3d import pathpatch_2d_to_3d
        label_span=min(np.ptp(settings['x_limits_km']),np.ptp(settings['y_limits_km']))
        label_x=settings['x_limits_km'][0]+.05*label_span
        label_y=settings['y_limits_km'][0]+.05*label_span
        label_path=TextPath((label_x,label_y),f'{z:g} m',size=.04*label_span,
                            prop={'weight':'bold'})
        label_patch=PathPatch(label_path,facecolor='white',edgecolor='none',zorder=layer_order+7)
        ax.add_patch(label_patch)
        pathpatch_2d_to_3d(label_patch,z=z,zdir='z')
    ax.set(ylabel='y (km)',
           xlim=settings['x_limits_km'],ylim=settings['y_limits_km'],
           zlim=(results.loc[results.inner_valid,'depth_m'].max()+30,0),)
    ax.set_xlabel('x (km)', labelpad=0); ax.tick_params(axis='x', pad=0)
    ax.set_zlabel('Depth (m)', labelpad=8)
        #    title='SOLO horizontal slices and centreline — vertical scale exaggerated')
    for limits,set_ticks in ((settings['x_limits_km'],ax.set_xticks),
                             (settings['y_limits_km'],ax.set_yticks)):
        set_ticks(np.arange(np.ceil(limits[0]/50)*50,
                            np.floor(limits[1]/50)*50+1,50))
    ax.set_box_aspect((1,1,1.6))
    ax.view_init(elev=settings['slice_elevation_deg'],azim=settings['slice_azimuth_deg'])

    # Two continuous opaque paths, with a fixed painter's order:
    # bottom slice -> lower line -> middle slice -> upper line -> top slice.
    # This intentionally targets the configured publication view, not arbitrary
    # camera-dependent occlusion. No fitted centres or depth values are changed.
    points=results.loc[results.inner_valid,['xc','yc','depth_m']].to_numpy(float,copy=True)
    points[:,:2]/=1000
    middle_depth=float(np.median(depths))
    # Normally the middle slice is a measured fitted depth. For a two-slice
    # fallback, insert only a display junction on the existing polyline.
    if len(points) and not np.any(points[:,2]==middle_depth):
        insertion=int(np.searchsorted(points[:,2],middle_depth))
        if 0 < insertion < len(points):
            first,second=points[insertion-1:insertion+1]
            fraction=(middle_depth-first[2])/(second[2]-first[2])
            points=np.insert(points,insertion,first+fraction*(second-first),axis=0)
    lower=points[points[:,2]>=middle_depth]
    upper=points[points[:,2]<=middle_depth]
    lower_line,=ax.plot(*lower.T,color='orange',alpha=1.,
        lw=settings['slice_centreline_width'],zorder=8,label='Accepted inner centreline')
    upper_line,=ax.plot(*upper.T,color='orange',alpha=1.,
        lw=settings['slice_centreline_width'],zorder=18)
    fig._solo_centreline_paths=(lower_line,upper_line)
    # ax.legend()
    if include_colorbar:
        fig.colorbar(ScalarMappable(norm=norm,cmap='Blues_r'),ax=ax,
                     label=r'Current speed (ms$^{-1}$)',shrink=.5,pad=0.0)
    return fig,depths


def combined_manuscript_figure(results,samples,section_data,settings):
    """Compose the existing 3D view and companion panels without refitting."""
    fig=plt.figure(figsize=(16.5,10),layout='constrained')
    fig.get_layout_engine().set(wspace=0.0, w_pad=0.0)
    grid=fig.add_gridspec(1,2,width_ratios=[2,1])
    slices_3d(results,samples,settings,figure=fig,subplot_spec=grid[0,0],
              include_colorbar=True)
    # The 3D axes retain a square drawing area inside a wider GridSpec cell.
    # Align that area towards the companion panels instead of centring it.
    fig.axes[0].set_anchor('E')
    fig.axes[0].text2D(.25,.94,'a)',transform=fig.axes[0].transAxes,
                       ha='right',va='top',fontweight='bold',fontsize=12)
    manuscript_panels(results,section_data,figure=fig,subplot_spec=grid[0,1],
                       include_track=False)
    return fig
