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
    row, stored = rows.iloc[0],fits[depth]
    fig,axes = plt.subplots(1,3,figsize=(14,4),layout='constrained')
    l = stored['l']; p = stored['inner']; dl=l-p['x0']; mask=np.asarray(p['core_mask'],bool)
    for ax,values,curve,label in zip(axes[:2],(stored['VN'],stored['VT']),
            (p['C']*dl+p['D']*dl**3,p['A']+p['B']*dl**2),('VN','VT')):
        ax.scatter(l/1000,values,color='.65',s=15,label='Search observations')
        ax.scatter(l[mask]/1000,values[mask],color='m',s=18,label='Inner fit')
        ylim = ax.get_ylim()
        ax.plot(l/1000,curve,color='k',lw=1)
        ax.set_ylim(ylim)
        ax.axvline(p['x0']/1000,color='k',ls='--',lw=.8)
        ax.set(xlabel='Along-track coordinate (km)',ylabel=f'{label} (m/s)')
    axes[0].legend(fontsize=8)
    if stored['outer'].get('status') == 'optimized':
        esp.plot_outer_core_fit(stored['outer'],ax=axes[2],km_flag=True,pred_flag=True)
        axes[2].set_title(f'Outer fit · Rc={row.candidate_Rc_m/1000:.1f} km',fontsize=10)
        if not row.outer_valid:
            axes[2].set_title('Candidate only: '+row.status.replace('_',' '))
    else:
        axes[2].text(.05,.5,row.status.replace('_',' '),transform=axes[2].transAxes)
    fig.suptitle(f'SOLO stored fit — {depth:g} m')
    return fig


def profiles(results):
    fig,axes = plt.subplots(1,4,figsize=(14,5),sharey=True,layout='constrained')
    for ax,key,label,scale in zip(axes,['east_offset_km','north_offset_km','Omega','Rc'],
            ['Eastward offset (km)','Northward offset (km)','Final Ω (s⁻¹)','Supported Rc (km)'],[1,1,1,1000]):
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


def vertical_sections(results,settings):
    """Signed cross-section velocities through the fixed 37-m centre.

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
    x_roots=np.full((len(depths),2),np.nan)
    y_roots=np.full((len(depths),2),np.nan)
    for i,row in enumerate(results.itertuples()):
        if not row.outer_valid:
            continue
        _,zonal[i]=esp.model_uv_at_xy(x,np.full_like(x,y_plane),row.xc,row.yc,row.Q,row.Omega,row.Rc)
        meridional[i],_=esp.model_uv_at_xy(np.full_like(y,x_plane),y,row.xc,row.yc,row.Q,row.Omega,row.Rc)
        x_roots[i],y_roots[i]=_section_peak_intersections(row.xc,row.yc,row.Q,row.Rc,x_plane,y_plane)
    limit=settings['section_velocity_max_m_s']
    if limit is None:
        finite=np.r_[zonal[np.isfinite(zonal)],meridional[np.isfinite(meridional)]]
        limit=max(step,np.ceil(np.max(np.abs(finite))/step)*step) if len(finite) else step
    if not np.isfinite(limit) or limit<=0:
        raise ValueError('section_velocity_max_m_s must be null or positive and finite')
    count=max(1,int(np.ceil(limit/step)))
    levels=np.linspace(-limit,limit,2*count+1)
    fig,axes=plt.subplots(1,2,figsize=(12,6),sharey=True,layout='constrained')
    for ax,coordinate,values,roots,centre,label,title,component in zip(axes,
            (x,y),(zonal,meridional),(x_roots,y_roots),(results.xc,results.yc),
            ('x (km)','y (km)'),
            (f'Zonal section · y={y_plane/1000:.2f} km',f'Meridional section · x={x_plane/1000:.2f} km'),
            ('Northward velocity v (m/s)','Eastward velocity u (m/s)')):
        if len(depths)>=2 and np.isfinite(values).any():
            image=ax.contourf(coordinate/1000,depths,np.ma.masked_invalid(values),levels=levels,
                             cmap='RdBu_r',extend='both',corner_mask=False)
            fig.colorbar(image,ax=ax,location='top',label=component)
        else:
            ax.text(.5,.5,'No supported depth reconstruction',ha='center',transform=ax.transAxes)
        ax.plot(centre.where(results.inner_valid)/1000,depths,color='orange',lw=2,label='Projected centreline')
        for branch in range(2):
            ax.plot(roots[:,branch]/1000,depths,'r--',lw=1.5,
                    label='Peak-contour intersection' if branch==0 else None)
        ax.set(xlabel=label,title=title,xlim=(coordinate[0]/1000,coordinate[-1]/1000))
    axes[0].set_ylabel('Depth (m)')
    axes[0].set_ylim(depths.max(),depths.min())
    axes[0].legend(loc='lower left',fontsize=8)
    fig.suptitle('Fixed section planes through the 37-m centre')
    # Saved display data make plane locations, signed components and actual
    # contour intersections inspectable without another estimator call.
    data=dict(depth_m=depths,x_m=x,y_m=y,zonal_v_m_s=zonal,meridional_u_m_s=meridional,
              x_peak_roots_m=x_roots,y_peak_roots_m=y_roots,
              x_plane_m=x_plane,y_plane_m=y_plane,velocity_limit_m_s=limit)
    return fig,data


def slices_3d(results,samples,settings):
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
    fig=plt.figure(figsize=(11,10),layout='constrained')
    ax=fig.add_subplot(111,projection='3d',computed_zorder=False)
    theta=np.linspace(0,2*np.pi,180)
    span=settings['x_limits_km'][1]-settings['x_limits_km'][0]
    # Match 2D quiver convention: bigger scale -> shorter arrows.
    observed_length=span/settings['slice_observed_quiver_scale']
    model_length=span/settings['slice_model_quiver_scale']
    linewidth=settings['slice_quiver_width']*fig.get_figwidth()*72
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
            ax.scatter(row.xc/1000,row.yc/1000,z,marker='.',color='orange',s=80,zorder=layer_order+5)
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
    ax.set(xlabel='x (km)',ylabel='y (km)',zlabel='Depth (m)',
           xlim=settings['x_limits_km'],ylim=settings['y_limits_km'],
           zlim=(results.loc[results.inner_valid,'depth_m'].max()+30,0),)
        #    title='SOLO horizontal slices and centreline — vertical scale exaggerated')
    ax.set_xticks(np.linspace(*settings['x_limits_km'],5)[1:-1])
    ax.set_yticks(np.linspace(*settings['y_limits_km'],5)[1:-1])
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
    fig.colorbar(ScalarMappable(norm=norm,cmap='Blues_r'),ax=ax,
                 label=r'Current speed (ms$^{-1}$)',shrink=.5,pad=0.0)
    return fig,depths
