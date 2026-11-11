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
        ax.plot(l/1000,curve,color='k',lw=1)
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
        ax.set(title=f'{z:g} m',xlabel='East (km)',ylabel='North (km)',
               xlim=settings['x_limits_km'],ylim=settings['y_limits_km'],aspect='equal')
    fig.colorbar(ScalarMappable(norm=norm,cmap='Blues'),ax=list(axes.flat),label='Reconstructed speed (m/s)',shrink=.8)
    fig.suptitle('SOLO · magenta: inner · green: outer (inner colour takes priority) · black: other\ngrey: reconstructed · red: centre / fitted peak')
    return fig,depths


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
    axis=np.linspace(*settings['x_limits_km'],75);yaxis=np.linspace(*settings['y_limits_km'],75)
    xx,yy=np.meshgrid(axis,yaxis);norm=Normalize(0,settings['speed_max_m_s'])
    fig=plt.figure(figsize=(11,8),layout='constrained');ax=fig.add_subplot(111,projection='3d',computed_zorder=False)
    theta=np.linspace(0,2*np.pi,180)
    for z in depths:
        row=supported[supported.depth_m==z].iloc[0];sample=samples[z]
        u,v=esp.model_uv_at_xy(xx*1000,yy*1000,row.xc,row.yc,row.Q,row.Omega,row.Rc)
        layer=ax.contourf(xx,yy,np.hypot(u,v),zdir='z',offset=z,
                    levels=np.linspace(0,settings['speed_max_m_s'],31),
                    cmap='Blues_r',norm=norm,alpha=.65,extend='max',zorder=0)
        layer.set_edgecolor('none')
        layer.set_linewidth(0)
        ax.quiver(xx[::10,::10],yy[::10,::10],np.full_like(xx[::10,::10],z),
                  u[::10,::10],v[::10,::10],np.zeros_like(u[::10,::10]),
                  color='grey',alpha=.4,length=12,arrow_length_ratio=.2,linewidth=.5,zorder=2)
        # Horizontal arrows are in km per m/s; vertical velocity is not inferred.
        for mask,color in [(sample.finite,'black'),(sample.finite & sample.outer_used,'green'),
                           (sample.finite & sample.inner,'magenta')]:
            d=sample.loc[mask]
            ax.quiver(d.x_m/1000,d.y_m/1000,np.full(len(d),z),d.u,d.v,np.zeros(len(d)),
                      color=color,length=12,arrow_length_ratio=.2,linewidth=.7,zorder=3)
        radius=row.Rc/np.sqrt(2)/1000
        ax.plot(row.xc/1000+radius*np.cos(theta),row.yc/1000+radius*np.sin(theta),np.full_like(theta,z),'r--',lw=1.5,zorder=4)
        ax.scatter(row.xc/1000,row.yc/1000,z,marker='x',color='r',s=35,zorder=5)
        ax.text(settings['x_limits_km'][0],settings['y_limits_km'][1],z,f'{z:g} m')
    ax.plot(results.xc/1000,results.yc/1000,results.depth_m,'k.-',lw=2,label='Accepted inner centreline',zorder=6)
    ax.set(xlabel='East (km)',ylabel='North (km)',zlabel='Depth (m)',
           xlim=settings['x_limits_km'],ylim=settings['y_limits_km'],
           zlim=(results.loc[results.inner_valid,'depth_m'].max()+30,0),
           title='SOLO horizontal slices and centreline — vertical scale exaggerated')
    ax.set_xticks(np.linspace(*settings['x_limits_km'],5)[1:-1])
    ax.set_yticks(np.linspace(*settings['y_limits_km'],5)[1:-1])
    ax.set_box_aspect((1,1,1.3));ax.view_init(elev=20,azim=-60);ax.legend()
    fig.colorbar(ScalarMappable(norm=norm,cmap='Blues_r'),ax=ax,label='Reconstructed speed (m/s)',shrink=.5,pad=.12)
    return fig,depths
