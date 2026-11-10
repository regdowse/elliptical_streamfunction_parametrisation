import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def decode_ce23_mat_time(tim_sadcp):
    """Decode the original, uncorrected CE23 MAT timestamps as UTC labels.

    Standard astronomical Julian conversion gives labels 12 h ahead of the
    independently compared CSIRO Data Trawler UTC positions/velocities for
    this dataset. Apply that empirical correction exactly once here. The raw
    MAT values are preserved; returned datetime64 values represent UTC.
    Do not use for already-corrected data or other instruments/products.
    """
    ref = datetime(2000, 1, 1, 12)
    time = np.array([
        ref + timedelta(days=float(jd) - 2451545.0)
        for jd in np.asarray(tim_sadcp).ravel()
    ], dtype="datetime64[ns]")
    return time - np.timedelta64(12, "h")


def interpolate_satellite_at_time(ds_sat, time):
    """Interpolate native satellite samples to a UTC target within coverage."""
    time = pd.Timestamp(time).to_datetime64()
    times = ds_sat.time.values
    if (not times.size or np.isnat(time)
            or time < times.min() or time > times.max()):
        raise ValueError(f"Satellite time coverage does not bracket ADCP target {time}; use native satellite samples.")
    sat = ds_sat.interp(time=time)
    valid = np.isfinite(sat.u.values) & np.isfinite(sat.v.values)
    if not np.any(valid):
        raise ValueError(f"No finite satellite velocity pairs at ADCP target {time}.")
    return sat


# SOLO 
def solo(l, VT, VN, core_thresh=30_000, plot=False, ax=None, selection_details=None):
    from scipy.optimize import curve_fit
    '''
    l: The along track distance, 0 at the beginning
    VT: The tangential velocity i.e., the along-track velocity 
    VN: The normal velocity i.e., the across-track velocity 

    parameters:
    x0 is the point of closest approach (i.e., VN(x0)=0)
    l0 and r0 is the eddy center postion along and across the transect, respectively.
    w is the inner-core vorticity
    Q is the deformation matrix
    Omega is the angular velocity
    '''
    
    l, VT, VN = map(np.asarray, (l, VT, VN))
    m = np.isfinite(l) & np.isfinite(VT) & np.isfinite(VN)
    l, VT, VN = l[m], VT[m], VN[m]

    if l.size < 4:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    def vn_model(l, x0, C, D):
        dl = l - x0
        return C * dl + D * dl**3

    def vt_model(l, x0, A, B):
        dl = l - x0
        return A + B * dl**2

    def fit_x0_from_vn(l, VN):
        x0_guess = l[np.argmin(np.abs(VN))]
        C_guess = 0.0
        D_guess = 0.0

        try:
            popt, _ = curve_fit(
                vn_model, l, VN,
                p0=[x0_guess, C_guess, D_guess],
                maxfev=10000
            )
            return popt
        except Exception:
            return np.array([np.nan, np.nan, np.nan])

    x0, C, D = fit_x0_from_vn(l, VN)
    if not np.isfinite(x0):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # focus on core
    mask = np.abs(l - x0) <= core_thresh
    if selection_details is not None:
        core_mask = np.zeros(m.shape, dtype=bool)
        core_mask[np.flatnonzero(m)[mask]] = True
        selection_details["core_mask"] = core_mask
        selection_details["l_bounds"] = (x0 - core_thresh, x0 + core_thresh)
    l, VT, VN = l[mask], VT[mask], VN[mask]

    # refit
    x0, C, D = fit_x0_from_vn(l, VN)
    if not np.isfinite(x0):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    dl = l - x0

    Xu = np.vstack([np.ones_like(dl), dl**2]).T
    A, B = np.linalg.lstsq(Xu, VT, rcond=None)[0]

    l0 = x0 # the eddy center's along track position
    r0 = A / C # the eddy center's across track position
    Q = np.array([[1., 0.], [0., 1.]])
    Omega =  C 
    w = 2 * Omega

    if plot:
        lfit = np.linspace(l.min(), l.max(), 500)

        if ax is None:
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        else:
            axs = ax

        axs = np.asarray(axs).ravel()

        axs[0].scatter(l, VN)
        axs[0].plot(lfit, vn_model(lfit, x0, C, D))
        axs[0].axvline(x0, linestyle='--')
        axs[0].axhline(0, linewidth=0.8)
        axs[0].set_xlabel('l')
        axs[0].set_ylabel('VN')
        axs[0].set_title('VN fit')

        axs[1].scatter(l, VT)
        axs[1].plot(lfit, vt_model(lfit, x0, A, B))
        axs[1].axvline(x0, linestyle='--')
        axs[1].set_xlabel('l')
        axs[1].set_ylabel('VT')
        axs[1].set_title('VT fit')

        if ax is None:
            plt.tight_layout()
            plt.show()

    return l0, r0, w, Q, Omega


def project_sadcp_to_transect(x, y, u, v):
    '''
    Project quasi-straight data to transect data i.e., length along transect (l), along-track (VT) and across-track (VN) velocities
    '''
    x, y, u, v = map(np.asarray, (x, y, u, v))
    msk = np.isfinite(x) & np.isfinite(y) & np.isfinite(u) & np.isfinite(v)
    x, y, u, v = x[msk], y[msk], u[msk], v[msk]
    n = x.size
    if n == 0:
        return pd.DataFrame(columns=["x","y","u","v","V_N","V_T","l"]), np.nan

    # best-fit line y = m x + c
    A = np.c_[x, np.ones(n)]
    m, c0 = np.linalg.lstsq(A, y, rcond=None)[0]

    # unit tangent along line and normal (right-hand)
    t = np.array([1.0, m]); t /= np.linalg.norm(t)
    nvec = np.array([-t[1], t[0]])

    p0 = np.array([x[0], y[0]])
    pts = np.c_[x, y]
    s = (pts - p0) @ t          # signed along-track coordinate

    idx = np.argsort(s)
    s, u, v = s[idx], u[idx], v[idx]

    s_new = np.linspace(s.min(), s.max(), n)
    u_new = np.interp(s_new, s, u)
    v_new = np.interp(s_new, s, v)

    pts_new = p0 + np.outer(s_new, t)
    x_new, y_new = pts_new[:, 0], pts_new[:, 1]

    VT = u_new*t[0] + v_new*t[1]
    VN = u_new*nvec[0] + v_new*nvec[1]

    df = pd.DataFrame({"x": x_new, "y": y_new, "u": u_new, "v": v_new, "V_N": VN, "V_T": VT})
    df["l"] = s_new - s_new.min()   # 0 at start, increasing along transect

    return df, m


def translate_solo_results(x_l_start, y_l_start, m, l0, r0):
    '''
    Function to translate tranect eddy center (l0, r0) to cartesian eddy center (xc, yc)
    '''
    denom = np.sqrt(1 + m**2)
    xc = (l0 - r0*m)/denom + x_l_start
    yc = (l0*m + r0)/denom + y_l_start
    return xc, yc

# DOPPIO

def doppio(x1, y1, u1, v1, x2, y2, u2, v2, plot=False, fit_details=None):
    '''
    1st transect (i.e., zonal) data: x1, y1, u1, v1
    2nd transect (i.e., meridional) data: x2, y2, u2, v2

    parameters:
    x0, y0 are inflection points (i.e., v1(x0)=0 and u2(y0)=0)
    (xc, yc) is the eddy center
    w is the inner-core vorticity
    Q is the deformation matrix
    Omega is the angular velocity
    '''

    if fit_details is not None:
        fit_details.clear()

    def nan_return(reason="invalid_fit"):
        if fit_details is not None:
            fit_details["status"] = reason
        nan2 = np.full((2, 2), np.nan)
        coeffs = pd.DataFrame(
            np.nan,
            index=['0', '1', '2', '3'],
            columns=['A', 'B', 'C', 'D']
        )
        return np.nan, np.nan, np.nan, np.full((2,2), np.nan), np.nan

    def clean(*arrs):
        arrs = [np.asarray(a, dtype=float).ravel() for a in arrs]
        m = np.logical_and.reduce([np.isfinite(a) for a in arrs])
        return [a[m] for a in arrs]

    def poly3(p, z):
        return p[0] + p[1]*z + p[2]*z**2 + p[3]*z**3

    def find_root(x, y, degree=3):
        if x.size < degree + 1:
            return np.nan

        try:
            coeffs = np.polyfit(x, y, degree)
            roots = np.roots(coeffs)
        except Exception:
            return np.nan

        real_roots = roots[np.isreal(roots)].real
        if real_roots.size == 0:
            return np.nan

        mid_x = np.median(x)
        return real_roots[np.argmin(np.abs(real_roots - mid_x))]

    x1, y1, u1, v1 = clean(x1, y1, u1, v1)
    x2, y2, u2, v2 = clean(x2, y2, u2, v2)

    if len(x1) < 4 or len(y2) < 4:
        return nan_return()

    pts1 = np.column_stack((x1, y1))
    pts2 = np.column_stack((x2, y2))

    common = np.array([p for p in pts1 if np.any(np.all(pts2 == p, axis=1))])

    if len(common) != 1:
        return nan_return()

    center_x, center_y = common[0]

    # Set origin at transect center
    x = x1 - center_x
    y = y2 - center_y

    # Initial guesses
    # Unconstrained cubic fits used only to estimate x0 and y0
    x0_guess = find_root(x, v1)
    y0_guess = find_root(y, u2)
    if not np.isfinite([x0_guess, y0_guess]).all():
        return nan_return("no_real_polynomial_root")

    pA = np.polyfit(x, u1, 3)[::-1]
    pB = np.polyfit(x, v1, 3)[::-1]
    pC = np.polyfit(y, u2, 3)[::-1]
    pD = np.polyfit(y, v2, 3)[::-1]

    # Free parameters:
    # x0, y0,
    # A0,A1,A2,A3,
    # B1,B2,B3,   with B0 set so V1(x0)=0 exactly
    # C1,C2,C3,   with C0 set so U2(y0)=0 exactly
    # D0,D2,D3,   with D1 set so d1=-a1 exactly
    p0 = np.array([
        x0_guess, y0_guess,
        pA[0], pA[1], pA[2], pA[3],
        pB[1], pB[2], pB[3],
        pC[1], pC[2], pC[3],
        pD[0], pD[2], pD[3]
    ], dtype=float)

    def unpack(p):
        x0, y0 = p[0], p[1]

        A0, A1, A2, A3 = p[2:6]
        B1, B2, B3 = p[6:9]
        C1, C2, C3 = p[9:12]
        D0, D2, D3 = p[12:15]

        a1 = A1 + 2*A2*x0 + 3*A3*x0**2
        b1 = B1 + 2*B2*x0 + 3*B3*x0**2
        c1 = C1 + 2*C2*y0 + 3*C3*y0**2

        # These are unshifted cubics. Subtract the polynomial value, not
        # root * derivative; the latter is only correct for a linear function.
        B0 = -(B1*x0 + B2*x0**2 + B3*x0**3)  # V1(x0)=0
        C0 = -(C1*y0 + C2*y0**2 + C3*y0**3)  # U2(y0)=0
        D1 = -a1 - 2*D2*y0 - 3*D3*y0**2 # d1=-a1

        A = np.array([A0, A1, A2, A3], dtype=float)
        B = np.array([B0, B1, B2, B3], dtype=float)
        C = np.array([C0, C1, C2, C3], dtype=float)
        D = np.array([D0, D1, D2, D3], dtype=float)

        return x0, y0, A, B, C, D

    def residuals(p):
        x0, y0, A, B, C, D = unpack(p)

        U1 = poly3(A, x)
        V1 = poly3(B, x)
        U2 = poly3(C, y)
        V2 = poly3(D, y)

        return np.r_[U1 - u1, V1 - v1, U2 - u2, V2 - v2]

    try:
        res = least_squares(residuals, p0, method='trf', x_scale='jac')
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return nan_return("optimizer_failed")

    if not res.success:
        return nan_return("optimizer_failed")

    x0, y0, A, B, C, D = unpack(res.x)

    A0, A1, A2, A3 = A
    B0, B1, B2, B3 = B
    C0, C1, C2, C3 = C
    D0, D1, D2, D3 = D

    # Tangent coefficients
    a1 = A1 + 2*A2*x0 + 3*A3*x0**2
    a0 = (A0 + A1*x0 + A2*x0**2 + A3*x0**3) - x0*a1

    b1 = B1 + 2*B2*x0 + 3*B3*x0**2
    b0 = -x0*b1

    c1 = C1 + 2*C2*y0 + 3*C3*y0**2
    c0 = -y0*c1

    d1 = D1 + 2*D2*y0 + 3*D3*y0**2
    d0 = (D0 + D1*y0 + D2*y0**2 + D3*y0**3) - y0*d1

    # Enforce d1 = -a1 numerically if tiny mismatch remains
    d1 = -a1

    radicand = -b1*c1 - a1**2
    if radicand <= 0 or not np.isfinite(radicand):
        return nan_return()

    Omega = np.sign(b1) * np.sqrt(radicand)
    if Omega == 0 or not np.isfinite(Omega):
        return nan_return()

    Q = (1 / Omega) * np.array([
        [b1,  -a1],
        [-a1, -c1]
    ], dtype=float)

    xc = (a0*a1 + c1*d0) / Omega**2
    yc = (a0*b1 - a1*d0) / Omega**2

    w = Omega * (Q[0,0] + Q[1,1])
    if not is_elliptical_Q(Q):
        return nan_return("invalid_geometry")
    if fit_details is not None:
        fit_details.update(status="optimized", x_root=x0, y_root=y0,
                           A=A.copy(), B=B.copy(), C=C.copy(), D=D.copy(),
                           coordinate_origin=(center_x, center_y))

    coeffs = pd.DataFrame(
        {
            'A': A,
            'B': B,
            'C': C,
            'D': D
        },
        index=['0', '1', '2', '3']
    )

    yu1 = a0 + a1*x
    yv1 = b0 + b1*x
    yu2 = c0 + c1*y
    yv2 = d0 + d1*y

    if plot:
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))

        xs = np.linspace(x.min(), x.max(), 400)
        ys = np.linspace(y.min(), y.max(), 400)

        U1s = poly3(A, xs)
        V1s = poly3(B, xs)
        U2s = poly3(C, ys)
        V2s = poly3(D, ys)

        axs[0, 0].scatter(x, u1, s=10)
        axs[0, 0].plot(xs, U1s)
        axs[0, 0].plot(x, yu1)
        axs[0, 0].axvline(x0, ls='--')
        axs[0, 0].set_title("U1(x)")
        axs[0, 0].set_xlabel("x (m)")
        axs[0, 0].set_ylabel(r"(ms$^{-1}$)")

        axs[0, 1].scatter(x, v1, s=10)
        axs[0, 1].plot(xs, V1s)
        axs[0, 1].plot(x, yv1)
        axs[0, 1].axvline(x0, ls='--')
        axs[0, 1].axhline(0, ls='--')
        axs[0, 1].set_title("V1(x)")
        axs[0, 1].set_xlabel("x (m)")
        axs[0, 1].set_ylabel(r"(ms$^{-1}$)")

        axs[1, 0].scatter(y, u2, s=10)
        axs[1, 0].plot(ys, U2s)
        axs[1, 0].plot(y, yu2)
        axs[1, 0].axvline(y0, ls='--')
        axs[1, 0].axhline(0, ls='--')
        axs[1, 0].set_title("U2(y)")
        axs[1, 0].set_xlabel("y (m)")
        axs[1, 0].set_ylabel(r"(ms$^{-1}$)")

        axs[1, 1].scatter(y, v2, s=10)
        axs[1, 1].plot(ys, V2s)
        axs[1, 1].plot(y, yv2)
        axs[1, 1].axvline(y0, ls='--')
        axs[1, 1].set_title("V2(y)")
        axs[1, 1].set_xlabel("y (m)")
        axs[1, 1].set_ylabel(r"(ms$^{-1}$)")

        plt.tight_layout()
        plt.show()

    return xc + center_x, yc + center_y, w, Q, Omega

# LATTE

def latte(xi, yi, ui, vi, fit_details=None):
    '''
    ui, vi is the zonal and meridional velocity data
    xi, yi is the zonal and meridional positions 

    parameters:
    (xc, yc) is the eddy center
    w is the inner-core vorticity
    Q is the deformation matrix
    Omega is the angular velocity
    '''
    from scipy.optimize import least_squares
    if fit_details is not None:
        fit_details.clear()

    def failed(reason):
        if fit_details is not None:
            fit_details["status"] = reason
        return np.nan, np.nan, np.nan, np.full((2, 2), np.nan), np.nan, np.nan

    xi, yi, ui, vi = map(lambda a: np.asarray(a, float).ravel(), (xi, yi, ui, vi))
    if len({a.size for a in (xi, yi, ui, vi)}) != 1:
        raise ValueError("LATTE position/velocity arrays must have equal lengths.")
    m = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(ui) & np.isfinite(vi)
    x, y, u_i, v_i = xi[m], yi[m], ui[m], vi[m]
    n = x.size
    if n < 3:
        return failed("insufficient_samples")

    # A collinear/degenerate sample cannot identify the full 2-D affine flow.
    scale = max(np.ptp(x), np.ptp(y))
    if scale == 0:
        return failed("rank_deficient_sampling")
    dx, dy = (x-x.mean()) / scale, (y-y.mean()) / scale
    design = np.zeros((2*n, 5))
    design[:n, 0], design[n:, 1] = 1, 1
    design[:n, 3], design[:n, 4] = -dx, -dy
    design[n:, 2], design[n:, 3] = dx, dy
    if np.linalg.matrix_rank(design) < 5:
        return failed("rank_deficient_sampling")

    p0 = np.array([x.mean(), y.mean(), 1., 0., 1.])  # xc,yc,Oq11,Oq12,Oq22

    def fun(p):
        xc,yc,Oq11,Oq12,Oq22 = p
        dx, dy = x-xc, y-yc
        u = -Oq22*dy - Oq12*dx
        v =  Oq11*dx + Oq12*dy
        r = np.empty(2*n); r[:n]=u-u_i; r[n:]=v-v_i
        return r

    def jac(p):
        xc,yc,Oq11,Oq12,Oq22 = p
        dx, dy = x-xc, y-yc
        J = np.zeros((2*n,5))
        # u = -Omega*q22*(y-yc) - Omega*q12*(x-xc)
        J[:n,0], J[:n,1], J[:n,3], J[:n,4] = Oq12, Oq22, -dx, -dy
        # v = Omega*q11*(x-xc) + Omega*q12*(y-yc)
        J[n:,0], J[n:,1], J[n:,2], J[n:,3] = -Oq11, -Oq12, dx, dy
        return J

    try:
        result = least_squares(fun, p0, jac=jac)
    except (ValueError, RuntimeError, np.linalg.LinAlgError):
        return failed("optimizer_failed")
    if not result.success or not np.isfinite(result.x).all():
        return failed("optimizer_failed")
    xc,yc,Oq11,Oq12,Oq22 = result.x

    dx, dy = x-xc, y-yc
    u = -Oq22*dy - Oq12*dx
    v =  Oq11*dx + Oq12*dy
    err2 = ((u-u_i)**2 + (v-v_i)**2).sum()
    tot2 = ((u_i-u_i.mean())**2 + (v_i-v_i.mean())**2).sum()
    r2 = 1 - err2/tot2 if tot2 > 0 else np.nan

    w = Oq11+Oq22
    OQ = np.array([[Oq11,Oq12],[Oq12,Oq22]])
    det = OQ[0,0]*OQ[1,1] - OQ[0,1]*OQ[1,0]
    # OQ must be definite (either rotation sign), never hyperbolic/indefinite.
    if not np.isfinite(det) or det <= 0:
        return failed("non_elliptic_flow")
    Omega = np.sign(Oq11)*np.sqrt(det)
    Q = OQ/Omega if Omega != 0 else np.full((2,2), np.nan)
    if not is_elliptical_Q(Q):
        return failed("invalid_geometry")
    if fit_details is not None:
        fit_details.update(status="optimized", sample_count=n)

    return xc, yc, w, Q, Omega, r2


def is_elliptical_Q(Q, require_det_one=True):
    """True only for a finite symmetric positive-definite ESP geometry."""
    Q = np.asarray(Q, dtype=float)
    if Q.shape != (2, 2) or not np.isfinite(Q).all():
        return False
    if not np.allclose(Q, Q.T, rtol=1e-10, atol=1e-12):
        return False
    if np.linalg.eigvalsh(Q).min() <= 0:
        return False
    return (not require_det_one or
            bool(np.isclose(np.linalg.det(Q), 1, rtol=1e-6, atol=1e-8)))


def refine_latte(xi, yi, ui, vi, center=(0.0, 0.0), Q_pre=None,
                 rho_core=35_000.0, fit_details=None):
    """Two-stage inner fit: refine the centre AND ellipse using stage one's Q.

    Invalid/non-elliptic fits return NaN parameters with a recorded stage/status.
    This function does not advance tracking state; only an accepted fit may do so.
    """
    Q_pre = np.eye(2) if Q_pre is None else np.asarray(Q_pre, float)
    if not is_elliptical_Q(Q_pre) or not np.isfinite(center).all() or rho_core <= 0:
        raise ValueError("A finite centre, positive radius and valid ESP seed Q are required.")
    arrays = [np.asarray(a, float).ravel() for a in (xi, yi, ui, vi)]
    if len({a.size for a in arrays}) != 1:
        raise ValueError("Position/velocity arrays must have equal lengths.")
    x, y, u, v = arrays

    def mask_at(c, Q):
        dx, dy = x-c[0], y-c[1]
        rho2 = Q[0, 0]*dx**2 + 2*Q[0, 1]*dx*dy + Q[1, 1]*dy**2
        return rho2 <= rho_core**2

    initial = mask_at(center, Q_pre)
    stage = {}
    result = latte(x[initial], y[initial], u[initial], v[initial], fit_details=stage)
    details = dict(initial_mask=initial, refined_mask=np.zeros(len(x), dtype=bool),
                   status="initial_" + stage["status"])
    if stage["status"] == "optimized":
        details["preliminary_Q"] = result[3].copy()
        details["preliminary_center"] = result[:2]
        refined = mask_at(result[:2], result[3])
        details["refined_mask"] = refined
        result = latte(x[refined], y[refined], u[refined], v[refined], fit_details=stage)
        details["status"] = "optimized" if stage["status"] == "optimized" else "refined_" + stage["status"]
    if fit_details is not None:
        fit_details.clear()
        fit_details.update(details)
    return result


def tracking_acceptance(previous_center, result, rotation_sign=None, max_step=100_000.0):
    """Accept a valid ellipse only within the step limit from the last accepted centre."""
    xc, yc, w, Q, Omega = result[:5]
    if not np.isfinite([xc, yc, w, Omega]).all() or not is_elliptical_Q(Q):
        return False, "invalid inner fit or non-elliptic geometry"
    step = np.hypot(xc-previous_center[0], yc-previous_center[1])
    if not np.isfinite(step) or step >= max_step:
        return False, f"displacement {step:g} m from last accepted centre exceeds limit {max_step:g} m"
    if w == 0 or (rotation_sign is not None and np.sign(w) != rotation_sign):
        return False, "rotation sign differs from the first accepted fit"
    return True, "accepted"


# Outer-core ESP parameter finder
    
def out_core_param_fit(
    xi, yi, ui, vi, xc, yc, Q,
    Omega0=None, Rc0=None,
    plot=False, ax=None,
    maxfev=10000, Rc_max=None,
    r2_flag=False,
    rho_plot_max=None, n_curve=400,
    km_flag=False,
    ci_flag=False,
    pred_flag=False,
    fit_details=None, core_mask=None
):
    """Fit SI outer-core parameters; failed fits return NaNs, never guesses.

    There is no default upper bound on the fitted Gaussian radius. Rc_max, if
    supplied, is an explicit optimizer bound, independent of the observation
    selection radius. A solution on that bound is reported as unavailable.
    Coordinates/velocities are scaled internally for optimization and covariance
    is converted back to SI. fit_details records status and stored plot data.
    """
    arrays = [np.asarray(a, float).ravel() for a in (xi, yi, ui, vi)]
    if len({a.size for a in arrays}) != 1:
        raise ValueError("Outer-fit position/velocity arrays must have equal lengths.")
    xi, yi, ui, vi = arrays
    Q = np.asarray(Q, dtype=float)
    if core_mask is not None and np.asarray(core_mask).shape != xi.shape:
        raise ValueError("core_mask must match the input observations.")
    if Rc_max is not None and (not np.isfinite(Rc_max) or Rc_max <= 1e-8):
        raise ValueError("Rc_max must be None or a finite positive radius greater than 1e-8 m.")
    if fit_details is not None:
        fit_details.clear()
    details = dict(status="not_fitted", rho=np.array([]), vt=np.array([]),
                   observation_indices=np.array([], dtype=int), core_mask=np.array([], dtype=bool),
                   Omega=np.nan, Rc=np.nan, psi0=np.nan, R2=np.nan,
                   pcov=None, sigma2=np.nan, Rc_standard_error=np.nan)

    def failed(status, message=""):
        details.update(status=status, message=message)
        if fit_details is not None:
            fit_details.update(details)
        return (np.nan, np.nan, np.nan, np.nan) if r2_flag else (np.nan, np.nan, np.nan)

    if not is_elliptical_Q(Q) or not np.isfinite([xc, yc]).all():
        return failed("invalid_geometry")
    dx, dy = xi-xc, yi-yc
    q11, q12, q22 = Q[0, 0], Q[0, 1], Q[1, 1]
    rho2_all = q11*dx**2 + 2*q12*dx*dy + q22*dy**2
    vt_all = tangential_velocity(xi, yi, ui, vi, xc, yc, Q)
    Qr_all = np.hypot(q11*dx + q12*dy, q12*dx + q22*dy)
    finite = np.isfinite(rho2_all) & np.isfinite(vt_all) & np.isfinite(Qr_all) & (rho2_all > 0) & (Qr_all > 0)
    if not np.any(finite):
        return failed("no_finite_samples")
    if Omega0 is None:
        rotation_sign = np.sign(np.median(vt_all[finite]))
    elif not np.isfinite(Omega0):
        return failed("invalid_initial_angular_velocity")
    else:
        rotation_sign = np.sign(Omega0)
    if rotation_sign == 0:
        return failed("undetermined_rotation_sign")
    selected = finite & (rotation_sign * vt_all >= 0)
    indices = np.flatnonzero(selected)
    rho2, Qr = rho2_all[selected], Qr_all[selected]
    rho = np.sqrt(rho2)
    vt = vt_all[selected] * (rho / Qr)
    details.update(rho=rho, vt=vt, observation_indices=indices,
                   core_mask=(np.asarray(core_mask, bool)[indices] if core_mask is not None else rho <= 30_000))
    if len(rho) < 3:
        return failed("insufficient_samples", "At least three retained observations are needed for fit uncertainty.")
    if np.ptp(rho) <= np.finfo(float).eps * max(1, np.max(rho)):
        return failed("unidentifiable_radius", "Observations must span distinct radii.")
    if Rc0 is None:
        Rc0 = max(rho[np.argmax(np.abs(vt))] * np.sqrt(2), 1e-6)
    if not np.isfinite(Rc0) or Rc0 <= 0:
        return failed("invalid_initial_radius")
    r_scale, v_scale = np.median(rho), np.max(np.abs(vt))
    if v_scale == 0:
        return failed("zero_velocity")
    r = rho / r_scale
    speed = np.abs(vt) / v_scale
    if Omega0 is None:
        denom = rho * np.exp(-rho2 / Rc0**2)
        ok = denom > 0
        Omega0 = rotation_sign * np.median(np.abs(vt[ok]) / denom[ok]) if np.any(ok) else np.nan
    if not np.isfinite(Omega0) or Omega0 == 0:
        return failed("invalid_initial_angular_velocity")

    def profile(r, amplitude, radius):
        return amplitude * r * np.exp(-(r / radius)**2)

    def jacobian(r, amplitude, radius):
        e = np.exp(-(r / radius)**2)
        return np.column_stack([r*e, 2*amplitude*r**3*e/radius**3])

    upper = np.inf if Rc_max is None else Rc_max/r_scale
    radius0 = Rc0/r_scale
    if np.isfinite(upper):
        radius0 = min(radius0, upper * (1-1e-6))
    try:
        popt, pcov_scaled = curve_fit(
            profile, r, speed, p0=[abs(Omega0)*r_scale/v_scale, radius0],
            jac=jacobian, x_scale="jac",
            bounds=([0, 1e-8/r_scale], [np.inf, upper]), maxfev=maxfev,
            ftol=1e-10, xtol=1e-10, gtol=1e-10)
    except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return failed("optimizer_failed", str(exc))
    if not np.isfinite(popt).all() or np.any(popt <= 0):
        return failed("nonfinite_or_zero_solution")
    if Rc_max is not None and popt[1] >= upper * (1-1e-6):
        return failed("radius_bound_reached", "Optimum reaches the caller-supplied Rc_max.")
    if np.linalg.matrix_rank(jacobian(r, *popt)) < 2 or not np.isfinite(pcov_scaled).all():
        return failed("unidentifiable_radius", "Fitted radius/covariance is numerically unresolved.")
    linear_amplitude = np.dot(r, speed) / np.dot(r, r)
    linear_ss = np.sum((speed-linear_amplitude*r)**2)
    fitted_ss = np.sum((speed-profile(r, *popt))**2)
    if linear_ss <= fitted_ss + 1e-12*np.dot(speed, speed):
        return failed("unidentifiable_radius", "No Gaussian curvature is resolved relative to the infinite-radius linear limit.")
    transform = np.diag([rotation_sign*v_scale/r_scale, r_scale])
    pcov = transform @ pcov_scaled @ transform.T
    Omega_opt, Rc_opt = rotation_sign*popt[0]*v_scale/r_scale, popt[1]*r_scale
    psi0_opt = -0.5 * Omega_opt * Rc_opt**2
    vt_fit = Omega_opt * rho * np.exp(-rho2/Rc_opt**2)
    ss_res = np.sum((vt-vt_fit)**2)
    ss_tot = np.sum((vt-vt.mean())**2)
    R2 = 1-ss_res/ss_tot if ss_tot > 0 else np.nan
    details.update(status="optimized", Omega=Omega_opt, Rc=Rc_opt, psi0=psi0_opt,
                   R2=R2, pcov=pcov, sigma2=ss_res/(len(vt)-2),
                   Rc_standard_error=float(np.sqrt(max(pcov[1, 1], 0))))
    if fit_details is not None:
        fit_details.update(details)
    if plot:
        plot_outer_core_fit(details, ax=ax, km_flag=km_flag,
                            rho_plot_max=rho_plot_max, n_curve=n_curve,
                            ci_flag=ci_flag, pred_flag=pred_flag)
    return (Rc_opt, psi0_opt, Omega_opt, R2) if r2_flag else (Rc_opt, psi0_opt, Omega_opt)


def plot_outer_core_fit(fit, ax=None, km_flag=True, rho_plot_max=None,
                        n_curve=400, ci_flag=False, pred_flag=True):
    """Display a stored SI outer fit without running an optimizer.

    `fit` is populated by out_core_param_fit(fit_details=...). Its core_mask
    records membership in the actual inner-core fit. rho_plot_max is in metres;
    km_flag converts every horizontal plot coordinate, never physical inputs.
    """
    if ax is None:
        _, ax = plt.subplots()
    if fit.get("status", "optimized") != "optimized":
        ax.text(.5, .5, "Outer fit unavailable: " + fit.get("status", "unknown"),
                ha="center", va="center", transform=ax.transAxes, wrap=True)
        return ax
    rho, vt = fit["rho"], fit["vt"]
    Omega, Rc = fit["Omega"], fit["Rc"]
    core = fit["core_mask"]
    scale = 1e3 if km_flag else 1.0
    r = np.linspace(0, np.max(rho) if rho_plot_max is None else rho_plot_max, n_curve)
    signed_curve = Omega * r * np.exp(-r**2 / Rc**2)
    curve = np.abs(signed_curve)
    ax.scatter(rho[core] / scale, np.abs(vt[core]), s=10, color="m", label="Inner-core \nobserved")
    ax.scatter(rho[~core] / scale, np.abs(vt[~core]), s=10, color="g", label="Outer-core \nobserved")
    ax.plot(r / scale, curve, lw=2, color="b", label="")
    ax.axvline(Rc / np.sqrt(2) / scale, ls="--", color="r", lw=2)
    if fit["pcov"] is not None:
        exp_term = np.exp(-r**2 / Rc**2)
        J = np.column_stack([r * exp_term, 2 * Omega * r**3 * exp_term / Rc**3])
        var = np.einsum("ij,jk,ik->i", J, fit["pcov"], J)
        for enabled, extra, color, label in [(ci_flag, 0, "orange", "95% CI"),
                                             (pred_flag, fit["sigma2"], "b", "")]:
            if enabled:
                delta = 1.96 * np.sqrt(np.maximum(var + extra, 0))
                lo, hi = signed_curve - delta, signed_curve + delta
                # Transform the signed interval to the displayed magnitude.
                lower = np.where((lo <= 0) & (hi >= 0), 0, np.minimum(np.abs(lo), np.abs(hi)))
                upper = np.maximum(np.abs(lo), np.abs(hi))
                ax.fill_between(r / scale, lower, upper, color=color, alpha=.15, label=label)
    ax.set_xlabel(r"$\rho$ (km)" if km_flag else r"$\rho$ (m)")
    ax.set_ylabel(r"$|v_t^\star|$ (m s$^{-1}$)")
    ax.set_title(f"Omega={Omega:.3g}, Rc={Rc:.3g}, psi0={fit['psi0']:.3g}, R²={fit['R2']:.2f}")
    ax.set_ylim(0, None)
    ax.set_xlim(0, None)
    ax.legend(loc="upper left")
    return ax


# Helper functions

def doppio_pipeliner(nxc, nyc, ut, vt, X_new, Y_new, r=30000.0):
    '''
    Return orthogonal transects, centered at (nxc, nyc) and readius r, from gridded velocity data
    '''
    nan = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    x = np.asarray(X_new[:, 0], float)
    y = np.asarray(Y_new[0, :], float)

    dx = np.nanmedian(np.abs(np.diff(x)))
    dy = np.nanmedian(np.abs(np.diff(y)))
    cell = np.nanmax([dx, dy])
    if not np.isfinite(cell) or cell == 0:
        return nan

    margin = int(np.ceil(r / cell))

    ic = int(np.clip(np.searchsorted(x, nxc), 1, x.size-1))
    ic -= (nxc - x[ic-1] < x[ic] - nxc)
    jc = int(np.clip(np.searchsorted(y, nyc), 1, y.size-1))
    jc -= (nyc - y[jc-1] < y[jc] - nyc)

    if ic < margin or ic >= x.size - margin or jc < margin or jc >= y.size - margin:
        return nan

    # x-transect (y = y[jc])
    i0 = np.searchsorted(x, nxc - r, side="left")
    i1 = np.searchsorted(x, nxc + r, side="right")
    x1 = x[i0:i1]
    y1 = np.full(x1.size, y[jc])
    u1 = ut[i0:i1, jc]
    v1 = vt[i0:i1, jc]

    # y-transect (x = x[ic])
    j0 = np.searchsorted(y, nyc - r, side="left")
    j1 = np.searchsorted(y, nyc + r, side="right")
    y2 = y[j0:j1]
    x2 = np.full(y2.size, x[ic])
    u2 = ut[ic, j0:j1]
    v2 = vt[ic, j0:j1]

    return x1, y1, u1, v1, x2, y2, u2, v2


def latte_source_selector(ds_sadcp, ds_sat, source="multi", z_target=37.0,
                          xc_pre=0.0, yc_pre=0.0, q11=1.0, q12=0.0, q22=1.0,
                          rho_core=35_000.0, rho_outer = 100_000.0, plot=False,
                          outer_fit_options=None):
    '''
    Return ESP paramters given source of velocity data
    '''

    nanQ = np.full((2, 2), np.nan)

    def empty_return(status="no_selected_observations"):
        df_row = pd.DataFrame([{
            "source": source, "xc": np.nan, "yc": np.nan, "w": np.nan, "Q": nanQ,
            "Omega": np.nan, "Rc": np.nan, "psi0": np.nan,
            "time": time, "alpha": np.nan
        }])
        df_xyuv = pd.DataFrame(columns=["source", "platform", "core", "xi", "yi", "ui", "vi"])
        df_fit = pd.DataFrame(columns=["rho2", "Qr", "vt", "platform"])
        df_row["fit_status"] = status
        df_fit.attrs["outer_fit"] = dict(status=status, rho=np.array([]))
        df_fit.attrs["settings"] = dict(z=z0, xc_pre=xc_pre, yc_pre=yc_pre,
                                        rho_core=rho_core, rho_outer=rho_outer)
        df_fit.attrs["selection_counts"] = []
        return df_row, df_xyuv, df_fit

    t_mean = ds_sadcp.time.mean().values
    time = t_mean

    # ADCP-only fits remain independent of satellite temporal coverage.
    sat_interp = (interpolate_satellite_at_time(ds_sat, t_mean)
                  if source in ("multi", "sat") else ds_sat.interp(time=t_mean))
    X, Y = ds_sat.x.values, ds_sat.y.values
    xg, yg = X.ravel(), Y.ravel()
    ut, vt = sat_interp.u.values.ravel(), sat_interp.v.values.ravel()

    sat_mask = np.isfinite(xg) & np.isfinite(yg) & np.isfinite(ut) & np.isfinite(vt)
    xg, yg, ut, vt = xg[sat_mask], yg[sat_mask], ut[sat_mask], vt[sat_mask]

    z0 = float(ds_sadcp.z.sel(z=z_target, method="nearest").values)
    xd, yd = ds_sadcp.x.values, ds_sadcp.y.values
    ud = ds_sadcp.u.sel(z=z0).values
    vd = ds_sadcp.v.sel(z=z0).values

    sadcp_mask = np.isfinite(xd) & np.isfinite(yd) & np.isfinite(ud) & np.isfinite(vd)
    xd, yd, ud, vd = xd[sadcp_mask], yd[sadcp_mask], ud[sadcp_mask], vd[sadcp_mask]

    if source == "multi":
        if not len(xd) or not len(xg):
            raise ValueError("A multi-platform fit requires finite ADCP and satellite observations.")
        xi = np.concatenate([xd, xg])
        yi = np.concatenate([yd, yg])
        ui = np.concatenate([ud, ut])
        vi = np.concatenate([vd, vt])
        platform = np.concatenate([np.full(len(xd), "sadcp"), np.full(len(xg), "sat")])
    elif source == "sadcp":
        xi, yi, ui, vi = xd, yd, ud, vd
        platform = np.full(len(xd), "sadcp")
    elif source == "sat":
        xi, yi, ui, vi = xg, yg, ut, vt
        platform = np.full(len(xg), "sat")
    else:
        return empty_return()

    dx, dy = xi - xc_pre, yi - yc_pre
    rho2 = q11 * dx**2 + 2 * q12 * dx * dy + q22 * dy**2
    core1 = rho2 <= rho_core**2

    def require_platforms(mask, stage):
        if source == "multi" and set(platform[mask]) != {"sadcp", "sat"}:
            raise ValueError(f"Multi-platform {stage} selection must include both ADCP and satellite observations.")

    require_platforms(core1, "initial inner-core")
    if not np.any(core1):
        return empty_return()

    inner_details = {}
    xc, yc, w, Q, Omega, _ = refine_latte(
        xi, yi, ui, vi, center=(xc_pre, yc_pre),
        Q_pre=np.array([[q11, q12], [q12, q22]]), rho_core=rho_core,
        fit_details=inner_details)
    if inner_details["status"] != "optimized":
        return empty_return(inner_details["status"])
    core2 = inner_details["refined_mask"]
    require_platforms(core2, "refined inner-core")
    alpha = axis_ratio_from_Q(Q)

    df_inner = pd.DataFrame({
        "source": source,
        "platform": platform[core2],
        "core": "inner",
        "xi": xi[core2],
        "yi": yi[core2],
        "ui": ui[core2],
        "vi": vi[core2]
    })

    q11, q12, q22 = Q[0, 0], Q[0, 1], Q[1, 1]
    dx, dy = xi - xc, yi - yc
    rho2 = q11 * dx**2 + 2 * q12 * dx * dy + q22 * dy**2
    outer = rho2 <= rho_outer**2
    outer_only = outer & ~core2

    df_outer = pd.DataFrame({
        "source": source,
        "platform": platform[outer_only],
        "core": "outer",
        "xi": xi[outer_only],
        "yi": yi[outer_only],
        "ui": ui[outer_only],
        "vi": vi[outer_only]
    })

    xi_o, yi_o = xi[outer], yi[outer]
    ui_o, vi_o = ui[outer], vi[outer]
    dx_o, dy_o = dx[outer], dy[outer]
    rho2_o = rho2[outer]

    vt_o = tangential_velocity(xi_o, yi_o, ui_o, vi_o, xc, yc, Q)
    Qr_o = np.sqrt((q11 * dx_o + q12 * dy_o)**2 + (q12 * dx_o + q22 * dy_o)**2)

    sign_mask = (vt_o <= 0) if (Omega < 0) else (vt_o >= 0)
    rho2_f = rho2_o[sign_mask]
    Qr_f = Qr_o[sign_mask]
    vt_f = vt_o[sign_mask]
    platform_f = platform[outer][sign_mask]
    finite_fit = (np.isfinite(rho2_f) & np.isfinite(Qr_f) & np.isfinite(vt_f)
                  & (rho2_f >= 0) & (Qr_f != 0))
    if source == "multi" and set(platform_f[finite_fit]) != {"sadcp", "sat"}:
        raise ValueError("Multi-platform outer fit must include both ADCP and satellite observations.")

    if len(rho2_f) == 0:
        return empty_return()

    outer_fit = {}
    Rc, psi0, Omega_opt = out_core_param_fit(
        xi_o, yi_o, ui_o, vi_o, xc, yc, Q, Omega0=Omega, plot=plot,
        pred_flag=True, fit_details=outer_fit, core_mask=core2[outer],
        **(outer_fit_options or {}))
    
    w_core, Omega_core = w, Omega
    w = Omega_opt * (q11 + q22)

    df_row = pd.DataFrame([{
        "source": source,
        "xc": xc,
        "yc": yc,
        "w": w,
        "Q": Q,
        "Omega": Omega_opt,
        "Omega_core": Omega_core,
        "w_core": w_core,
        "fit_status": outer_fit["status"],
        "Rc": Rc,
        "psi0": psi0,
        "time": time,
        "alpha": alpha
    }])

    df_fit = pd.DataFrame({
        "rho2": rho2_f,
        "Qr": Qr_f,
        "vt": vt_f,
        "platform": platform_f
    })

    df_fit.attrs["outer_fit"] = outer_fit
    df_fit.attrs["settings"] = dict(z=z0, xc_pre=xc_pre, yc_pre=yc_pre,
                                    rho_core=rho_core, rho_outer=rho_outer)
    stage_counts = []
    for stage, indices in [("available", np.arange(len(xi))),
                           ("initial_core", np.flatnonzero(core1)),
                           ("refined_core", np.flatnonzero(core2)),
                           ("outer_candidates", np.flatnonzero(outer)),
                           ("outer_fit", np.flatnonzero(outer)[outer_fit.get("observation_indices", np.array([], dtype=int))])]:
        for name in np.unique(platform):
            finite = np.isfinite(xi[indices]) & np.isfinite(yi[indices]) & np.isfinite(ui[indices]) & np.isfinite(vi[indices])
            stage_counts.append(dict(source=source, stage=stage, platform=name,
                                     finite_n=int(np.count_nonzero((platform[indices] == name) & finite))))
    df_fit.attrs["selection_counts"] = stage_counts
    return df_row, pd.concat([df_inner, df_outer], ignore_index=True), df_fit

def axis_ratio_from_Q(Q):
    if not is_elliptical_Q(Q):
        return np.nan
    lam = np.linalg.eigvalsh(Q)
    return np.sqrt(lam.max() / lam.min())

def tangential_velocity(xp, yp, up, vp, xc, yc, Q, det1=False):
    '''
    Finds the tangential velocity given a flow with center (xc,yc) and deformation Q
    '''
    Q = np.asarray(Q, float)
    if Q.shape == (3,):
        q11, q12, q22 = Q
        Q = np.array([[q11, q12], [q12, q22]], float)
    if not is_elliptical_Q(Q, require_det_one=False):
        raise ValueError("Tangential velocity requires a symmetric positive-definite Q.")
    if det1:
        Q = Q / np.sqrt(np.linalg.det(Q))

    xp, yp, up, vp = (np.asarray(a, float) for a in (xp, yp, up, vp))
    r   = np.stack((xp - xc, yp - yc), axis=-1)
    g   = 2.0 * (r @ Q.T)                    # ∇F
    J   = np.array([[0., -1.], [1., 0.]])    # +90° rot
    tau = g @ J.T                             # tangent
    nrm = np.linalg.norm(tau, axis=-1, keepdims=True)
    t_hat = np.divide(tau, nrm, out=np.zeros_like(tau), where=nrm > 0)

    vel = np.stack((up, vp), axis=-1)
    vt  = np.sum(vel * t_hat, axis=-1)
    vt  = np.where(nrm.squeeze() > 0, vt, np.nan)
    return vt

def model_uv_at_xy(xi, yi, xc, yc, Q, Omega, Rc):
    """ESP velocities; unavailable/invalid fit parameters give NaN predictions."""
    xi, yi = np.broadcast_arrays(np.asarray(xi, float), np.asarray(yi, float))
    Q = np.asarray(Q, float)
    if not is_elliptical_Q(Q) or not np.isfinite([xc, yc, Omega, Rc]).all() or Rc <= 0:
        return np.full(xi.shape, np.nan), np.full(yi.shape, np.nan)
    dx = xi - xc
    dy = yi - yc

    q11, q12, q22 = Q[0,0], Q[0,1], Q[1,1]

    rho2 = q11*dx*dx + 2*q12*dx*dy + q22*dy*dy
    fac = Omega * np.exp(-rho2 / (Rc*Rc))

    uhat = -fac * (q12*dx + q22*dy)
    vhat =  fac * (q11*dx + q12*dy)

    return uhat, vhat
    
def vector_R2(u, v, uhat, vhat):
    '''
    Coefficient of determination between ESP flow (uhat, vhat) and original flow (u, v)
    '''
    m = np.isfinite(u) & np.isfinite(v) & np.isfinite(uhat) & np.isfinite(vhat)
    if not np.any(m):
        return np.nan
    u, v, uhat, vhat = u[m], v[m], uhat[m], vhat[m]
    err2 = (uhat - u)**2 + (vhat - v)**2
    u0, v0 = np.mean(u), np.mean(v)
    tot2 = (u - u0)**2 + (v - v0)**2
    return 1 - np.sum(err2)/np.sum(tot2) if np.sum(tot2) > 0 else np.nan
