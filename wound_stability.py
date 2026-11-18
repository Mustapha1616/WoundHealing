"""
Spectral stability

Model: n_t = alpha*n_xx - (n*V[n])_x + n*(1-n), g(n)=n*(lam-n).
The equilibrium is n=1. Perturbations have zero endpoint/exterior values.
L is the HALF-length: the physical interval is (-L,L).

This module implements the manuscript's sine-Galerkin Algorithms 1 and 2.
It accepts the same kernel_fun(z, radius) callables as the user's RK4 code.
It does not substitute whole-line dispersion at discrete wavenumbers, and
does not diagonalize the FC discretization in wound_healing.py.
"""
from functools import lru_cache
from pathlib import Path
import argparse
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.special import roots_legendre


def webb_adhesion_kernel(z, radius):
    """Same fixed normalization and cutoff as the user's kernel."""
    z = np.asarray(z, dtype=float)
    return np.where(np.abs(z) <= radius,
                    0.1*np.arctan(0.2*z)/np.arctan(2.0), 0.0)


def polynomial_adhesion_kernel(z, radius):
    z = np.asarray(z, dtype=float)
    if radius == 0:
        return np.zeros_like(z)
    eta = z/radius
    return np.where(np.abs(z) <= radius,
                    3*z/radius**2*(1-eta**2)**2, 0.0)


@lru_cache(maxsize=32)
def _gauss(order):
    return roots_legendre(int(order))


def _parameters(L, rho, lam, alpha):
    if not np.all(np.isfinite([L, rho, lam, alpha])):
        raise ValueError('Parameters must be finite.')
    if L <= 0 or rho < 0 or alpha <= 0:
        raise ValueError('Require L>0, rho>=0, and alpha>0.')


def _kernel_values(kernel_fun, r, rho):
    """The symmetric formulation requires a real odd kernel."""

    positive = np.asarray(kernel_fun(r, rho))
    negative = np.asarray(kernel_fun(-r, rho))

    if np.iscomplexobj(positive) or np.iscomplexobj(negative):
        raise ValueError('The kernel must be real.')
    
    positive = np.broadcast_to(positive, np.shape(r)).astype(float)
    negative = np.broadcast_to(negative, np.shape(r)).astype(float)

    if not np.all(np.isfinite(positive)) or not np.all(np.isfinite(negative)):
        raise ValueError('The kernel returned nonfinite values.')
    if not np.allclose(negative, -positive, atol=1e-12, rtol=1e-10):
        raise ValueError('These algorithms require an odd kernel.')
    
    return positive


def galerkin_matrix(L, rho, lam, *, alpha=1.0, m=192, q=None,
                    kernel_fun=webb_adhesion_kernel):
    """
    Full weak Galerkin matrix, including sensing endpoint effects.

    e_j=L**(-1/2)*sin(k_j*(x+L)), k_j=j*pi/(2L).
    A_ij=-(alpha*k_j**2+1)*delta_ij+(lam-2)*<e_i',V e_j>.
    """
    _parameters(L, rho, lam, alpha)
    if int(m) != m or m < 2:
        raise ValueError('m must be an integer >=2.')
    
    m = int(m)
    k = np.arange(1, m+1)*np.pi/(2*L)

    if rho == 0 or lam == 2:
        return np.diag(-alpha*k*k-1)
    
    a = min(rho, 2*L)
    q = max(96, int(np.ceil(0.8*k[-1]*a))+16) if q is None else int(q)

    if q < 2:
        raise ValueError('Quadrature order q must be >=2.')
    
    z, w = _gauss(q)
    r = (z+1)*a/2
    wh = (w*a/2)*_kernel_values(kernel_fun, r, rho)
    kr = k[:, None]*r
    c = np.cos(kr) @ wh
    denominator = k[:, None]**2-k[None, :]**2
    np.fill_diagonal(denominator, 1.0)
    K = (2*k[:, None]*k[None, :]*(c[:, None]-c[None, :])
         / (L*denominator))
    diagonal = (k/L)*(np.sin(kr) @ ((2*L-r)*wh))
    np.fill_diagonal(K, diagonal)
    parity = np.arange(m) % 2
    K[parity[:, None] != parity[None, :]] = 0.0
    A = (lam-2)*K
    A[np.diag_indices(m)] -= alpha*k*k+1

    return A


def _leading_pair(A):
    scale = max(1.0, float(np.linalg.norm(A, ord=np.inf)))
    symmetry = float(np.max(np.abs(A-A.T)))

    if symmetry > 1e-11*scale:
        raise RuntimeError('The Galerkin matrix failed the symmetry check.')
    
    best = None

    for offset, parity in ((0, 'even'), (1, 'odd')):
        ids = np.arange(offset, len(A), 2)
        block = A[np.ix_(ids, ids)]
        values, vectors = eigh(block, subset_by_index=[len(ids)-1, len(ids)-1])
        if best is None or values[0] > best['s_L']:
            coefficients = np.zeros(len(A))
            coefficients[ids] = vectors[:, 0]
            best = {'s_L': float(values[0]), 'coefficients': coefficients,
                    'parity': parity}
    residual = float(np.linalg.norm(A @ best['coefficients']
                                   - best['s_L']*best['coefficients']))
    if residual/scale > 1e-10:
        raise RuntimeError('The eigenpair failed the residual check.')
    best.update(symmetry_error=symmetry, residual=residual,
                relative_residual=residual/scale, m=len(A))
    
    return best


def bounded_spectrum(L, rho, lam, *, alpha=1.0, m=192, q=None,
                     tol_q=1e-9, max_q=32768,
                     kernel_fun=webb_adhesion_kernel):
    """Leading eigenpair at fixed m, with independent quadrature refinement.

    This resolves quadrature but does NOT claim convergence in basis size m.
    Use compute_s_L for the complete Algorithm 1 refinement.
    """
    _parameters(L, rho, lam, alpha)
    if tol_q <= 0:
        raise ValueError('tol_q must be positive.')
    a = min(rho, 2*L)
    q = max(96, int(np.ceil(0.8*m*np.pi*a/(2*L)))+16) if q is None else int(q)
    old = None
    while q <= max_q:
        result = _leading_pair(galerkin_matrix(L, rho, lam, alpha=alpha,
                              m=m, q=q, kernel_fun=kernel_fun))
        change = abs(result['s_L']-old) if old is not None else np.inf
        result.update(L=float(L), rho=float(rho), lam=float(lam),
                      alpha=float(alpha), q=q, quadrature_change=change)
        if change <= tol_q or rho == 0 or lam == 2:
            result['quadrature_change'] = 0.0 if old is None else change
            return result
        old = result['s_L']
        q *= 2
    raise RuntimeError('Quadrature did not converge; increase max_q or relax tol_q.')


def compute_s_L(L, rho, lam, *, alpha=1.0, m0=64, max_m=1024,
                tol_s=1e-6, tol_q=1e-9,
                kernel_fun=webb_adhesion_kernel):
    """
    Algorithm 1: independently refine quadrature and Galerkin dimension.

    Returns a dictionary with s_L, unit L2 coefficients, parity, m, q,
    residual, and refinement history. Raises if refinement does not converge.
    Successive differences are numerical checks, not certified error bounds.
    """
    if int(m0) != m0 or m0 < 2 or max_m < 2*m0 or tol_s <= 0:
        raise ValueError('Require integer m0>=2, max_m>=2*m0, and tol_s>0.')
    history, previous = [], None
    m = int(m0)

    while m <= max_m:
        result = bounded_spectrum(L, rho, lam, alpha=alpha, m=m,
                                  tol_q=tol_q, kernel_fun=kernel_fun)
        change = abs(result['s_L']-previous) if previous is not None else np.inf
        history.append({'m': m, 'q': result['q'], 's_L': result['s_L'],
                        'quadrature_change': result['quadrature_change'],
                        'spatial_change': change})
        if change <= tol_s:
            result.update(converged=True, spatial_change=change, history=history)
            return result
        
        previous = result['s_L']
        m *= 2

    raise RuntimeError(f'Basis refinement did not converge through m={m//2}; '
                       f'last change={change:.3g}. Increase max_m.')


def eigenfunction(x, result):
    """
    Evaluate a returned bounded eigenfunction, with zero exterior values.
    """
    x = np.asarray(x, dtype=float)
    flat = x.ravel()
    L, coefficients = result['L'], result['coefficients']
    out = np.zeros_like(flat)
    inside = np.abs(flat) < L
    k = np.arange(1, len(coefficients)+1)*np.pi/(2*L)
    out[inside] = coefficients @ np.sin(k[:, None]*(flat[inside]+L))/np.sqrt(L)
    return out.reshape(x.shape)


def _whole_once(rho, lam, alpha, kernel_fun, q, kmax, points_per_period):
    z, w = _gauss(q)
    r = (z+1)*rho/2
    wh = w*rho/2*_kernel_values(kernel_fun, r, rho)

    def sigma(k):
        k = np.asarray(k, dtype=float)
        shape = k.shape
        flat = np.atleast_1d(k).ravel()
        out = np.empty_like(flat)
        for start in range(0, len(flat), 512):
            block = flat[start:start+512]
            S = 2*np.sin(block[:, None]*r) @ wh
            out[start:start+512] = -alpha*block**2-1+(lam-2)*block*S
        return out.reshape(shape) if shape else float(out[0])
    
    count = max(257, int(np.ceil(kmax*rho*points_per_period/(2*np.pi)))+1)
    grid = np.linspace(0.0, kmax, count)
    values = sigma(grid)
    peaks = np.flatnonzero((values[1:-1] >= values[:-2]) &
                          (values[1:-1] > values[2:]))+1
    candidates = [(float(values[0]), 0.0), (float(values[-1]), float(kmax))]

    for index in peaks:
        opt = minimize_scalar(lambda k: -sigma(k),
                              bounds=(grid[index-1], grid[index+1]),
                              method='bounded', options={'xatol': 1e-11})
        if not opt.success:
            raise RuntimeError('A local dispersion maximization failed.')
        candidates.append((-float(opt.fun), float(opt.x)))

    F, kstar = max(candidates)
    S = float(2*np.sin(kstar*r) @ wh)
    dS = float(2*(r*np.cos(kstar*r)) @ wh)

    return {'F': F, 'k_star': kstar,
            'wavelength': 2*np.pi/kstar if kstar > 0 else np.inf,
            'stationarity_residual': -2*alpha*kstar+(lam-2)*(S+kstar*dS),
            'q': q, 'kmax': kmax, 'grid_points': count}


def whole_line_growth(rho, lam, *, alpha=1.0,
                      kernel_fun=webb_adhesion_kernel, tol=1e-8,
                      points_per_period=24, max_refinements=4):
    """Return F=max_{k>=0} sigma(k) and its resolved maximizing wavenumber.

    A kernel L1 estimate chooses kmax>|lam-2|*H_rho/alpha; the quadratic
    bound then excludes a larger maximum beyond kmax. All resolved local
    maxima and both endpoints are considered. Quadrature and the wavenumber
    grid are refined separately. This is a numerical search, not a proof
    that arbitrary highly oscillatory custom kernels have no missed peak.
    """
    _parameters(1.0, rho, lam, alpha)
    if tol <= 0 or points_per_period < 4:
        raise ValueError('Require tol>0 and points_per_period>=4.')
    if rho == 0 or lam == 2:
        return {'F': -1.0, 'k_star': 0.0, 'wavelength': np.inf,
                'stationarity_residual': 0.0, 'quadrature_change': 0.0,
                'wavenumber_change': 0.0, 'q': 0, 'kmax': 0.0,
                'grid_points': 1, 'converged': True}
    integral, err = quad(lambda r: abs(float(kernel_fun(np.asarray(r), rho))),
                         0.0, rho, epsabs=1e-11, epsrel=1e-11)
    
    H = 2*(integral+10*err)
    kmax = max(1.0, abs(lam-2)*H/alpha+1.0)
    q = max(96, int(np.ceil(0.8*kmax*rho))+16)
    density = points_per_period

    for _ in range(max_refinements):
        base = _whole_once(rho, lam, alpha, kernel_fun, q, kmax, density)
        qfine = _whole_once(rho, lam, alpha, kernel_fun, 2*q, kmax, density)
        kfine = _whole_once(rho, lam, alpha, kernel_fun, 2*q, kmax, 2*density)
        qchange = abs(qfine['F']-base['F'])
        kchange = abs(kfine['F']-qfine['F'])
        if max(qchange, kchange) <= tol:
            kfine.update(quadrature_change=qchange, wavenumber_change=kchange,
                         converged=True)
            return kfine
        q, density = 2*q, 2*density

    raise RuntimeError('Whole-line maximum did not converge under refinement.')


def _scan_crossing(growth, interval, step, zero_tol=1e-10):
    """Scan from a strictly stable point; conservatively flag neutral samples."""
    lo, hi = map(float, interval)
    if not np.isfinite([lo, hi, step]).all() or not 0 <= lo < hi or step <= 0:
        raise ValueError('Require a finite 0<=lower<upper interval and positive scan_step.')
    
    grid = np.linspace(lo, hi, int(np.ceil((hi-lo)/step))+1)
    previous_x, previous_g = grid[0], float(growth(grid[0]))

    if previous_g >= -zero_tol:
        raise ValueError('The search interval must start in a strictly stable regime.')
    neutral = []
    for x in grid[1:]:
        gx = float(growth(x))
        if not np.isfinite(gx):
            raise RuntimeError('Nonfinite growth rate during the threshold scan.')
        if abs(gx) <= zero_tol:
            neutral.append(float(x))
            if len(neutral) >= 2:
                return {'status': 'unresolved_neutral_interval', 'value': None,
                        'interval': (neutral[0], neutral[-1])}
            continue
        if gx > zero_tol:
            return {'status': 'crossing', 'bracket': (float(previous_x), float(x))}
        if neutral:
            return {'status': 'unresolved_neutral_contact', 'value': None,
                    'interval': (float(previous_x), float(x))}
        previous_x, previous_g = x, gx

    if neutral:
        return {'status': 'unresolved_neutral_endpoint', 'value': None,
                'interval': (float(previous_x), float(grid[-1]))}
    return {'status': 'no_crossing_in_search_interval', 'value': None,
            'interval': (lo, hi)}


def _refined_scan(growth, interval, step):
    coarse = _scan_crossing(growth, interval, step)
    for _ in range(4):
        step /= 2
        fine = _scan_crossing(growth, interval, step)
        if coarse['status'] == fine['status']:
            if fine['status'] != 'crossing':
                fine['scan_step'] = step
                return fine
            a, b = coarse['bracket']; c, d = fine['bracket']
            if max(a, c) <= min(b, d):
                fine['scan_step'] = step
                return fine
        coarse = fine
    return {'status': 'scan_not_resolved', 'value': None, 'scan_step': step}


def _bisect(growth, bracket, tolerance):
    a, b = bracket
    if not float(growth(a)) < 0 < float(growth(b)):
        raise RuntimeError('Bisection requires a negative-to-positive bracket.')
    iterations = 0
    while b-a > tolerance:
        middle = (a+b)/2
        gm = float(growth(middle))
        if gm == 0:
            return middle, (middle, middle), iterations+1
        if gm < 0:
            a = middle
        else:
            b = middle
        iterations += 1
    return (a+b)/2, (a, b), iterations


def critical_parameter(kind, lam, *, L=None, rho=None, alpha=1.0,
                       search_interval=None, scan_step=0.5, tol_theta=1e-3,
                       m0=96, max_m=1536, tol_q=1e-9,
                       kernel_fun=webb_adhesion_kernel):
    """Algorithm 2; kind is 'rho_whole', 'rho_bounded', or 'L'."""

    if kind not in ('rho_whole', 'rho_bounded', 'L'):
        raise ValueError("kind must be 'rho_whole', 'rho_bounded', or 'L'.")
    if tol_theta <= 0 or tol_q <= 0:
        raise ValueError('Tolerances must be positive.')
    if kind == 'rho_bounded' and (L is None or L <= 0):
        raise ValueError('Supply the fixed half-length L>0.')
    if kind == 'L' and (rho is None or rho < 0):
        raise ValueError('Supply the fixed sensing radius rho>=0.')
    if search_interval is None:
        search_interval = (1.0, 250.0) if kind == 'L' else (0.5, 30.0)
    if kind == 'L' and search_interval[0] <= 0:
        raise ValueError('The half-length search must start above zero.')
    if int(m0) != m0 or m0 < 2 or max_m < 2*m0:
        raise ValueError('Require integer m0>=2 and max_m>=2*m0.')
    
    history, previous_root, previous_status = [], None, None
    m = int(m0)

    while m <= max_m:
        @lru_cache(maxsize=None)
        def evaluation(theta):
            if kind == 'rho_whole':
                return whole_line_growth(theta, lam, alpha=alpha,
                    kernel_fun=kernel_fun, tol=tol_q)
            fixed_L = theta if kind == 'L' else L
            fixed_rho = rho if kind == 'L' else theta
            return bounded_spectrum(fixed_L, fixed_rho, lam, alpha=alpha,
                                     m=m, tol_q=tol_q, kernel_fun=kernel_fun)
        key = 'F' if kind == 'rho_whole' else 's_L'
        growth = lambda theta: evaluation(float(theta))[key]
        scan = _refined_scan(growth, search_interval, scan_step)
        if scan['status'] != 'crossing':
            history.append({'m': None if kind == 'rho_whole' else m,
                            'status': scan['status']})
            if kind == 'rho_whole' or previous_status == scan['status']:
                scan.update(kind=kind, lam=lam, fixed_L=L, fixed_rho=rho,
                            search_interval=search_interval, history=history)
                return scan
            previous_status = scan['status']
            m *= 2
            continue
        root, bracket, iterations = _bisect(growth, scan['bracket'], tol_theta/4)
        mode = evaluation(float(root))
        change = abs(root-previous_root) if previous_root is not None else np.inf
        history.append({'m': None if kind == 'rho_whole' else m, 'value': root,
                        'parameter_change': change, 'scan_step': scan['scan_step'],
                        'bracket_width': bracket[1]-bracket[0]})
        result = {'status': 'converged' if kind == 'rho_whole' or change <= tol_theta
                  else 'basis_not_converged', 'value': root, 'kind': kind,
                  'lam': lam, 'fixed_L': L, 'fixed_rho': rho, 'alpha': alpha,
                  'bracket': bracket, 'iterations': iterations,
                  'search_interval': search_interval, 'history': history,
                  'growth_residual': mode[key], 'mode': mode}
        if result['status'] == 'converged':
            return result
        previous_root, previous_status = root, 'crossing'
        m *= 2

    if 'result' in locals():
        return result
    
    return {'status': 'basis_not_converged', 'value': None, 'history': history}


def critical_rho_whole_line(lam, **kwargs):
    return critical_parameter('rho_whole', lam, **kwargs)


def critical_rho_bounded(L, lam, **kwargs):
    return critical_parameter('rho_bounded', lam, L=L, **kwargs)


def critical_half_length(rho, lam, **kwargs):
    return critical_parameter('L', lam, rho=rho, **kwargs)


def plot_figure1(*, kernel_fun=webb_adhesion_kernel, alpha=1.0,
                 lambda_values=(5.0, 6.0), L_values=(30.0, 100.0, 200.0),
                 rho_ranges=((12.0, 24.0), (8.0, 16.0)),
                 rho_for_Lcurves=((16.0, 18.0), (10.0, 11.0)),
                 L_range=(5.0, 200.0), length_ylim=(-0.25, 0.12),
                 n_points=81, m_curve=192,
                 threshold_tol=1e-3, threshold_m0=96, threshold_max_m=1536,
                 output_dir=None, show=True, verbose=True):
    """
    Top: F and s_L versus rho. Bottom: s_L versus the half-length L.
    """
    import matplotlib.pyplot as plt

    if not (len(lambda_values) == len(rho_ranges) == len(rho_for_Lcurves)):
        raise ValueError('Supply one rho range and one radius tuple per lambda.')
    if n_points < 3 or not len(lambda_values) or not len(L_values):
        raise ValueError('Use n_points>=3 and nonempty parameter sequences.')
    
    out = (Path.cwd() / 'solution_storage' if output_dir is None
           else Path(output_dir).expanduser())
    out.mkdir(parents=True, exist_ok=True)
    data = {'alpha': alpha, 'kernel': getattr(kernel_fun, '__name__', str(kernel_fun)),
            'curve_modes': m_curve, 'threshold_tolerance': threshold_tol,
            'note': 'Continuum Galerkin spectra; not the discrete FC/RK4 Jacobian.',
            'radius': {}, 'length': {}}
    colors = ['#2166ac', '#b35806', '#238b45', '#7b3294', '#c51b7d']
    style = {'font.family': 'serif', 'font.size': 10, 'mathtext.fontset': 'stix',
             'axes.spines.top': False, 'axes.spines.right': False,
             'pdf.fonttype': 42, 'ps.fonttype': 42, 'legend.frameon': False,
             'text.usetex': False}
    common = dict(alpha=alpha, kernel_fun=kernel_fun, tol_theta=threshold_tol,
                  m0=threshold_m0, max_m=threshold_max_m)
    
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, len(lambda_values),
            figsize=(4.4*len(lambda_values), 6.6), squeeze=False, layout='constrained')
        for column, lam in enumerate(lambda_values):
            if verbose:
                print(f'Figure 1: lambda={lam:g}, computing radius curves...', flush=True)
            rr = np.linspace(*rho_ranges[column], n_points)
            F = np.array([whole_line_growth(r, lam, alpha=alpha,
                          kernel_fun=kernel_fun)['F'] for r in rr])
            ax = axes[0, column]
            ax.plot(rr, F, color='.15', ls='--', label=r'Whole line: $F$')
            rho_hi = float(rr[-1])
            root = critical_rho_whole_line(lam, search_interval=(0.5, rho_hi), **common)
            radius_row = {'rho': rr, 'F': F, 'whole_line_threshold': root, 'bounded': {}}
            if root['status'] == 'converged':
                ax.plot(root['value'], 0, 'D', color='.15', ms=4)
            for index, L in enumerate(L_values):
                color = colors[index % len(colors)]
                ss = np.array([bounded_spectrum(L, r, lam, alpha=alpha,
                    m=m_curve, kernel_fun=kernel_fun)['s_L'] for r in rr])
                ax.plot(rr, ss, color=color, label=fr'$L={L:g}$')
                # A finite interval is searched; no universal rho cutoff is
                # assumed because the polynomial kernel changes with rho.
                threshold = critical_rho_bounded(L, lam,
                    search_interval=(0.5, max(60.0, rho_hi)), **common)
                radius_row['bounded'][f'{L:g}'] = {'s_L': ss, 'threshold': threshold}
                if threshold['status'] == 'converged':
                    if rr[0] <= threshold['value'] <= rr[-1]:
                        ax.plot(threshold['value'], 0, 'o', color=color, ms=4)
                elif threshold['status'] == 'basis_not_converged':
                    warnings.warn(f'Radius threshold not converged: L={L}, lambda={lam}')
                if verbose:
                    print(f'  L={L:g}: {threshold["status"]}, '
                          f'rho_c={threshold.get("value")}', flush=True)
            ax.axhline(0, color='.4', lw=.8)
            ax.set(xlabel=r'Sensing radius $\rho$', ylabel='Largest growth rate',
                   xlim=(rr[0], rr[-1]))
            ax.set_title(fr'({chr(97+column)}) $\lambda={lam:g}$: radius threshold', loc='left')
            ax.legend(fontsize=8.5)
            #ax.grid(alpha=.15, lw=.5)
            data['radius'][f'{lam:g}'] = radius_row

            if verbose:
                print(f'Figure 1: lambda={lam:g}, computing length curves...', flush=True)
            ll = np.linspace(*L_range, n_points)
            ax = axes[1, column]
            length_row = {}
            for index, rho in enumerate(rho_for_Lcurves[column]):
                color = colors[index % len(colors)]
                ss = np.array([bounded_spectrum(L, rho, lam, alpha=alpha,
                    m=m_curve, kernel_fun=kernel_fun)['s_L'] for L in ll])
                ax.plot(ll, ss, color=color, label=fr'$\rho={rho:g}$')
                threshold = critical_half_length(rho, lam,
                    search_interval=(1.0, max(250.0, L_range[1])), scan_step=5.0, **common)
                length_row[f'{rho:g}'] = {'L': ll, 's_L': ss, 'threshold': threshold}
                if threshold['status'] == 'converged':
                    Lc = threshold['value']
                    if ll[0] <= Lc <= ll[-1]:
                        ax.plot(Lc, 0, 'o', color=color, ms=4)
                        ax.annotate(fr'$L_c\approx {Lc:.2f}$', (Lc, 0),
                            xytext=(-8 if Lc > 120 else 6, 13+27*index),
                            textcoords='offset points', color=color, fontsize=8.5,
                            ha='right' if Lc > 120 else 'left')
                elif threshold['status'] == 'basis_not_converged':
                    warnings.warn(f'Length threshold not converged: rho={rho}, lambda={lam}')
                if verbose:
                    print(f'  rho={rho:g}: {threshold["status"]}, '
                          f'L_c={threshold.get("value")}', flush=True)
            ax.axhline(0, color='.4', lw=.8)
            ax.set(xlabel=r'Domain half-length $L$', ylabel=r'$s_L(\rho;\lambda)$',
                   xlim=(ll[0], ll[-1]))
            if length_ylim is not None:
                ax.set_ylim(*length_ylim)
            ax.set_title(fr'({chr(97+len(lambda_values)+column)}) $\lambda={lam:g}$: length threshold', loc='left')
            ax.legend(fontsize=9, loc='lower right')
            #ax.grid(alpha=.15, lw=.5)
            data['length'][f'{lam:g}'] = length_row
            
        pdf_path = out / 'linear_thresholds.pdf'
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        if verbose:
            print(f'Saved {pdf_path.resolve()}', flush=True)
        if show:
            plt.show()
    return fig, data


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Compute linear stability thresholds and generate Figure 1.')
    parser.add_argument('--figure1', action='store_true',
                        help='Generate Figure 1 (the default; this flag is optional).')
    parser.add_argument('--output', help='PDF output directory (default: ./solution_storage).')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args(argv)
    return plot_figure1(output_dir=args.output, show=not args.no_show)


if __name__ == '__main__':
    main()
