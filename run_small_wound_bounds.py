"""Small wound experiment and conditional nonlinear exponential bounds."""

import hashlib
import importlib
import inspect
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import trapezoid


def decay_constants(lam=5.0, rho=4.0, delta0=0.4, M=1.2, alpha=1.0):
    """
    Constants for g(n)=n(lam-n) and the manuscript's Webb kernel.
    """
    if (not np.isfinite([lam, rho, delta0, M, alpha]).all()
            or rho < 0 or alpha <= 0 or not 0 < delta0 < 1 < M):
        raise ValueError('Require rho>=0, alpha>0, and 0<delta0<1<M, all finite.')
    
    H_rho = 0.2 / np.arctan(2.0) * (
        rho * np.arctan(0.2 * rho) - 2.5 * np.log1p(0.04 * rho**2))
    L_g = max(abs(lam - 2 * delta0), abs(lam - 2 * M))
    C_rho = M * L_g * H_rho
    discriminant = C_rho**2 - 4 * alpha * delta0
    L_stab = (np.inf if discriminant < 0 else
              np.pi * alpha / (C_rho + np.sqrt(discriminant)))
    
    return dict(alpha=float(alpha), lam=float(lam), rho=float(rho),
                delta0=float(delta0), M=float(M), H_rho=float(H_rho),
                L_g=float(L_g), C_rho=float(C_rho), L_stab=float(L_stab))


def mu_L(L, C_rho, delta0=0.4, alpha=1.0):
    """min_{r >= pi/(2L)} (alpha*r**2 - C_rho*r + delta0).

    This is the piecewise sufficient rate, not s_L.
    A nonpositive value gives no positive exponential decay guarantee.
    """
    L = np.asarray(L, dtype=float)
    if (not np.isfinite(L).all() or np.any(L <= 0)
            or not np.isfinite([C_rho, delta0, alpha]).all()
            or C_rho < 0 or delta0 <= 0 or alpha <= 0):
        raise ValueError('Require finite L, delta0, alpha>0 and C_rho>=0.')
    
    r = np.maximum(np.pi / (2 * L), C_rho / (2 * alpha))
    return alpha * r**2 - C_rho * r + delta0


def small_wound_initial(x, a=0.5, d=0.5):
    """n0=1-d*psi(x/a), with a smooth bump supported inside (-a,a)."""
    if not np.isfinite([a, d]).all() or a <= 0 or not 0 < d < 1:
        raise ValueError('Require a>0 and 0<d<1.')
    
    z = np.asarray(x, dtype=float) / a
    psi = np.zeros_like(z)
    inside = np.abs(z) < 1
    psi[inside] = np.exp(1 - 1 / (1 - z[inside]**2))
    
    return 1 - d * psi


def _load_solver(solver_module='wound_healing'):
    try:
        solver = importlib.import_module(solver_module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f'Missing dependency {exc.name!r}. Keep {solver_module}.py '
        ) from exc
    if 'snapshot_callback' not in inspect.signature(solver.wound_healing).parameters:
        raise TypeError(f'Use the version of {solver_module}.py that '
                        'supports snapshot_callback for fractional output times. ')
    return solver


def _source_record(module):
    name = getattr(module, '__file__', None)
    path = Path(name).resolve() if name else None
    return dict(path=str(path) if path else None,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest()
                if path and path.is_file() else None)


def _pdf_path(output_dir):
    folder = (Path.cwd() / 'solution_storage' if output_dir is None
              else Path(output_dir).expanduser())
    return folder.resolve() / 'nonlinear_decay_bounds.pdf'


def plot_small_wound_bounds(cases, constants, output_dir=None, *, show=True):

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    colors = ['#2166ac', '#b35806', '#238b45', '#7b3294']
    L_stab = constants['L_stab']
    ratios = [case['ratio'] for case in cases]
    curve = np.linspace(min(0.5, 0.9 * min(ratios)),
                        max(1.6, 1.1 * max(ratios)), 500)
    style = {'font.family': 'serif', 'font.size': 10, 'mathtext.fontset': 'stix',
             'text.usetex': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
             'axes.spines.top': False, 'axes.spines.right': False}
    
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), layout='constrained')
        left, right = axes
        left.plot(curve, mu_L(curve * L_stab, constants['C_rho'],
                             constants['delta0'], constants['alpha']), color=colors[0])
        left.axhline(0, color='0.5', lw=0.8)
        left.axvline(1, color='0.5', lw=0.8, ls='--')
        left.set(xlabel=r'$L/L_{\mathrm{stab}}$', ylabel=r'$\mu_L$',
                 xlim=(curve[0], curve[-1]))
        left.set_title('(a) Sufficient decay rate', loc='left', fontsize=11)
        left.text(0.97, 0.95, fr'$L_{{\mathrm{{stab}}}}={L_stab:.5f}$',
                  transform=left.transAxes, ha='right', va='top')
        any_bound = False

        for index, case in enumerate(cases):
            color = colors[index % len(colors)]
            left.plot(case['ratio'], case['mu_L'], 'o', color=color, ms=5)
            relative = case['normalized_error']
            right.semilogy(case['t'], np.where(relative > 0, relative, np.nan),
                           color=color, lw=1.6,
                           label=fr'$L/L_{{\mathrm{{stab}}}}={case["ratio"]:g}$')
            if case['mu_L'] > 0 and case['sampled_density_band_ok']:
                right.semilogy(case['t'], case['bound'], color=color, ls='--', lw=1.2)
                any_bound = True

        right.set(xlabel=r'$t$', ylabel=r'$E_L(t)/E_L(0)$',
                  xlim=(0, max(case['t'][-1] for case in cases)))
        right.set_title('(b) Normalized errors and bounds', loc='left', fontsize=11)
        right.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        handles, labels = right.get_legend_handles_labels()

        if any_bound:
            handles.append(Line2D([], [], color='0.3', ls='--', lw=1.2))
            labels.append(r'Conditional bound $e^{-\mu_L t}$')
        right.legend(handles, labels, loc='lower left', fontsize=8, frameon=False)

        # for ax in axes:
        #     ax.grid(alpha=0.15, lw=0.5)
        pdf_path = _pdf_path(output_dir)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f'Saved {pdf_path}', flush=True)

        if show:
            plt.show()
    return fig


def run_small_wound_bounds(*, lam=5.0, rho=4.0, delta0=0.4, M=1.2,
                          ratios=(0.6, 0.9, 1.4), a=0.5, d=0.5,
                          h=0.025, tau=1e-4, T=8.0, output_dt=0.1,
                          deg=5, C=25, quadrature_order=8,
                          output_dir=None, show=True, solver_module='wound_healing'):
    """Run the requested domain sizes using the wound solver, with alpha=1.

    The full interval is (-L,L), where L=ratio*L_stab. Output spacing is
    distinct from the RK4 step size. 
    """
    constants = decay_constants(lam, rho, delta0, M, alpha=1.0)
    ratios = tuple(map(float, ratios))

    if (not ratios or not np.isfinite([h, tau, T, output_dt, a, d, *ratios]).all()
            or min(h, tau, T, output_dt, a, *ratios) <= 0
            or not 0 < d <= 1 - delta0 or len(set(ratios)) != len(ratios)):
        raise ValueError('Use distinct positive ratios, positive mesh/time settings, '
                         'a>0, and 0<d<=1-delta0.')
    
    if not np.isfinite(constants['L_stab']):
        raise ValueError('L_stab is infinite for these parameters, so finite ratios '
                         'L/L_stab cannot define the experiment. Use the default parameters.')
    if a >= min(ratios) * constants['L_stab']:
        raise ValueError('The wound half-width a must be smaller than every domain half-length.')
    if (int(deg) != deg or deg < 1 or int(C) != C or C < 1
            or int(quadrature_order) != quadrature_order or quadrature_order < 1):
        raise ValueError('deg, C, and quadrature_order must be positive integers.')
    
    solver = _load_solver(solver_module)
    pdf_path = _pdf_path(output_dir)
    requested = np.unique(np.append(np.arange(0.0, T, output_dt), T))
    Nt = int(np.ceil(T / tau))
    actual_tau = T / Nt
    expected_steps = np.unique(np.rint(requested / actual_tau).astype(int))
    sources = {name: _source_record(sys.modules.get(name)) for name in
               (solver_module, 'adhesion_product', 'fc', 'scalar_multipliers')}
    settings = dict(constants, ratios=ratios, a=a, d=d, requested_h=h,
                    requested_tau=tau, actual_tau=actual_tau, T=T,
                    output_dt=output_dt, deg=deg, C=C, quadrature_order=quadrature_order,
                    flux_derivative='fc_spectral', sources=sources)
    cases = []
    for ratio in ratios:
        L = ratio * constants['L_stab']
        rate = float(mu_L(L, constants['C_rho'], delta0))
        if round(2 * L / h) + 1 < 2 * deg:
            raise ValueError('Too few grid points for the boundary matching data; reduce h.')
        grid = solver.domain_prep((-L, L), h, rho, C=C)
        x = grid.x
        steps, times, profiles = [], [], []

        def snapshot(step, time, physical):
            physical = np.asarray(physical, dtype=float)
            if physical.shape != x.shape or not np.isfinite(physical).all():
                raise FloatingPointError(f'Invalid PDE snapshot at t={time:g}.')
            if not np.array_equal(physical[[0, -1]], [1.0, 1.0]):
                raise RuntimeError('Prescribed endpoint values were not retained.')
            steps.append(int(step))
            times.append(float(time))
            profiles.append(physical.copy())

        print(f'L/L_stab={ratio:g}: L={L:.9f}, mu_L={rate:.9f}', flush=True)
        # save_sol selects RK4 steps; the callback receives the profiles in memory.
        final = solver.wound_healing(
            interval=(-L, L), rho=rho, lam=lam,
            u_init=lambda xx: small_wound_initial(xx, a=a, d=d),
            bdry_condition=lambda xx, tt: np.ones_like(xx, dtype=float),
            h=h, T=T, tau=tau, deg=deg, C=C,
            adhesion_kernel=solver.webb_adhesion_kernel,
            flux_derivative='fc_spectral', quadrature_order=quadrature_order,
            save_sol=requested, snapshot_callback=snapshot)
        
        if (not np.array_equal(steps, expected_steps) or times[-1] != T
                or not np.array_equal(profiles[-1], final)):
            raise RuntimeError('Sampled outputs do not match the requested RK4 steps/final state.')
        
        t, n = np.asarray(times), np.asarray(profiles)
        E = np.sqrt(trapezoid((n - 1)**2, x=x, axis=1) / (2 * L))

        if E[0] == 0:
            raise ValueError('The mesh does not resolve a nonzero initial wound.')
        
        normalized = E / E[0]
        bound = np.exp(-rate * t) if rate > 0 else np.full_like(t, np.nan)
        nmin, nmax = n.min(axis=1), n.max(axis=1)
        band_ok = bool(np.all(nmin >= delta0) and np.all(nmax <= M))
        excess = float(np.maximum(normalized - bound, 0).max()) if rate > 0 else None
        case = dict(ratio=ratio, L=L, mu_L=rate, x=x, t=t, n=n, E=E,
                    normalized_error=normalized, bound=bound,
                    U=np.max(np.abs(n - 1), axis=1), sampled_density_band_ok=band_ok)
        report = dict(ratio=ratio, L=L, mu_L=rate, actual_h=float(grid.h),
                      actual_tau=actual_tau, initial_E=float(E[0]),
                      final_normalized_error=float(normalized[-1]),
                      sampled_min_density=float(nmin.min()), sampled_max_density=float(nmax.max()),
                      sampled_density_band_ok=band_ok, conditional_bound_drawn=rate > 0 and band_ok,
                      max_positive_normalized_bound_excess=excess)
        case.update(steps=np.asarray(steps), actual_h=float(grid.h), actual_tau=actual_tau,
                    sampled_min_n=nmin, sampled_max_n=nmax, settings=settings, summary=report,
                    output_dir=str(pdf_path.parent), pdf_path=str(pdf_path))
        cases.append(case)

        print(f'  Sampled density range: [{nmin.min():.8g}, {nmax.max():.8g}]; '
              f'final E/E(0)={normalized[-1]:.6g}', flush=True)
        if not band_ok:
            print('  Density-band violation detected: the dashed bound is omitted.', flush=True)
        if excess is not None and excess > 1e-8:
            print(f'  Recorded error exceeds the exponential bound by up to {excess:.3g}. '
                  'Check the hypotheses and spatial/time convergence.', flush=True)
            
    fig = plot_small_wound_bounds(cases, constants, pdf_path.parent, show=show)
    return fig, cases


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rates-only', action='store_true', help='Print analytical rates without FC.')
    parser.add_argument('--output', type=Path, metavar='DIR',
                        help='PDF directory (default: ./solution_storage).')
    parser.add_argument('--no-show', action='store_true', help='Save the figure without opening a window.')
    parser.add_argument('--h', type=float, default=0.025, help='Requested mesh spacing.')
    parser.add_argument('--tau', type=float, default=1e-4, help='Requested RK4 time step.')
    args = parser.parse_args()
    
    if args.rates_only:
        constants = decay_constants()
        print(f'C_rho = {constants["C_rho"]:.12f}\nL_stab = {constants["L_stab"]:.12f}')
        print('L/L_stab       L               mu_L          Positive decay bound')
        for ratio in (0.6, 0.9, 1.4):
            L = ratio * constants['L_stab']
            rate = float(mu_L(L, constants['C_rho']))
            print(f'{ratio:6.1f}    {L:13.9f}    {rate:13.9f}    {rate > 0}')
    else:
        if args.no_show:
            import matplotlib
            matplotlib.use('Agg')
        try:
            run_small_wound_bounds(h=args.h, tau=args.tau,
                                  output_dir=args.output, show=not args.no_show)
        except (ModuleNotFoundError, TypeError, ValueError) as exc:
            parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
