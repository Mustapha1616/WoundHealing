"""
Compare the wound healing simulation with the leading linear rates.

This runs the full original nonlinear wound healing equation for small
eigenfunction perturbations. It does not advance the linearized equation.
"""
import argparse
import hashlib
import importlib
import inspect
from pathlib import Path
from runpy import run_path
import sys

import numpy as np
from scipy.integrate import trapezoid

def _stability_functions():
    """Load the adjacent renamed script without running its Figure 1 experiment."""
    path = Path(__file__).resolve().with_name('wound_stability.py')
    if path.is_file():
        module = run_path(str(path))
        return module['compute_s_L'], module['eigenfunction']
    module = importlib.import_module('wound_stability')
    return module.compute_s_L, module.eigenfunction


def _load_solver(name):
    try:
        solver = importlib.import_module(name) if isinstance(name, str) else name
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f'Cannot import {name!r}: dependency {exc.name!r} is unavailable. '
            'Make your original fc and scalar_multipliers modules importable. '
        ) from exc
    if 'snapshot_callback' not in inspect.signature(solver.wound_healing).parameters:
        raise TypeError(
            'Use the version of wound_healing.py that accepts snapshot_callback: '
            'its snapshot_callback supports output spacing 0.25. '
        )
    return solver


def _source_record(module):
    name = getattr(module, '__file__', None)
    path = Path(name).resolve() if name else None
    return {'path': str(path) if path else None,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()
            if path and path.is_file() else None}


def _pdf_path(output_dir):
    folder = Path.cwd() / 'solution_storage' if output_dir is None else Path(output_dir).expanduser()
    return folder.resolve() / 'modal_rates.pdf'


def analyze_projection(x, t, profiles, phi, *, fit_start=2.0,
                       fit_end=None, smallness=1e-2, amplitude_floor=None):
    """
    Compute a(t)=integral (n-1)*phi dx and fit log|a(t)| versus actual time.
    """
    error = np.asarray(profiles) - 1.0
    amplitude = trapezoid(error*phi[None, :], x=x, axis=1)
    uniform_error = np.max(np.abs(error), axis=1)

    if amplitude_floor is None:
        amplitude_floor = 100*np.finfo(float).eps*trapezoid(np.abs(phi), x=x)
    amplitude_floor = float(amplitude_floor)
    if amplitude_floor <= 0 or not np.isfinite(amplitude_floor):
        raise ValueError('amplitude_floor must be finite and positive.')
    if abs(amplitude[0]) <= amplitude_floor:
        raise ValueError('Initial projection is below the selected amplitude floor.')
    
    normalized = np.abs(amplitude)/abs(amplitude[0])
    stop = t[-1] if fit_end is None else min(float(fit_end), t[-1])
    ids = np.flatnonzero((t >= fit_start) & (t <= stop))
    valid = ((uniform_error <= smallness) & (np.abs(amplitude) > amplitude_floor)
             & (amplitude*amplitude[0] > 0))
    bad = np.flatnonzero(~valid[ids])

    if bad.size:
        ids = ids[:bad[0]]

    fit_mask = np.zeros(len(t), dtype=bool)
    fit_mask[ids] = True
    fit = {'rate': None, 'r_squared': None, 'points': len(ids),
           'time_interval': None, 'status': 'insufficient_eligible_outputs'}
    
    if len(ids) >= 5 and np.ptp(t[ids]) > 0:
        y = np.log(np.abs(amplitude[ids]))
        rate, intercept = np.polyfit(t[ids], y, 1)
        residual = y - (rate*t[ids]+intercept)
        variation = np.sum((y-y.mean())**2)
        fit.update(rate=float(rate), intercept=float(intercept),
                   r_squared=float(1-np.sum(residual**2)/variation) if variation > 0 else None,
                   time_interval=[float(t[ids[0]]), float(t[ids[-1]])],
                   status='fitted')
        
    return {'amplitude': amplitude, 'normalized': normalized,
            'uniform_error': uniform_error, 'fit_mask': fit_mask,
            'fit': fit, 'amplitude_floor': amplitude_floor}


def plot_modal_rates(results, output_dir=None, *, show=True):
    """Plot measured projections and independent Galerkin predictions."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    colors = ['#2166ac', '#b35806', '#238b45', '#7b3294']
    markers = ['o', 's', '^', 'd']
    style = {'font.family': 'serif', 'font.size': 10, 'mathtext.fontset': 'stix',
             'text.usetex': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
             'axes.spines.top': False, 'axes.spines.right': False}
    
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, len(results), squeeze=False,
                                figsize=(4.4*len(results), 3.6), layout='constrained')
        for column, case in enumerate(results):
            ax = axes[0, column]
            s = case['mode']['s_L']
            notes = [fr'$s_L={s:.6f}$']
            for index, run in enumerate(case['runs']):
                eps = run['epsilon']
                power = int(np.rint(np.log10(eps)))
                eps_text = (fr'10^{{{power}}}' if np.isclose(eps, 10.**power, rtol=1e-12, atol=0)
                            else f'{eps:g}')
                visible = np.where(np.abs(run['amplitude']) > run['amplitude_floor'],
                                   run['normalized'], np.nan)
                ax.semilogy(run['t'], visible, color=colors[index % len(colors)],
                            marker=markers[index % len(markers)], markevery=12,
                            ms=3, lw=1.4, label=fr'Simulation, $\varepsilon={eps_text}$')
                fitted = run['fit']['rate']
                value = f'{fitted:.6f}' if fitted is not None else 'not fitted'
                notes.append(fr'Fit ($\varepsilon={eps_text}$): {value}')
            t = case['runs'][0]['t']
            ax.semilogy(t, np.exp(s*t), color='.15', ls='--', lw=1.2,
                        label=r'Predicted $e^{s_Lt}$')
            decay = s < 0
            ax.text(.97 if decay else .03, .97, '\n'.join(notes),
                    transform=ax.transAxes, va='top',
                    ha='right' if decay else 'left', fontsize=8)
            status = 'decay' if decay else ('growth' if s > 0 else 'neutral')
            ax.set_title(fr'({chr(97+column)}) $\rho={case["rho"]:g}$: modal {status}',
                         loc='left', fontsize=11)
            #ax.set(xlabel=r'$t$', ylabel=r'$|a(t)|/|a(0)|$', xlim=(0, t[-1]))
            ax.set(xlabel=r'$t$', ylabel=r'$|\langle n(\cdot,t)-1,\phi\rangle|/|\langle n_0-1,\phi\rangle|$',
                   xlim=(0, t[-1]))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            #ax.grid(alpha=.15, lw=.5)
            ax.legend(loc='lower left' if decay else 'lower right', fontsize=8, frameon=False)
        first = results[0]
        #fig.suptitle(fr'$L={first["L"]:g}$, $\lambda={first["lam"]:g}$')
        pdf_path = _pdf_path(output_dir)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        if show:
            plt.show()
    return fig


def run_modal_validation(*, solver_module='wound_healing',
                         L=100.0, lam=5.0, rhos=(16.0, 18.0),
                         epsilons=(1e-4, 1e-5), T=30.0,
                         h=0.125, tau=0.0025, output_dt=0.25,
                         deg=5, C=25, quadrature_order=8,
                         flux_derivative='fc_spectral',
                         fit_start=2.0, fit_end=None, smallness=1e-2,
                         amplitude_floor=None, m0=96, max_m=1536,
                         tol_s=1e-7, tol_q=1e-9,
                         output_dir=None, show=True, verbose=True):
    """
    Run the full wound model; return (figure, list of radius-case results).
    """
    rhos, epsilons = tuple(map(float, rhos)), tuple(map(float, epsilons))
    parameters = [L, lam, T, h, output_dt, smallness, fit_start, *rhos, *epsilons]
    
    if (not np.isfinite(parameters).all() or min(L, T, h, output_dt, smallness) <= 0
            or fit_start < 0 or not rhos or not epsilons
            or min(rhos) < 0 or min(epsilons) <= 0
            or len(set(rhos)) != len(rhos) or len(set(epsilons)) != len(epsilons)):
        raise ValueError('Use finite positive sizes/times, nonnegative radii, and distinct cases.')
    if tau is not None and (not np.isfinite(tau) or tau <= 0):
        raise ValueError('tau must be positive or None.')
    if fit_end is not None and (not np.isfinite(fit_end) or fit_end <= fit_start):
        raise ValueError('fit_end must exceed fit_start.')
    
    solver = _load_solver(solver_module)
    compute_s_L, eigenfunction = _stability_functions()
    kernel = solver.webb_adhesion_kernel
    source = {'solver': _source_record(solver), 'fc': _source_record(getattr(solver, 'fc', None)),
              'scalar_multipliers': _source_record(sys.modules.get(
                  getattr(getattr(solver, 'L_f_1', None), '__module__', '')))}
    pdf_path = _pdf_path(output_dir)
    requested = np.unique(np.append(np.arange(0.0, T, output_dt), T))
    results = []

    if verbose:
        print(f'Calling {getattr(solver, "__name__", "supplied solver")}; output: {pdf_path}', flush=True)
    for rho in rhos:
        if verbose:
            print(f'rho={rho:g}: computing the leading bounded eigenpair...', flush=True)
        mode = compute_s_L(L, rho, lam, alpha=1.0, m0=m0, max_m=max_m,
                           tol_s=tol_s, tol_q=tol_q, kernel_fun=kernel)
        grid = solver.domain_prep((-L, L), h, rho, C=C)
        x = grid.x
        phi = eigenfunction(x, mode)
        phi_max = float(np.max(np.abs(phi)))
        if phi_max == 0:
            raise ValueError('The selected grid does not resolve the leading eigenfunction.')
        case = {'L': float(L), 'lam': float(lam), 'rho': rho, 'mode': mode,
                'x': x, 'phi': phi, 'runs': [], 'output_dir': str(pdf_path.parent)}
        for epsilon in epsilons:
            snapshots, actual_times, steps = [], [], []

            def initial(x_values):
                return 1.0 + epsilon*eigenfunction(x_values, mode)/phi_max

            def boundary(x_values, time):
                return np.ones_like(x_values)

            def collect(step, time, state):
                if state.shape != x.shape or not np.isfinite(state).all():
                    raise FloatingPointError('Nonfinite or incorrectly shaped PDE snapshot.')
                if not np.allclose(state[[0, -1]], 1.0, atol=1e-12, rtol=0):
                    raise RuntimeError('The endpoint density must remain 1.')
                steps.append(int(step))
                actual_times.append(float(time))
                snapshots.append(state.copy())

            if verbose:
                print(f'  epsilon={epsilon:g}, h={grid.h:g}, requested dt={tau}', flush=True)
            final = solver.wound_healing(
                interval=(-L, L), rho=rho, lam=lam, u_init=initial,
                bdry_condition=boundary, h=h, T=T, tau=tau, deg=deg, C=C,
                adhesion_kernel=kernel, flux_derivative=flux_derivative,
                quadrature_order=quadrature_order, save_sol=requested,
                snapshot_callback=collect,
            )
            t = np.asarray(actual_times)
            profiles = np.asarray(snapshots)

            if len(t) < 2 or t[0] != 0 or t[-1] != T or np.any(np.diff(t) <= 0):
                raise RuntimeError('Outputs must include distinct initial and final times.')
            if not np.array_equal(profiles[-1], final):
                raise RuntimeError('The final callback state differs from the solver return.')
            actual_tau = T/steps[-1]
            expected_steps = np.unique(np.rint(requested/actual_tau).astype(int))
            if not np.array_equal(steps, expected_steps):
                raise RuntimeError('The callback output steps do not match the requested schedule.')
            
            diagnostics = analyze_projection(
                x, t, profiles, phi, fit_start=fit_start, fit_end=fit_end,
                smallness=smallness, amplitude_floor=amplitude_floor)
            metadata = {
                'L': float(L), 'rho': rho, 'lam': float(lam), 'alpha': 1.0,
                'epsilon': epsilon, 'T': float(T), 'actual_h': float(grid.h),
                'requested_tau': tau, 'actual_tau': float(actual_tau),
                'requested_output_spacing': output_dt,
                'min_actual_output_spacing': float(np.min(np.diff(t))),
                'max_actual_output_spacing': float(np.max(np.diff(t))),
                'time_steps': steps[-1], 'output_count': len(t),
                'deg': deg, 'C': C, 'quadrature_order': quadrature_order,
                'flux_derivative': flux_derivative, 'kernel': kernel.__name__,
                'predicted_rate': mode['s_L'], 'fit': diagnostics['fit'],
                'amplitude_floor': diagnostics['amplitude_floor'],
                'smallness_cutoff': smallness, 'source': source,
                'note': 'Projection floor screens roundoff; assess discretization error by refinement.',
            }
            case['runs'].append(dict(diagnostics, epsilon=epsilon, t=t,
                                     n=profiles, summary=metadata))
            if verbose:
                fitted = diagnostics['fit']['rate']
                print(f'    predicted={mode["s_L"]:.8f}; fitted={fitted}', flush=True)
        results.append(case)

    fig = plot_modal_rates(results, pdf_path.parent, show=show)
    if verbose:
        print(f'Saved {pdf_path}', flush=True)
    return fig, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot the predicted and fitted modal rates.')
    parser.add_argument('--no-show', action='store_true', help='Save the PDF without opening a plot window.')
    parser.add_argument('--output', metavar='DIR', help='PDF output directory (default: ./solution_storage).')
    args = parser.parse_args()
    figure, results = run_modal_validation(output_dir=args.output, show=not args.no_show)
