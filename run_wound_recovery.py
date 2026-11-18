"""
Run the wound recovery experiments with wound_healing.py and save only the PDF.
"""
import hashlib
import importlib
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import trapezoid


SOLVER_MODULE = 'wound_healing'
PDF_FILENAME = 'wound_recovery.pdf'

# L is the HALF-length: L=100 means the interval (-100, 100).
# The original solver has diffusion coefficient alpha=1.
CASES = ((30.0, 400.0), (100.0, 400.0), (200.0, 800.0))  # (L, final time)
SETTINGS = {
    'rho': 16.0,
    'lam': 5.0,
    'h': 0.125,
    'tau': None,                 # use the original solver's automatic step
    'deg': 5,
    'C': 25,
    'flux_derivative': 'fd',     # choice for this recovery experiment
    'output_dt': 1.0,            # requested spacing for profiles held in memory
    'healing_tol': 1e-2,
    'hold_time': 20.0,
}


def bdry_condition(x, t):
    return np.ones_like(x)


def u_init(x):
    return 1.0 - 0.75 * np.exp(-(0.1 * x)**6)


def _load_solver(name):
    if not isinstance(name, str):
        solver = name
    else:
        try:
            solver = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Cannot import solver {name!r}: {exc.name!r} is unavailable. "
            ) from exc
    for attribute in ('wound_healing', 'domain_prep', 'webb_adhesion_kernel'):
        if not callable(getattr(solver, attribute, None)):
            raise TypeError(f'The solver must provide {attribute}().')
    return solver


def _source_record(module):
    filename = getattr(module, '__file__', None)
    if filename is None:
        return {'file': None, 'sha256': None}
    path = Path(filename).resolve()
    return {'file': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _requested_times(T, output_dt):
    """Sample from t=0 through t=T; the solver supplies the actual RK4 times."""
    if not np.isfinite([T, output_dt]).all() or T <= 0 or output_dt <= 0:
        raise ValueError('T and output_dt must be finite and positive.')
    requested = np.arange(0.0, T, output_dt)
    return np.append(requested[requested < T], T)


def _diagnostics(x, times, profiles, healing_tol, hold_time):
    if profiles.shape != (len(times), len(x)) or not np.all(np.isfinite(profiles)):
        raise ValueError('Sampled profiles have an unexpected shape or nonfinite values.')
    if not np.allclose(profiles[:, [0, -1]], 1.0, atol=1e-12, rtol=0):
        raise ValueError('Sampled profiles do not satisfy the prescribed endpoint values.')
    
    error = profiles - 1.0
    U = np.max(np.abs(error), axis=1)
    E = np.sqrt(trapezoid(error**2, x=x, axis=1) / (x[-1] - x[0]))
    future_max = np.maximum.accumulate(U[::-1])[::-1]
    eligible = np.flatnonzero((times <= times[-1] - hold_time) & (future_max <= healing_tol))
    healing_time = float(times[eligible[0]]) if eligible.size else None

    summary = {
        'observed_healing_time': healing_time,
        'healing_status': ('observed_at_stored_times' if healing_time is not None
                           else 'not_observed_with_required_hold_time'),
        'final_uniform_error': float(U[-1]),
        'final_normalized_L2_error': float(E[-1]),
        'final_pattern_amplitude': float(np.ptp(profiles[-1])),
        'minimum_sampled_density': float(np.min(profiles)),
        'maximum_sampled_density': float(np.max(profiles)),
        'maximum_actual_output_gap': float(np.max(np.diff(times))),
        'note': 'Healing and density extrema are checked at stored times only.',
    }
    return U, E, summary


def _run_case(solver, L, T, settings, source):
    L, T = float(L), float(T)
    h, rho, lam = (float(settings[k]) for k in ('h', 'rho', 'lam'))

    if not np.isfinite([L, h, rho, lam]).all() or min(L, h, rho) <= 0:
        raise ValueError('Require finite parameters with L, h, rho > 0.')
    if int(round(2*L/h)) < 2:
        raise ValueError('The physical grid needs at least two cells.')
    if settings['flux_derivative'] not in ('fd', 'fc_spectral'):
        raise ValueError("flux_derivative must be 'fd' or 'fc_spectral'.")
    
    tau = settings['tau']
    if tau is not None and (not np.isfinite(tau) or tau <= 0):
        raise ValueError('The requested time step must be finite and positive.')
    grid = solver.domain_prep((-L, L), h, rho, C=settings['C'])
    requested = _requested_times(T, settings['output_dt'])
    samples = []

    def collect(step, time, profile):
        # wound_healing passes a copy of the physical profile at this time.
        samples.append((step, time, profile))

    print(f'L={L:g}, T={T:g}, h={grid.h:g}', flush=True)

    # This is the user's function. No time-stepping loop is implemented here.
    final = solver.wound_healing(
        interval=(-L, L), rho=rho, lam=lam,
        u_init=u_init, bdry_condition=bdry_condition,
        h=h, T=T, tau=tau, deg=settings['deg'], C=settings['C'],
        adhesion_kernel=solver.webb_adhesion_kernel,
        flux_derivative=settings['flux_derivative'],
        save_sol=requested, snapshot_callback=collect,
    )

    if not samples or samples[0][1] != 0.0 or samples[-1][1] != T:
        raise RuntimeError('The solver did not report both the initial and final profiles.')
    steps, times, profiles = zip(*samples)
    times = np.asarray(times, dtype=float)
    profiles = np.stack(profiles)
    Nt = int(steps[-1])
    if not np.array_equal(profiles[-1], final):
        raise RuntimeError('The final sampled profile differs from the returned solution.')
    U, E, diagnostics = _diagnostics(
        grid.x, times, profiles, settings['healing_tol'], settings['hold_time'])
    metadata = dict(settings, L=L, T=T, alpha=1.0, actual_h=float(grid.h),
                    actual_tau=T/Nt, time_steps=Nt, solver=source,
                    kernel='webb_adhesion_kernel', status='completed', **diagnostics)
    healing = diagnostics['observed_healing_time']
    print(f'  Final U={U[-1]:.6g}; observed healing time={healing}', flush=True)
    return {'x': grid.x, 't': times, 'n': profiles, 'U': U, 'E': E,
            'summary': metadata}



def plot_results(results, output_dir=None, show=True):
    """
    Plot solver results
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output_dir = (Path.cwd() / 'solution_storage' if output_dir is None
                  else Path(output_dir).expanduser())
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / PDF_FILENAME

    BLUE, ORANGE, GREEN = '#2166ac', '#b35806', '#238b45'
    PURPLE, GRAY = '#7b3294', '0.3'
    EXTRA_COLOR = '#a50f15' 
    style = {'font.family': 'serif', 'font.size': 10, 'mathtext.fontset': 'stix',
             'text.usetex': False, 'pdf.fonttype': 42,
             'axes.spines.top': False, 'axes.spines.right': False}
    
    with plt.rc_context(style):
        ncols = len(results)
        fig, axes = plt.subplots(2, ncols, squeeze=False,
                                figsize=(4.2*ncols, 6.5),
                                layout='constrained')
        for col, row in enumerate(results):
            x, t, n = row['x'], row['t'], row['n']
            info = row['summary']
            # Explicit colors keep a time's color independent of plotting order.
            desired = [(0, GRAY), (5, PURPLE), (20, ORANGE), (100, GREEN),
                       (200, EXTRA_COLOR), (t[-1], BLUE)]
            selected = {}
            for target, color in desired:
                if target <= t[-1]:
                    j = int(np.argmin(abs(t-target)))
                    selected[j] = color  # the final profile always uses blue
            ax = axes[0, col]
            for j, color in sorted(selected.items()):
                ax.plot(x, n[j], color=color, lw=1.25,
                        label=fr'$t={t[j]:.0f}$')
            ax.axhline(1.0, color='.5', lw=.7, ls='--')
            ax.set(xlabel=r'$x$', ylabel=r'$n(x,t)$', xlim=(x[0], x[-1]))
            ax.set_title(fr'({chr(97+col)}) $L={info["L"]:g}$: density',
                         loc='left', fontsize=11, pad=10)
            ax.legend(fontsize=8, frameon=False)
            #ax.grid(alpha=.15)
            ax = axes[1, col]
            for values, label, color in ((row['U'], r'$U_L$', BLUE),
                                         (row['E'], r'$E_L$', ORANGE)):
                # Exact zeros are omitted on a logarithmic axis, not clipped.
                visible = np.where(values > 0, values, np.nan)
                ax.semilogy(t, visible, label=label, color=color, lw=1.4)
            ax.axhline(info['healing_tol'], color='.4', ls='--', lw=1.0,
                       label='Healing tolerance')
            ax.set(xlabel=r'$t$', ylabel='Error', xlim=(0, t[-1]))
            ax.set_title(fr'({chr(97+ncols+col)}) $L={info["L"]:g}$: errors',
                         loc='left', fontsize=11, pad=10)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            #ax.grid(alpha=.15)
            ax.legend(fontsize=8, frameon=False)
            healing = info['observed_healing_time']
            label = (fr'$T_{{\mathrm{{heal}}}}\approx {healing:.0f}$'
                     if healing is not None else 'Healing criterion not met')
            ax.text(.97, .97, label, transform=ax.transAxes,
                    ha='right', va='top', fontsize=9)
        info = results[0]['summary']
        #fig.suptitle(fr'Wound recovery: $\lambda={info["lam"]:g}$, $\rho={info["rho"]:g}$')
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f'Saved {pdf_path.resolve()}', flush=True)
        if show:
            plt.show()
    return fig


def run_experiments(*, solver_module=SOLVER_MODULE, cases=CASES,
                    output_dir=None, show=True, **overrides):
    """
    Run (L,T) cases; return (figure, list of results).
    """
    unknown = set(overrides) - set(SETTINGS)
    if unknown:
        raise TypeError(f'Unknown setting(s): {sorted(unknown)}')
    settings = {**SETTINGS, **overrides}
    if (not np.isfinite([settings['healing_tol'], settings['hold_time']]).all()
            or settings['healing_tol'] <= 0 or settings['hold_time'] < 0):
        raise ValueError('Require healing_tol>0 and hold_time>=0.')
    cases = tuple(cases)
    if not cases or len(set(cases)) != len(cases):
        raise ValueError('Supply a nonempty sequence of distinct (L,T) tuples.')
    solver = _load_solver(solver_module)
    source = _source_record(solver)
    source['fc'] = _source_record(getattr(solver, 'fc', None))
    multiplier = getattr(solver, 'L_f_1', None)
    source['scalar_multipliers'] = _source_record(
        sys.modules.get(getattr(multiplier, '__module__', '')))
    print(f'Calling solver: {getattr(solver, "__name__", type(solver).__name__)}', flush=True)
    results = [_run_case(solver, L, T, settings, source) for L, T in cases]
    fig = plot_results(results, output_dir, show=show)
    return fig, results


if __name__ == '__main__':
    figure, results = run_experiments()
