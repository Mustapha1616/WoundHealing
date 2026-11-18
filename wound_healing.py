"""
Bounded domain wound healing
Author: Ilyas Mustapha
Date: November 17, 2026
"""

from pathlib import Path
import numpy as np
from numpy.fft import fft, fftfreq, ifft
from adhesion_product import build_product_adhesion_force

import fc
from scalar_multipliers import L_f_1


class domain_prep:

    def __init__(self, interval, h, rho, C=25):

        self.xl, self.xr = map(float, interval)
        self.L = self.xr - self.xl  # length of the interval
        cells = int(round(self.L / h))
        self.N = cells + 1
        self.h = self.L / cells
        self.x = np.linspace(self.xl, self.xr, self.N)
        self.rho = float(rho)
        self.C = int(C)

        # The returned solution retains the original [collar | physical | collar] layout.
        self.extra = int(np.ceil(self.rho / self.h))
        self.pN = self.N + 2 * self.extra
        self.px = self.xl + (np.arange(self.pN) - self.extra) * self.h
        self.pxl, self.pxr = self.px[0], self.px[-1]
        self.physical_slice = slice(self.extra, self.extra + self.N)
        self.pind = np.r_[
            :self.extra + 1,
            self.extra + self.N - 1:self.pN,
        ]

        # FC differentiates physical data, avoiding slope jumps into the collar.
        self.eN = self.N + self.C
        self.eL = self.eN * self.h
        self.ex = self.xl + np.arange(self.eN) * self.h


def default_reaction(u):
    return u * (1.0 - u)


def default_adhesion_law(u, lam):
    return u * (lam - u)


def webb_adhesion_kernel(z, radius):
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    mask = np.abs(z) <= radius
    out[mask] = 0.1 * np.arctan(0.2 * z[mask]) / np.arctan(2.0)
    return out


def polynomial_adhesion_kernel(z, radius):
    z = np.asarray(z, dtype=float)
    out = np.zeros_like(z)
    if radius == 0:
        return out
    mask = np.abs(z) <= radius
    eta = z[mask] / radius
    out[mask] = (3.0 * z[mask] / radius**2) * (1.0 - eta**2)**2
    return out


def build_adhesion_force(grid, lam, kernel_fun, adhesion_law=default_adhesion_law,
                         quadrature_order=8):
    """
    Return the product-integration adhesion velocity on physical nodes.
    Input u includes the prescribed collar. Sensing endpoints are resolved
    without rounding rho to the grid, including partially covered cells.
    """
    return build_product_adhesion_force(
        grid, lam, kernel_fun, adhesion_law, quadrature_order=quadrature_order)


def fc_derivative(values, multiplier, Q, E):
    """
    Differentiate physical values through a smooth Fourier continuation.
    This can be interpreted as applying the classical laplacian on values.
    """
    # The derivative of a constant is zero; subtract it to preserve equilibria exactly.
    extended = fc.continue_function(values - values[0], Q, E)
    return ifft(multiplier * fft(extended)).real[:len(values)]


def build_rhs(
        grid,
        lam,
        bdry_condition,
        deg=5,
        reaction=default_reaction,
        adhesion_law=default_adhesion_law,
        adhesion_kernel='webb',
        flux_derivative='fc_spectral',
        quadrature_order=8
    ):
    """Build u_t = u_xx - (u F)_x + reaction(u), evolving physical interiors only."""

    if flux_derivative not in ('fd', 'fc_spectral'):
        raise ValueError("flux_derivative must be 'fd' or 'fc_spectral'.")

    if callable(adhesion_kernel):
        kernel_fun = adhesion_kernel
    elif adhesion_kernel == 'webb':
        kernel_fun = webb_adhesion_kernel
    elif adhesion_kernel == 'polynomial':
        kernel_fun = polynomial_adhesion_kernel
    else:
        raise ValueError(
            "adhesion_kernel must be 'webb', 'polynomial', or a callable kernel(z, radius)."
        )

    Q, E = fc.generate_continuations(0.15, 25, 29, deg, grid.C)
    laplacian = L_f_1(grid.eN, grid.eL)
    ik = 1j * 2.0 * np.pi * fftfreq(grid.eN, d=grid.h)

    if grid.eN % 2 == 0:
        ik[grid.eN // 2] = 0.0

    adhesion_force = build_adhesion_force(
        grid, lam, kernel_fun, adhesion_law, quadrature_order=quadrature_order)
    boundary_x = grid.px[grid.pind]

    def f(u, t):
        # Each RK4 stage needs the boundary/exterior data at its own stage time.
        state = np.array(u, dtype=float, copy=True)
        state[grid.pind] = bdry_condition(boundary_x, t)
        physical_u = state[grid.physical_slice]

        diffusion = fc_derivative(physical_u, laplacian, Q, E)
        flux = physical_u * adhesion_force(state)

        if flux_derivative == 'fd':
            adhesion = np.zeros(grid.N)
            adhesion[1:-1] = -(flux[2:] - flux[:-2]) / (2.0 * grid.h)
        else:
            # Continue the physical flux using the user's existing FC routine.
            adhesion = -fc_derivative(flux, ik, Q, E)

        rhs = np.zeros(grid.pN)
        rhs[grid.physical_slice] = diffusion + adhesion + reaction(physical_u)
        rhs[grid.pind] = 0.0

        return rhs

    return f


def wound_healing(
        interval,
        rho,
        lam,
        u_init,
        bdry_condition,
        h,
        T,
        tau=None,
        deg=5,
        C=25,
        reaction=default_reaction,
        adhesion_law=default_adhesion_law,
        adhesion_kernel=webb_adhesion_kernel,
        flux_derivative='fc_spectral',
        save_sol=None,
        storage_dir=None,
        quadrature_order=8,
        snapshot_callback=None
    ):
    """
    Solve with classical RK4 and return the final physical domain state.

    bdry_condition(x, t) prescribes both endpoints and the rho-collar; it may
    return a scalar or an array. Only physical interior nodes evolve.

    adhesion_kernel accepts 'webb', 'polynomial', or a callable kernel(z, radius)
    returning a real array with the same shape as z.

    Product integration restricts its integrals to [-rho, rho], so the
    callable need not apply that cutoff. quadrature_order controls the
    Gauss rule on each intersected interpolation cell; the default is 8.
    flux_derivative='fc_spectral' uses the existing fc_derivative routine.
    """
    grid = domain_prep(interval, h, rho, C)

    if snapshot_callback is not None and not callable(snapshot_callback):
        raise TypeError('snapshot_callback must be callable or None.')

    if tau is None:
        tau = 0.3 / np.max(np.abs(L_f_1(grid.eN, grid.eL)))

    Nt = int(np.ceil(T / tau))
    tau = T / Nt if Nt else 0.0
    print(tau)

    save_steps = set()
    if save_sol is not None:
        times = np.atleast_1d(np.asarray(save_sol, dtype=float))

        if times.ndim != 1 or not np.all(np.isfinite(times)) or np.any((times < 0) | (times > T)):
            raise ValueError('save_sol must contain finite times in [0, T].')
        if times.size:
            save_steps = set(np.rint(times / tau).astype(int)) if Nt else {0}

        storage_dir = (Path(__file__).resolve().parent / 'solution_storage'
                       if storage_dir is None else Path(storage_dir))

    def save_solution(step):
        if step in save_steps:
            time = T if step == Nt else step * tau
            if snapshot_callback is not None:
                snapshot_callback(step, time, u[grid.physical_slice].copy())
            else:
                storage_dir.mkdir(parents=True, exist_ok=True)
                name = f'u_rho{rho}_lam{lam}_T{time:.0f}_interval{grid.xr:g}.npy'
                np.save(storage_dir / name, u[grid.physical_slice])

    u = np.empty(grid.pN)
    u[grid.physical_slice] = u_init(grid.x)
    u[grid.pind] = bdry_condition(grid.px[grid.pind], 0.0)

    if not np.all(np.isfinite(u)):
        raise ValueError('Initial and boundary values must be finite.')

    save_solution(0)

    if Nt == 0:
        return u[grid.physical_slice].copy()

    f = build_rhs(grid, lam, bdry_condition, deg, reaction, adhesion_law,
                  adhesion_kernel, flux_derivative, quadrature_order)

    for it in range(Nt):
        t = it * tau
        t_next = T if it + 1 == Nt else (it + 1) * tau
        k1 = f(u, t)
        k2 = f(u + 0.5 * tau * k1, t + 0.5 * tau)
        k3 = f(u + 0.5 * tau * k2, t + 0.5 * tau)
        k4 = f(u + tau * k3, t_next)
        u += (tau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        u[grid.pind] = bdry_condition(grid.px[grid.pind], t_next)
        save_solution(it + 1)

    return u[grid.physical_slice].copy()
