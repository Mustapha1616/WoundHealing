# Nonlocal Wound Healing on Bounded Domains
Stability thresholds and simulations for a nonlocal advection diffusion model of wound healing on bounded domains

The repository contains the routines used to compute the stability thresholds
and to run the wound healing simulations reported in the paper.

## The model

We study the cell density $n(x,t)$ governed by

$$
n_t = \alpha n_{xx} - \partial_x\big(nV[n]\big) + n(1-n),
\quad V[n] (x,t) = \int_{-\rho}^{\rho} g\big(n^{e}(x+\xi,t)\big)h(\xi) d\xi,
$$

with adhesion law $g(n)=n(\lambda-n)$ and the odd sensing kernel

$$
h(\xi) \;=\; \frac{0.1\,\arctan(0.2\xi)}{\arctan 2}, \qquad |\xi|\le\rho .
$$

On the bounded interval $\Omega=(-L,L)$ the density is prescribed at the healed
value $n\equiv 1$ on an exterior collar of width $\rho$.

| Parameter | Meaning |
|---|---|
| `alpha` | random cell motility (diffusion) |
| `lambda` | cell–cell adhesion strength |
| `rho` | sensing radius |
| `L` | domain half-length |

## What the code computes

- **Whole-line linear stability.** The dispersion relation $\sigma(k;\rho,\lambda)$,
  the largest growth rate $F(\rho;\lambda)$, the critical sensing radius
  $\rho_c^{\infty}(\lambda)$, and the associated wavelength $\ell_c$.
- **Bounded-domain spectrum.** Galerkin discretisation of the linearised
  operator $\mathcal{A}_{L,\rho,\lambda}$ in the Dirichlet sine basis, its
  largest eigenvalue $s_L(\rho;\lambda)$, and the critical eigenfunction.
- **Bounded-domain thresholds.** The critical sensing radius $\rho_c^{(L)}(\lambda)$
  and the linear length threshold $L_{\mathrm{lin}}(\rho,\lambda)$, located by
  bisection.
- **Nonlinear sufficient bound.** The decay rate $\mu_L$ and the sufficient
  half-length $L_{\mathrm{stab}}(\lambda,\rho)$.
- **Time-dependent simulations.** Fourier-continuation spatial derivatives,
  direct evaluation of the adhesion integral, and fourth-order Runge–Kutta
  time integration.



## Reference values

Computed with the default parameters ($\alpha=1$, arctangent kernel):

| Quantity | $\lambda=5$ | $\lambda=6$ |
|---|---|---|
| $\rho_c^{\infty}(\lambda)$ | 15.789 | 9.788 |
| $\rho_c^{(100)}(\lambda)$ | 16.397 | 9.886 |

These can be used to check that a fresh installation reproduces the published
numbers.


## Contact

Ilyas Mustapha — `ilyasm@wustl.edu`
Department of Mathematics, Washington University in St. Louis
