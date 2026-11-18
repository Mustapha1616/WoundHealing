# Nonlocal Wound Healing on Bounded Domains
This is a code repository for the manuscript _Stability analysis 
and critical thresholds in a nonlocal advection diffusion model 
of wound healing on bounded domains_ by Ilyas Mustapha and Mikayla Gregory.

The repository contains the routines used to compute the stability 
thresholds and to run the wound healing simulations reported in the paper.

## The model

We study the cell density $n(x,t)$ governed by

$$
n_t = \alpha n_{xx} - \partial_x\big(nV[n]\big) + n(1-n),
\quad V[n] (x,t) = \int_{-\rho}^{\rho} g\big(n(x+\xi,t)\big)h(\xi) d\xi,
$$

with adhesion law $g(n)=n(\lambda-n)$ and the odd sensing kernel

$$
h(\xi) = \frac{0.1\arctan(0.2\xi)}{\arctan 2}, \qquad |\xi|\le\rho .
$$

On the bounded interval $\Omega=(-L,L)$, the density is prescribed at the healed value $n\equiv 1$ on an exterior collar of width $\rho$.

| Parameter | Meaning |
|---|---|
| `alpha` | random cell motility (diffusion) |
| `lambda` | cell-cell adhesion strength |
| `rho` | sensing radius |
| `L` | domain half-length |

## What the code computes

- **Whole-line linear stability.** The dispersion relation $\sigma(k;\rho,\lambda)$,
  the largest growth rate $F(\rho;\lambda)$, the critical sensing radius
  $\rho_c^{\infty}(\lambda)$, and the associated wavelength $\ell_c$.
- **Bounded domain spectrum.** Galerkin discretisation of the linearised
  operator $\mathcal{A}_{L,\rho,\lambda}$ in the Dirichlet sine basis, its
  largest eigenvalue $s_L(\rho;\lambda)$, and the critical eigenfunction.
- **Bounded domain thresholds.** The critical sensing radius $\rho_c^{(L)}(\lambda)$
  and the linear length threshold $L_c(\rho,\lambda)$, located by
  bisection.
- **Nonlinear sufficient bound.** The decay rate $\mu_L$ and the sufficient
  half-length $L_{\mathrm{stab}}(\lambda,\rho)$.
- **Time-dependent simulations.** Fourier continuation spatial derivatives,
  direct evaluation of the adhesion integral, and fourth-order Runge-Kutta
  time integration.


## Reproducing the results

The first five entries reproduce Figures 1–5, in order, and save the PDFs in `solution_storage/`. The last two entries identify the functions used for the stability calculations.

| Script or function | Produces |
| --- | --- |
| `wound_stability.py` | Growth rate curves for the whole line and bounded intervals, showing the critical sensing radii $\rho_c^\infty$ and $\rho_c^{(L)}$ and the critical domain half-length $L_c$. |
| `run_modal_validation.py` | Normalized amplitudes of the leading mode compared with $e^{s_L t}$: decay at $\rho=16$ and growth at $\rho=18$ for $L=100$ and $\lambda=5$. |
| `plot_initial_wound.py` | The initial wound profile $n_0(x)=1-0.75\exp[-(0.1x)^6]$ together with the healthy density $n=1$. |
| `run_wound_recovery.py` | Wound recovery profiles and error histories for $L=30,100,200$, with observed healing times when the numerical healing criterion is met. |
| `run_small_wound_bounds.py` | The sufficient decay rate $\mu_L$ versus $L/L_{\mathrm{stab}}$, and the normalized small wound errors $E_L(t)/E_L(0)$ compared with the conditional bound $e^{-\mu_L t}$ when $\mu_L>0$. |
| `wound_stability.py`: `compute_s_L()` | The largest eigenvalue $s_L(\rho;\lambda)$ of the bounded linearized operator and the sine coefficients of its normalized eigenfunction (Algorithm 1). |
| `wound_stability.py`: `critical_parameter()` | A critical sensing radius or domain half-length located by bisection, together with the critical wavenumber on the whole line or the critical eigenfunction on the bounded interval (Algorithm 2). |



## Reference values

Computed with the default parameters ($\alpha=1$, arctangent kernel):

| Quantity | $\lambda=5$ | $\lambda=6$ |
|---|---|---|
| $\rho_c^{\infty}(\lambda)$ | 15.789 | 9.788 |
| $\rho_c^{(100)}(\lambda)$ | 16.397 | 9.886 |

These can be used to check that a fresh installation reproduces the published
numbers.

