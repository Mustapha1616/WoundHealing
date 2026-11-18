# Nonlocal Wound Healing on Bounded Domains

This repository accompanies the manuscript *Stability analysis and critical thresholds in a nonlocal advection diffusion model of wound healing on bounded domains* by Ilyas Mustapha and Mikayla Gregory. It contains the Python code used to compute stability thresholds for the healed state and to simulate wound recovery.

## The model

On the interval $\Omega=(-L,L)$, the cell density $n(x,t)$ satisfies

$$
n_t = \alpha n_{xx} - \partial_x\!\left(nV[n]\right) + n(1-n),
\qquad
V[n](x,t) = \int_{-\rho}^{\rho} g\!\left(n^e(x+\xi,t)\right)h(\xi)\,d\xi.
$$

The endpoint values are $n(\pm L,t)=1$. In the adhesion integral, $n^e$ equals $n$ inside the interval and equals $1$ on the exterior collar of width $\rho$ at each end. The constant density $n\equiv1$ is the healed equilibrium. The adhesion law and the odd sensing kernel are

$$
g(n)=n(\lambda-n), \qquad
h(\xi)=\frac{0.1\arctan(0.2\xi)}{\arctan(2)}, \quad |\xi|\leq\rho.
$$

| Parameter | Meaning |
| --- | --- |
| $\alpha$ (`alpha`) | Cell diffusion coefficient; the time-dependent solver uses $\alpha=1$. |
| $\lambda$ (`lam`) | Cell–cell adhesion parameter. |
| $\rho$ (`rho`) | Sensing radius. |
| $L$ (`L`) | Domain half-length; the full interval has length $2L$. |


## What the code computes

- **Whole-line stability.** Maximizing the dispersion relation $\sigma(k;\rho,\lambda)$ over wavenumbers gives the largest linear growth rate $F(\rho;\lambda)$. At the critical sensing radius $\rho_c^\infty(\lambda)$, the code also computes the critical wavenumber $k_c$ and wavelength $2\pi/k_c$.
- **Bounded-domain stability.** A Galerkin method in the Dirichlet sine basis computes the largest eigenvalue $s_L(\rho;\lambda)$ of the linearized operator and its associated eigenfunction. The critical radius $\rho_c^{(L)}(\lambda)$ marks the onset of positive linear growth as $\rho$ varies. The critical half-length $L_c(\rho,\lambda)$ marks the loss of strict exponential stability as $L$ increases.
- **Sufficient condition for healing.** For $L<L_{\mathrm{stab}}(\lambda,\rho)$, the analysis gives a positive decay rate $\mu_L$ for any sufficiently regular solution that exists globally and stays in the assumed density interval. Healing can also occur for larger $L$.

The time-dependent solver uses Fourier continuation for diffusion and classical fourth-order Runge–Kutta (RK4) for time stepping. It computes the adhesion velocity by piecewise-linear product integration, using eight-point Gauss–Legendre quadrature for the weights and a zero-padded FFT for the convolution. For the derivative of the adhesion flux $nV[n]$, the recovery experiment (Figure 4) uses a second-order centered difference; the modal and small-wound experiments (Figures 2 and 5) use Fourier continuation.

## Reproducing the results

The scripts use Python 3, NumPy, SciPy, and Matplotlib. From the repository root, install the packages in your Python environment and generate the initial wound profile:

```bash
python3 -m pip install numpy scipy matplotlib
python3 plot_initial_wound.py --no-show
```

This writes `solution_storage/initial_wound.pdf`. To generate the spectral threshold figure, run `python3 wound_stability.py --no-show`. Neither figure requires the time-dependent solver.

Figures 2, 4, and 5 also require `wound_healing.py`, `adhesion_product.py`, and the project's `fc` and `scalar_multipliers` modules. Keep these modules alongside the figure scripts or add their directory to `PYTHONPATH` so Python can import them.

Run each figure script with `python3` followed by its filename. The first five entries below reproduce Figures 1–5, in order, and write their PDFs to `solution_storage/` in the current working directory.

| Script or function | Produces |
| --- | --- |
| `wound_stability.py` | Growth-rate curves for the whole line and bounded intervals, with the critical sensing radii $\rho_c^\infty$, $\rho_c^{(L)}$, and the critical half-length $L_c$. |
| `run_modal_validation.py` | Normalized amplitudes of the leading mode compared with $e^{s_Lt}$ for $L=100$ and $\lambda=5$: decay at $\rho=16$ and growth at $\rho=18$. |
| `plot_initial_wound.py` | The initial wound profile $n_0(x)=1-0.75\exp[-(0.1x)^6]$ and the healthy density $n=1$. |
| `run_wound_recovery.py` | Density profiles, errors relative to the healed state, and observed healing times (when the numerical criterion is met) for $L=30,100,200$. |
| `run_small_wound_bounds.py` | The sufficient rate $\mu_L$ versus $L/L_{\mathrm{stab}}$ and the simulated errors $E_L(t)/E_L(0)$, with the conditional upper bound $e^{-\mu_Lt}$ when $\mu_L>0$. |
| `wound_stability.py`: `compute_s_L()` | The leading eigenvalue $s_L(\rho;\lambda)$ and the sine coefficients of its normalized eigenfunction (Algorithm 1). |
| `wound_stability.py`: `critical_parameter()` | A threshold located by bisection at a resolved sign change, together with its critical wavenumber on the whole line or its critical eigenfunction on the bounded interval (Algorithm 2). |

## Reference values

For $\alpha=1$ and the arctangent kernel above, the critical sensing radii reported in the manuscript are shown below to three decimal places.

| Quantity | $\lambda=5$ | $\lambda=6$ |
| --- | ---: | ---: |
| Whole line: $\rho_c^\infty(\lambda)$ | 15.789 | 9.788 |
| Bounded interval, $L=100$: $\rho_c^{(100)}(\lambda)$ | 16.397 | 9.886 |

To check the bounded result for $\lambda=5$ from Python:

```python
from wound_stability import critical_rho_bounded

result = critical_rho_bounded(L=100, lam=5)
print(round(result['value'], 3))  # 16.397
```


