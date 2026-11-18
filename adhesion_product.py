"""
Adhesion product integration, independent of the custom FC modules.

The nodal values of g(n), including the prescribed collar, are interpolated
linearly. Each interpolation cell is intersected with [-rho,rho] and its
linear basis functions are integrated against the kernel by Gauss quadrature.
The resulting convolution weights are computed once per grid.
"""
import numpy as np
from numpy.fft import rfft, irfft
from scipy.fft import next_fast_len
from scipy.special import roots_legendre


def product_integration_weights(h, rho, kernel_fun, quadrature_order=8):
    """Return weights W[j] for V_i = sum_j W[j] * g(n_{i+j}).

    W[j] = integral_{-rho}^{rho} h_kernel(z) B_j(z) dz, where B_j
    is the nodal piecewise-linear hat function centered at j*h.
    The returned array is ordered by offsets -m,...,m, m=ceil(rho/h).
    Real odd kernels are required, as in the wound-healing model.

    A nonzero weight at m*h>rho is correct: that node's hat function
    overlaps the sensing interval. Never zero that weight afterward.
    """
    h, rho = float(h), float(rho)
    if not np.isfinite([h, rho]).all() or h <= 0 or rho < 0:
        raise ValueError('Require finite h>0 and rho>=0.')
    if int(quadrature_order) != quadrature_order or quadrature_order < 1:
        raise ValueError('quadrature_order must be a positive integer.')
    
    m = int(np.ceil(rho/h))

    if m == 0:
        return np.zeros(1)
    
    nodes, gauss_weights = roots_legendre(int(quadrature_order))
    positive = np.zeros(m+1)

    for j in range(m):
        left = j*h
        right = min((j+1)*h, rho)
        if right <= left:
            continue
        z = left + (nodes+1)*(right-left)/2
        hp = np.asarray(kernel_fun(z, rho))
        hn = np.asarray(kernel_fun(-z, rho))
        if np.iscomplexobj(hp) or np.iscomplexobj(hn):
            raise ValueError('The adhesion kernel must be real.')
        hp = np.broadcast_to(hp, z.shape).astype(float)
        hn = np.broadcast_to(hn, z.shape).astype(float)
        if not np.isfinite(hp).all() or not np.isfinite(hn).all():
            raise ValueError('The adhesion kernel returned nonfinite values.')
        if not np.allclose(hn, -hp, atol=1e-13, rtol=1e-11):
            raise ValueError('This formulation requires an odd adhesion kernel.')
        theta = (z-left)/h
        weighted_kernel = gauss_weights*(right-left)/2*hp
        positive[j] += weighted_kernel @ (1-theta)
        positive[j+1] += weighted_kernel @ theta
    # The center hat is even and the kernel is odd, so its weight is zero.
    return np.concatenate((-positive[:0:-1], [0.0], positive[1:]))


def build_product_adhesion_force(grid, lam, kernel_fun, adhesion_law,
                                quadrature_order=8):
    """Build the velocity evaluator for the user's [collar|physical|collar] grid.

    The FFT is a padded linear convolution with the x+z orientation.
    Subtracting g(1) preserves n=1 exactly and is valid because the integral
    of an odd kernel on [-rho,rho] is zero. Prescribed collar values are
    taken from the input state; Fourier-continuation values are never used.
    """
    m = grid.extra
    weights = product_integration_weights(
        grid.h, grid.rho, kernel_fun, quadrature_order)
    
    if len(weights) != 2*m+1:
        raise ValueError('The collar size must equal ceil(rho/h).')
    
    fft_size = next_fast_len(grid.pN + len(weights) - 1, real=True)
    kernel_hat = rfft(weights[::-1], n=fft_size)
    physical_output = slice(2*m, 2*m+grid.N)
    reference = np.asarray(adhesion_law(np.ones(1), lam))

    if (reference.size != 1 or np.iscomplexobj(reference)
            or not np.isfinite(reference).all()):
        raise ValueError('The adhesion law must return a finite real value at n=1.')
    g_one = float(reference.ravel()[0])

    def adhesion_force(u):
        u = np.asarray(u)
        if u.shape != (grid.pN,):
            raise ValueError('Pass the full physical-and-collar state.')
        g = np.asarray(adhesion_law(u, lam))
        if np.iscomplexobj(g):
            raise ValueError('The adhesion law must return real values.')
        q = np.broadcast_to(g, u.shape).astype(float) - g_one
        if not np.isfinite(q).all():
            raise FloatingPointError('Nonfinite adhesion-law values in the stage state.')
        convolution = irfft(rfft(q, n=fft_size)*kernel_hat, n=fft_size)
        return convolution[physical_output]

    return adhesion_force
