def fft_modes(N):
    from numpy.fft import ifftshift
    from numpy import arange
    return ifftshift( arange(-int(N/2),int((N-1)/2)+1) )

def zero_pad(f,m):
    """
    Interpolates the point values given by f to m points.
    """

    import numpy as np
    from numpy.fft import fft, ifft

    # length of original and interpolated functions
    n = len(f)
    assert(m>n)

    # take the FFT
    ft = fft(f)

    # get Fourier modes
    k = fft_modes(n)

    # perform the padding
    gt = (float(m)/n)*np.concatenate( (ft[k>=0], np.zeros(m-n), ft[k<0]) )

    # invert the fft
    return ifft(gt).real