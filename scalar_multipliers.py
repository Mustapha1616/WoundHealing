from mpmath import *

def cdelta(n,beta,delta):
    """
    Computes the c_delta coefficient.

    Args:
        n (int): The spatial dimension.
        beta (float):  The peridynamic exponent.
        delta (float): The peridynamic radius.

    Returns:
        c_delta
    """

    # convert to arbitrary precision
    n     = mpf(n)
    beta  = mpf(beta)
    delta = mpf(delta)

    # compute
    c_delta = ( 2*(n+2-beta)*gamma(n/2+1) )/( pi**(n/2)*delta**(n+2-beta) )

    return float(c_delta)

def multiplier(n,beta,delta,nu,nu_is_squared=False):
    """
    Computes Fourier multipliers of the L-delta operator.

    Args:
        n (int): The spatial dimension.
        beta (float): The peridynamic exponent.
        delta (float): The peridynamic radius.
        nu (array(float)): Norms of the Fourier modes to apply the operator to.

    Returns:
        The multipliers at the given Fourier modes.
    """

    import numpy as np

    # convert to arbitrary precision
    n     = mpf(n)
    delta = mpf(delta)

    # square nu if needed
    if nu_is_squared:
        nu2 = mpf(1)*nu
    else:
        nu2 = mpf(1)*(nu**2)


    # function to evaluate for each number
    if beta == -np.inf:

        f = lambda x: hyp1f2(1,2,(n+2)/2,-x*delta**2/4)

    else:

        beta  = mpf(beta)
        f = lambda x: hyp2f3(1,(n+2-beta)/2,2,(n+2)/2,(n+4-beta)/2,-x*delta**2/4)

    # apply it to the nu values
    f_nu2 = np.array(list(map(f, nu2)))
    m = -nu2*f_nu2

    # convert back to double precision
    return m.astype(float)


def L_delta_f_1(n,l,beta,delta):
    """
    Computes the 1D L_delta operator in Fourier space.

    Args:
        n:     the number of grid points
        l:     the length of the interval
        beta:  the peridynamic parameter
        delta: the peridynamic radius

    Returns:
        The array of Fourier multipliers
    """

    from fourier import fft_modes
    import numpy as np

    # initialize multiplier array
    m = np.zeros(n)

    # positive nu values
    nu = (2*np.pi/l) * np.arange(1,int(n/2)+1)

    # evaluate the multipliers
    m[1:int(n/2)+1] = multiplier(1,beta,delta,nu)

    # fill in multipliers for negative modes
    if n%2 == 0:
        m[-int(n/2)+1:] = m[int(n/2)-1:0:-1]
    else:
        m[-int((n-1)/2):] = m[int((n-1)/2):0:-1]

    return m

def L_f_1(n,l):
    """
    Computes the 1D Laplacian operator in Fourier space.

    Args:
        n:        iterable of number of grid points
        l:        iterable of interval lengths

    Returns:
        The array of Fourier multipliers
    """

    from fourier import fft_modes
    import numpy as np

    # initialize multiplier array
    m = np.zeros(n)

    # positive nu values
    nu = (2*np.pi/l) * np.arange(1,int(n/2)+1)

    # evaluate the multipliers
    m[1:int(n/2)+1] = -nu**2

    # fill in multipliers for negative modes
    if n%2 == 0:
        m[-int(n/2)+1:] = m[int(n/2)-1:0:-1]
    else:
        m[-int((n-1)/2):] = m[int((n-1)/2):0:-1]

    return m

def L_f_2(n,l):
    """
    Computes the 2D Laplacian operator in Fourier space.

    Args:
        n:        iterable of number of grid points
        l:        iterable of interval lengths

    Returns:
        The array of Fourier multipliers
    """
    import numpy as np
    # initialize multiplier array
    m = np.zeros((n[0],n[1]))
    
    # positive nu values
    nu_x = (2*np.pi/l[0]) * np.arange(int(n[0]/2)+1)
    nu_y = (2*np.pi/l[1]) * np.arange(int(n[1]/2)+1)
    
    # form grid
    nu_X,nu_Y = np.meshgrid(nu_x,nu_y,indexing='ij')
    
    # generate norm-squared
    nu2 = np.ravel(nu_X**2 + nu_Y**2)
    mul = -nu2.reshape(nu_X.shape)
    
    # fill in the multipliers for positive modes
    m[:int(n[0]/2)+1,:int(n[1]/2)+1] = mul
    
    # fill remaining multipliers
    if n[0]%2 == 0:
        m[-int(n[0]/2)+1:,:] = m[int(n[0]/2)-1:0:-1,:]
    else:
        m[-int((n[0]-1)/2):,:] = m[int((n[0]-1)/2):0:-1,:]
        
    if n[1]%2 == 0:
        m[:,-int(n[1]/2)+1:] = m[:,int(n[1]/2)-1:0:-1]
    else:
        m[:,-int((n[1]-1)/2):] = m[:,int((n[1]-1)/2):0:-1]
    
    return m

# multipliers at the given Fourier nodes for 2D
def multiplier2(n,beta,delta,nu_x,nu_y):
    """
    Computes Fourier multipliers of the L-delta operator.

    Args:
        n (int): The spatial dimension.
        beta (float): The peridynamic exponent.
        delta (float): The peridynamic radius.
        nu (array(float)): Norms of the Fourier modes to apply the operator to.

    Returns:
        The multipliers at the given Fourier modes.
    """

    import numpy as np

    # convert to arbitrary precision
    n     = mpf(n)
    delta = mpf(delta)
    nu2   = mpf(1) * (nu_x**2+nu_y**2)

    if beta == -np.inf:
        f = lambda x: hyp1f2(1,2,(n+2)/2,-x*delta**2/4)
    else:

        beta  = mpf(beta)
        f = lambda x: hyp2f3(1,(n+2-beta)/2,2,(n+2)/2,(n+4-beta)/2,-x*delta**2/4)

    f_nu2 = np.zeros(nu_x.shape)

    # apply it to the nu values
    for i in range(len(nu_x)):
        f_nu2[i] = np.array(list(map(f, nu2[i])))

    m = -nu2*f_nu2

    return m.astype(float)

"""def L_delta_f_2(n,l,beta,delta):
    '''
    Computes the 2D L_delta operator in Fourier space.

    Args:
        n:     iterable of number of grid points
        l:     iterable of interval lengths
        beta:  the peridynamic parameter
        delta: the peridynamic radius

    Returns:
        The 2D array of Fourier multipliers
    '''

    import numpy as np
    
    nx, ny = n[0], n[1]
    lx, ly = l[0], l[1]

    # initialize multiplier array
    m = np.zeros((nx,ny))

    # positive nu values
    nu_x = (2*np.pi/lx) * np.arange(int(nx/2)+1)
    nu_y = (2*np.pi/ly) * np.arange(int(ny/2)+1)
    nu_x_grid, nu_y_grid = np.meshgrid(nu_x, nu_y, indexing='ij')

    # multipliers at the positive nu values
    mult = multiplier2(2,beta,delta,nu_x_grid,nu_y_grid)

    m[:int(nx/2)+1, :int(ny/2)+1] = mult
    
    # fill remaining multipliers
    if nx%2 == 0:
        m[-int(nx/2)+1:,:] = m[int(nx/2)-1:0:-1,:]
    else:
        m[-int((nx-1)/2):,:] = m[int((nx-1)/2):0:-1,:]

    if ny%2 == 0:
        m[:,-int(ny/2)+1:] = m[:,int(ny/2)-1:0:-1]
    else:
        m[:,-int((ny-1)/2):] = m[:,int((ny-1)/2):0:-1]

    '''# fill in multipliers for negative modes
    if nx%2 == 0 and ny%2 == 0:
        m[0][-int(ny/2)+1:]           = mult[0][int(ny/2)-1:0:-1]
        m[:,0][-int(nx/2)+1:]         = mult[:,0][int(nx/2)-1:0:-1]
        m[1:int(nx/2)+1,int(ny/2)+1:] = mult[1:,int(ny/2)-1:0:-1]
        m[int(nx/2)+1:,1:int(ny/2)+1] = mult[int(nx/2)-1:0:-1,1:]
        m[int(nx/2)+1:,int(ny/2)+1:]  = mult[int(nx/2)-1:0:-1,int(ny/2)-1:0:-1]

    elif nx%2 == 0 and ny%2 == 1:
        m[0][-int((ny-1)/2):]         = mult[0][int((ny-1)/2):0:-1]
        m[:,0][-int(nx/2)+1:]         = mult[:,0][int(nx/2)-1:0:-1]
        m[1:int(nx/2)+1,int(ny/2)+1:] = mult[1:,int((ny-1)/2):0:-1]
        m[int(nx/2)+1:,1:int(ny/2)+1] = mult[int(nx/2)-1:0:-1,1:]
        m[int(nx/2)+1:,int(ny/2)+1:]  = mult[int(nx/2)-1:0:-1,int((ny-1)/2):0:-1]

    elif nx%2 == 1 and ny%2 == 0:
        m[0][-int(ny/2)+1:]           = mult[0][int(ny/2)-1:0:-1]
        m[:,0][-int((nx-1)/2):]       = mult[:,0][int((nx-1)/2):0:-1]
        m[1:int(nx/2)+1,int(ny/2)+1:] = mult[1:,int(ny/2)-1:0:-1]
        m[int(nx/2)+1:,1:int(ny/2)+1] = mult[int((nx-1)/2):0:-1,1:]
        m[int(nx/2)+1:,int(ny/2)+1:]  = mult[int((nx-1)/2):0:-1,int(ny/2)-1:0:-1]

    else:
        m[0][-int((ny-1)/2):]         = mult[0][int((ny-1)/2):0:-1]
        m[:,0][-int((nx-1)/2):]       = mult[:,0][int((nx-1)/2):0:-1]
        m[1:int(nx/2)+1,int(ny/2)+1:] = mult[1:,int((ny-1)/2):0:-1]
        m[int(nx/2)+1:,1:int(ny/2)+1] = mult[int((nx-1)/2):0:-1,1:]
        m[int(nx/2)+1:,int(ny/2)+1:]  = mult[int((nx-1)/2):0:-1,int((ny-1)/2):0:-1]'''

    return m"""
    
def L_delta_f_2(n,l,beta,delta):
    """
    Computes the 2D L_delta operator in Fourier space.

    Args:
        n:        iterable of number of grid points
        l:        iterable of interval lengths
        beta:     the peridynamic parameter
        delta:    the peridynamic radius

    Returns:
        The array of Fourier multipliers
    """

    from fourier import fft_modes
    import numpy as np

    # initialize multiplier array
    m = np.zeros((n[0],n[1]))

    # positive nu values
    nu_x = (2*np.pi/l[0]) * np.arange(int(n[0]/2)+1)
    nu_y = (2*np.pi/l[1]) * np.arange(int(n[1]/2)+1)

    # form grid
    nu_X,nu_Y = np.meshgrid(nu_x,nu_y,indexing='ij')

    # generate norm-squared
    nu2 = np.ravel(nu_X**2 + nu_Y**2)

    # compute the multipliers
    mul = multiplier(2,beta,delta,nu2,nu_is_squared=True).reshape(nu_X.shape)

    # fill in the multipliers for positive modes
    m[:int(n[0]/2)+1,:int(n[1]/2)+1] = mul

    # fill remaining multipliers
    if n[0]%2 == 0:
        m[-int(n[0]/2)+1:,:] = m[int(n[0]/2)-1:0:-1,:]
    else:
        m[-int((n[0]-1)/2):,:] = m[int((n[0]-1)/2):0:-1,:]

    if n[1]%2 == 0:
        m[:,-int(n[1]/2)+1:] = m[:,int(n[1]/2)-1:0:-1]
    else:
        m[:,-int((n[1]-1)/2):] = m[:,int((n[1]-1)/2):0:-1]

    return m