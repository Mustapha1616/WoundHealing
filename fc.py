from mpmath import *

def generate_continuations(alpha,M,W,deg,C,nr=1):
    
    import numpy as np
    
    Cr = C*nr
    
    # save old precision
    old_prec = mp.dps
    
    # update precision
    mp.dps = 500
    
    # convert to arbitrary precision
    alpha = mpf(alpha)
    
    # step size in windowed region
    eta = (1-2*alpha)/(M+1)

    # coefficient function
    def coeff(l,k):

        if l != k:

            if (l%2)!=(k%2):
                return 0
            else:
                return 2*sin((l+k)*pi*alpha)/(l+k) + 2*sin((l-k)*pi*alpha)/(l-k)

        elif k != 0:
            return 2*pi*alpha + (mpf(1)/k)*sin(2*k*pi*alpha)

        else:
            return 4*pi*alpha

    def rhs(k):
        if k == 0:
            return 2*pi*alpha
        else:
            return (-1)**k * 2*sin(k*pi*alpha)/k

    # exclude even nonzero rows
    L = [0] + list(range(1,W+1,2))

    # create the matrix
    A = matrix( [ [coeff(l,k) for k in L] for l in L] )

    # create the right-hand side
    b = matrix( [rhs(k) for k in L ] )

    # solve the system
    a = lu_solve(A,b)

    # evaluate on the sample points
    psi = []
    for j in range(1,M*nr+1):
        x = alpha + j*eta/nr
        p = 0
        for i,k in enumerate(L):
            p += a[i]*cos(k*pi*x)
        psi.append(p)
        
        
    # x values in sample and extension region
    N = deg + 1
    xs = matrix(list(range(N)))
    xe = matrix(list(range(-Cr,0)))/nr

    # initialize polynomials and extensions
    Q = zeros(N,N)
    E = zeros(Cr,N)

    # normalize the constant function
    nrm = sqrt(N)
    Q[:,0] = ones(N,1)/nrm
    E[:,0] = ones(Cr,1)/nrm

    # handle the linear function
    # P = a( x + b )
    b = sum(xs)/N
    Q[:,1] = xs - b
    E[:,1] = xe - b
    nrm = sqrt( (Q[:,1].T*Q[:,1])[0,0] )
    Q[:,1] /= nrm
    E[:,1] /= nrm

    # the remaining polynomials
    # P_{n+1} = a( (x+b)P_n + cP_{n-1} )
    for l in range(2,N):

        # x P_n part
        xP = matrix( [xs[i]*Q[i,l-1] for i in range(N)])

        # b and c
        b = (-xP.T*Q[:,l-1])[0,0]
        c = (-xP.T*Q[:,l-2])[0,0]

        # unnormalized polynomial
        Q[:,l] = matrix( [(xs[i]+b)*Q[i,l-1] for i in range(N)]) + c*Q[:,l-2]
        E[:,l] = matrix( [(xe[i]+b)*E[i,l-1] for i in range(Cr)]) + c*E[:,l-2]

        # normalize
        nrm = sqrt( (Q[:,l].T*Q[:,l])[0,0] )
        Q[:,l] /= nrm
        E[:,l] /= nrm

    # blend the polynomials
    for l in range(N):
        E[:M*nr,l] = matrix( [ E[i,l]*psi[i] for i in range(M*nr)] )

    # convert to double precision
    Q = np.array(Q.tolist(),dtype=float)
    E = np.array(E.tolist(),dtype=float)
    
    #revert precision
    mp.dps = old_prec
    
    # return continuation matrices
    return Q,E

def continue_function(f,Q,E):
    
    import numpy as np
    
    # initialize extended function
    fe = np.zeros(len(f)+len(E))
    fe[:len(f)] = f
    
    # sample sizes
    N = len(Q)
    
    # left and right extension
    le = E.dot(Q.T.dot(f[:N]))
    re = E.dot(Q.T.dot(f[-1:-N-1:-1]))
    
    # combine the extension
    fe[len(f):] = le + re[::-1]
    
    return fe

def blend_to_zero(f,Q,E):
    
    import numpy as np
    
    # initialize extended function
    fe = np.zeros(len(f)+len(E))
    fe[:len(f)] = f
    
    # sample sizes
    N = len(Q)
    
    # right extension
    re = E.dot(Q.T.dot(f[-1:-N-1:-1]))
    
    # add extension to function
    fe[len(f):] = re[::-1]
    
    return fe