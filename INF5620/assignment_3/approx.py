import sympy as sym

def least_square(f, psi, Omega):
    '''
    Using the least square method to find a solution to Poisson's
    equation when we assume that we have a orthogonal set of
    basis functions. Poisson: -d^2u/dx^2=f(x)
    f is supposed to be a sympy function
    psi is supposed to be a list of sympy functions
    Omega is supposed to be a 2-list containing the boundaries
    '''
    N = len(psi) - 1
    c = [0]*(N+1)	# plain list to hold symbolic expressions
    u = 0
    x = sym.Symbol('x')
    for i in range(N+1):
        ddpsi = sym.diff(sym.diff(psi[i], x), x)
        A = sym.integrate(ddpsi*ddpsi, (x, Omega[0], Omega[1]))
        b = -sym.integrate(f*ddpsi, (x, Omega[0], Omega[1]))
        c[i] = b/A
        u   += c[i]*psi[i]
    return u, c

def galerkin(f, psi, Omega):
    '''
    Using the Galerkin method to find a solution to Poisson's
    equation when we assume that we have a orthogonal set of
    basis functions. Poisson: -d^2u/dx^2=f(x)
    f is supposed to be a sympy function
    psi is supposed to be a list of sympy functions
    Omega is supposed to be a 2-list containing the boundaries
    '''
    N = len(psi) - 1
    c = [0]*(N+1)	# plain list to hold symbolic expressions
    u = 0
    x = sym.Symbol('x')
    for i in range(N+1):
        ddpsi = sym.diff(sym.diff(psi[i], x), x)
        A = sym.integrate(psi[i]*ddpsi, (x, Omega[0], Omega[1]))
        b = -sym.integrate(f*psi[i],  (x, Omega[0], Omega[1]))
        c[i] = b/A
        u   += c[i]*psi[i]
    return u, c
