from approx import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn

def sine(x, N):
    '''
    Function space with only 'odd' sine functions
    '''
    return [sym.sin((2*i+1)*(sym.pi/2)*x) for i in range(N+1)]
    
    
def all_sines(x, N):
    '''
    Function space with all sine functions
    '''
    return [sym.sin((i+1)*(sym.pi/2)*x) for i in range(N+1)]


def coeffs(x, f, Omega, method, N = 10):
    '''
    Analyse the coefficients using least square method
    and "odd" sines
    N+1 is the number of sines
    '''
    psi = sine(x, N)
    if method == 'lsm':
        u, c = least_square(f, psi, Omega)
    elif method == 'gm':
        u, c = galerkin(f, psi, Omega)
    else: 
        raise NameError("Only Least Square Method (lsm) and \
                         Galerkin Method (gm) are accepted")
    print '\nCoefficients'
    for C in c:
        print float(C)
    print '\nCoefficient ratios'
    for i in range(len(c) - 1):
        print c[i+1]/c[i]


def midpoint_error(x, f, u_e, Omega, method, x_meas = 1, N = 0):
    '''
    Finds the error at the midpoint when we only
    have one sine function. LSM and odd sines used
    x_meas is the measure point (x_meas = 1 is the midpoint)
    '''
    psi = sine(x, N)
                             
    if method == 'lsm':
        u, c = least_square(f, psi, Omega)
    elif method == 'gm':
        u, c = galerkin(f, psi, Omega)
    else: 
        raise NameError("Only Least Square Method (lsm) and \
                         Galerkin Method (gm) are accepted")
    print '\nError at the midpoint'
    print 'Exact:  ', u_e.subs(x, x_meas)
    print 'Approx: ', u.subs(x, x_meas)
    print 'Error:  ', abs(float(u_e.subs(x, 1))-float(u.subs(x, 1)))


def viz(x, f, u_e, Omega, method, Psi, N_list, M):
    '''
    This function plots the approximations for a list of
    N-values (N_list) in the same plot. Take method 
    ['lsm', 'gm'] and psi ['odd', 'all'] as arguments. 
    M is the number of points along the cable
    '''
    x_list = np.linspace(Omega[0], Omega[1], M)
    for N in N_list:
        if Psi == 'odd':
            psi = sine(x, N)
        elif Psi == 'all':
            psi = all_sines(x, N)
        else:
            raise NameError("Only 'odd' sines ('odd') or \
                             all sines ('all') is accepted")
                             
        if method == 'lsm':
            u, c = least_square(f, psi, Omega)
        elif method == 'gm':
            u, c = galerkin(f, psi, Omega)
        else: 
            raise NameError("Only Least Square Method (lsm) or \
                             Galerkin Method (gm) is accepted")
        v = np.empty(M)
        for i in range(M):
            v[i] = u.subs(x, x_list[i])
        plt.plot(x_list, v, label='N = %d'%N)
    w = np.empty(M)
    for i in range(M):
        w[i] = u_e.subs(x, x_list[i])
    
    plt.plot(x_list, w, '--r', label='Exact')
    plt.legend()
    plt.title('Exact and approx cable for %s and %s'%(str(Omega), method))
    plt.xlabel('x-direction')
    plt.ylabel('y-direction')
    plt.savefig('%s_%s_%d.png'%(method, Psi, Omega[1]))
    if Omega[1] == 2:
        plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    pass
    
x   = sym.Symbol('x')
u_e = 0.5*x**2 - x        # Exact function
f   = -1
Omega = [0, 1]

# --- Least square method (lsm)---
print '\n === LEAST SQUARE METHOD ==='
coeffs(x, f, Omega, method='lsm')
midpoint_error(x, f, u_e, Omega, method = 'lsm')
viz(x, f, u_e, Omega, 'lsm', 'odd', [0, 1, 20], 100)

# --- Galerkin method (gm) ---
print '\n === GALERKIN METHOD ==='
coeffs(x, f, Omega, method='gm')
midpoint_error(x, f, u_e, Omega, method = 'gm')
viz(x, f, u_e, Omega, 'gm', 'odd', [0, 1, 20], 100)
viz(x, f, u_e, Omega, 'gm', 'all', [0, 1, 20], 100)
viz(x, f, u_e, [0, 2], 'gm', 'all', [0, 1, 20], 100)
