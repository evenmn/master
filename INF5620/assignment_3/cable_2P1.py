from fe_approx import *
'''
x = sym.Symbol('x')
approximate(0.5*x**2 - x, symbolic=False, d=1, N_e=2, 
            Omega=[0, 2], filename='fe_p1_x2_2e')
'''
def mesh_uniform(N_e, Omega=[0, 1]):
    '''
    Creates the mesh grid based on the number of
    elements N_e and the domain Omega
    '''
    nodes = np.linspace(Omega[0], Omega[1], N_e + 1).tolist()
    elements = [[e + i for i in range(2)] \
                for e in range(N_e)]
    return nodes, elements

def coeffs(f, N_e=2, Omega=[0, 1]):
    '''
    Gives the coefficients using finite element
    method
    '''
    h = (Omega[1] - Omega[0])/float(N_e)
    A = np.zeros((N_e+1, N_e+1))
    b = np.full((N_e+1), f*h**2)
    for i in range(N_e+1):
        A[i,i] = 2
    for i in range(N_e):
        A[i+1,i] = -1
        A[i,i+1] = -1
    
    c = np.linalg.solve(A, b)
    return c

def basis(N_e=2, Omega=[0, 1]):
    '''
    Creates linear (d=1) basis function based
    on the number of elements N_e. Assuming 
    two nodes per element, which makes phi_i
    1 at node i and 0 at the neighbor nodes
    '''
    nodes, elements = mesh_uniform(N_e, Omega)
    b = (nodes[0] - nodes[1])**(-1)
    phi = []
    phi.append(lambda x: 1 - b*x)
    for i in range(1, N_e):
        p = lambda x: (1 - i) + b*x if x < nodes[i] else (1 + i) - b*x
        phi.append(p)
    phi.append(lambda x: (1 + N_e) - b*x)
    return phi
    
def initialize(c, x0 = 0):
    '''Set the first coefficient to the initial condition x0'''
    diff = x0 - c[0]
    c += diff
    return c
    
def approximate(f, N_e, Omega):
    '''
    Calls the coefficient function, initialize with the
    initialize function and make beautiful plots
    '''
    c = coeffs(f, N_e, Omega)
    c = initialize(c)
    nodes, elements = mesh_uniform(N_e, Omega)
    plt.plot(nodes, c, label='N_e = %d'%N_e)
    
if __name__ == '__main__':
    pass

Omega = [0, 2]          # Domain
f = -1                  # Source function in Poisson equation '-u''=f'

approximate(f, 2, Omega)
approximate(f, 4, Omega)
approximate(f, 8, Omega)

# Show plots
X = np.linspace(Omega[0], Omega[1], 100)
plt.plot(X,[0.5*x**2 - x for x in X], '--', label='Exact')
plt.legend(loc='best')
plt.title('Solving $d^2u/dx^2=1$ using finite element method')
plt.xlabel('x-direction')
plt.ylabel('y-direction')
plt.savefig('finite_elements.png')
plt.axis('equal')
plt.show()
