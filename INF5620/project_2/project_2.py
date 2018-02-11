from fenics import *
import numpy as np
from plot_exact_solutions import *

def solve_diffusion(I, f, alpha, rho, T, Nx, Ny, Nz, Nt, 
                    elements = 1, dimension = 3, Plot = False):
    '''
    Exercise c
    This function solves the diffusion equation
    
        rho*(du/dt)=grad*(alpha*grad(u)) + f(u)
    
    given the value of rho and the functions alpha and f.
    We need a (Diriclet) boundary function I to find an
    initial function u_n
    '''
    # Define constants
    dt = float(T)/Nt                # Time step
    C = dt/(2*rho)                  # To reduce the number of FLOPS

    # Create mesh and define function space
    mesh = [UnitIntervalMesh(Nx), UnitSquareMesh(Nx, Ny), UnitCubeMesh(Nx, Ny, Nz)]
    V = FunctionSpace(mesh[dimension - 1], 'P', elements)   # Function space
    u_n = interpolate(I, V)                               # Define initial value

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = u*v*dx + C*alpha*dot(grad(u), grad(v))*dx
    L = u_n*v*dx - C*alpha*dot(grad(u_n), grad(v))*dx + 2*C*f*v*dx
    
    u = Function(V)                     # This gonna be the u approximation
    t_list = np.linspace(dt, T, Nt)     # Time-stepping

    for t in t_list:
        # Update current time
        f.t = t
        alpha.u = u_n

        solve(a == L, u)    # Solve system
        u_n.assign(u)       # Update previous solution
    

    # Plot and hold plot
    if Plot:
        plot(u, interactive=True)
    return u, V


def test_constant_solution(T, Nx, Ny, Nz, Nt, tolerance = 1e-9, Plot = False):
    '''
    Exercise d
    With a constant solution u(x,t) = C we expect rho and
    alpha to be constants and f to be constant zero
    independent of the values of C, rho and alpha
    '''
    rho = Constant(1.0)
    alpha = Constant(1.0)
    f = Constant(0.0)
    u_D = Constant(1.0)
    
    print '\nConstant solution'
    for Ne in [1,2]:
        for dim in [1,2,3]:
            '''Find the solution for P1 and P2 elements studying 1D, 2D and 3D'''
            if Ne == 2:
                Plot_ = Plot
            else:
                Plot_ = False
            u, V = solve_diffusion(u_D, f, alpha, rho, T, Nx, Ny, Nz, Nt, 
                                   Ne, dim, Plot=Plot)
            
            # Compute error at vertices
            u_e = interpolate(u_D, V)
            error = np.abs(u_e.vector().array() - u.vector().array()).max()
            print 'P{} element, {} dimensions, Error: '.format(Ne, dim), error
            
            success = error < tolerance
            msg = 'The script is apparently not running as it should'
            assert success, msg


def test_linear_solution(T, Nz, tolerance = 1e-1, Plot=False):
    '''
    Exercise e
    With a exponential fluctuating solution u(x,y,t) = exp(-pi^2*t) cos(pi*x)
    we expect rho and
    alpha to be constants and f to be constant zero
    independent of the values of C, rho and alpha
    '''
    rho = Constant(1.0)
    alpha = rho
    f = Constant(0.0)
    I = Expression('cos(pi*x[0])', degree=3)
    u_D  = Expression('exp(-pi*pi*t) * cos(pi*x[0])', degree=2, t=T)
    h_list = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]

    print '\nLinear solution'
    for dim in [1]:
        '''Find the solution for P1 elements studying 1D, 2D and 3D. 
        dx = dy = dt = sqrt(h)  =>  Nx = Ny = Nt = h^(-1/2)'''

        for h in h_list:
            if h == h_list[-1]:
                Plot_ = Plot
            else:
                Plot_ = False
                
            Nx = Ny = Nt = int(round(h**(-0.5)))        # Steps
            Ne = 1                                      # P1 element
            u, V = solve_diffusion(I, f, alpha, rho, T, Nx, Ny, Nz, Nt, 
                                   Ne, dim, Plot=Plot_)
            
            # Compute error at vertices
            u_e = interpolate(u_D, V)
            e = u_e.vector().array() - u.vector().array()
            E = np.sqrt(np.sum(e**2)/u.vector().array().size)
            print '{} dimensions, h = {}, Error = {}, E/h = {}'\
                  .format(dim, h, E, E/h)
            
            success = E < tolerance
            msg = 'The script is apparently not running as it should'
            assert success, msg
    
    
def test_nonlinear_solution(T, Nx, Ny, Nz, Nt, tolerance = 1, Plot = False):
    '''
    Exercise f
    A nonlinear solution u(x,y,t) = tx^2(1/2-x/3)
    '''
    rho = Constant(1.0)
    I = Constant(0.0)
    alpha = Expression('1 + u*u', degree=2, u=I)
    f = Expression('-rho*pow(x[0], 3)/3 + rho*pow(x[0], 2)/2 + 8*pow(t, 3)*'
                    'pow(x[0],7)/9 - 28*pow(t, 3)*pow(x[0], 6)/9 + 7*pow(t, 3)*'
                    'pow(x[0], 5)/2 - 5*pow(t, 3)*pow(x[0], 4)/4 + 2*t*x[0] - t', 
                    degree = 2, rho = rho, t=0)
                    
    u_D = Expression('t*x[0]*x[0]*(0.5 - x[0]/3)', degree = 2, t=0)
    
    u, V = solve_diffusion(I, f, alpha, rho, T, Nx, Ny, Nz, Nt, 1, 1, Plot=Plot)
    u_e = interpolate(u_D, V)
    e = u_e.vector().array() - u.vector().array()
    print '\nNon-linear solution \nError: {}'.format(np.max(abs(e)))
    
    msg = 'The non-linear numerical solution has too big error'
    assert np.max(abs(e)) < tolerance, msg
    
    
if __name__ == '__main__':
    pass

set_log_active(False)

test_constant_solution(T=1.0, Nx=10, Ny=10, Nz=10, Nt=10, Plot=False)
test_linear_solution(T=1.0, Nz=10, Plot=False)
test_nonlinear_solution(T=1.0, Nx=10, Ny=10, Nz=10, Nt=100, Plot=True)

#plot_exact(f = u_linear, Nx = 100, T = 1.0)
plot_exact(f = u_non_linear, Nx = 100, T = 1.0)

'''
terminal >>> python2 project_2.py

Constant solution
P1 element, 1 dimensions, Error:  2.77555756156e-15
P1 element, 2 dimensions, Error:  3.5527136788e-15
P1 element, 3 dimensions, Error:  2.60902410787e-14
P2 element, 1 dimensions, Error:  1.26343380202e-13
P2 element, 2 dimensions, Error:  1.39666056498e-13
P2 element, 3 dimensions, Error:  8.69304628281e-14

Linear solution
1 dimensions, h = 0.001, Error = 3.1101672842e-06, E/h = 0.0031101672842
1 dimensions, h = 0.0005, Error = 1.58817190084e-06, E/h = 0.00317634380168
1 dimensions, h = 0.0001, Error = 3.23369943115e-07, E/h = 0.00323369943115
1 dimensions, h = 5e-05, Error = 1.62653483093e-07, E/h = 0.00325306966185
1 dimensions, h = 1e-05, Error = 3.23584499113e-08, E/h = 0.00323584499113

Non-linear solution 
Error: 0.167165830348
'''
