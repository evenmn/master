import numpy as np
from solver import *

def convergence_rate(u_exact, I, V, q, f, b, Lx, Ly, T, dt):
    E = []
    h_list = [5e-1, 2.5e-1, 1e-1, 7.5e-2, 5e-2, 2.5e-2, 1e-2, 7.5e-3, 5e-3, 2.5e-3]#, 1e-3]
    for h in h_list:
        dx_ = h; dy_ = h; dt_ = h/4.
        Nx = int(round(Lx/dx_)); Ny = int(round(Ly/dy_)); Nt = int(round(T/dt_))
        
        u, cpu = solver(f, q, I, V, b, Lx, Ly, T, dt_, dx_, dy_,
                        version = 'vector', stability_safety_factor = 1.0)
        
        x = np.linspace(0, Lx, Nx + 1)
        y = np.linspace(0, Ly, Ny + 1)

        e = 0
        for i in xrange(Nx + 1):
            for j in xrange(Ny + 1):
                e_new = abs(u_exact(x[i], y[j], T) - u[i,j])
                if e_new > e:
                    e = e_new           # True error

        E.append(e)                     # Total error

    print 'h = %.4f,    E = %.5f,' % (h_list[0], E[0])
    for i in xrange(len(h_list) - 1):
        r = np.log(E[i+1]/E[i])/np.log((dt/h_list[i])/(dt/h_list[i+1]))
        print 'h = %.4f,    E = %.5f,    r = %.10f' % (h_list[i+1], E[i+1], r)
    return r
