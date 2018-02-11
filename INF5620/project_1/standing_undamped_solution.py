import numpy as np
from solver import *
from plot import *
from convergence_rate import *

def test_standing_undamped(Lx, Ly, T, dt, dx, dy):
    '''
    In this function we will look at a standing undamped
    wave and check if the convergence rate approaches 2, 
    and give a error message if it doesn't. 
    '''
    mx = 3
    my = 3
    kx = (mx * np.pi)/Lx            # Wave number in x-dir
    ky = (my * np.pi)/Ly            # Wave number in y-dir
    w  = np.sqrt(kx**2 + ky**2)     # w = kc, k = sqrt(kx^2 + ky^2)
    A  = 1.0
    b  = 0
    C  = 1.0

    u_exact = lambda x, y, t: A * np.cos(kx * x) * np.cos(kx * y) * np.cos(w * t)
    I       = lambda x, y:    u_exact(x, y, 0)
    V       = lambda x, y:    0
    q       = lambda x, y:    C
    f       = lambda x, y, t: 0

    u, cpu = solver(f, q, I, V, b, Lx, Ly, T, dt, dx, dy,
                    version = 'vector', stability_safety_factor = 1.0)
    plot(u, u_exact, 0, 'Standing undamped wave', Lx, dy, T)
    
    print '\n === Standing undamped wave ==='
    r = convergence_rate(u_exact, I, V, q, f, b, Lx, Ly, T, dt)
    assert r > -1 and r < 2

'''
Output:
 === Standing undamped wave ===
h = 0.5000,    E = 0.75910,
h = 0.2500,    E = 0.86823,    r = -0.1937888039
h = 0.1000,    E = 0.94406,    r = -0.0913777815
h = 0.0750,    E = 0.93149,    r = 0.0465952553
h = 0.0500,    E = 0.82551,    r = 0.2978780875
h = 0.0250,    E = 0.51665,    r = 0.6760971821
h = 0.0100,    E = 0.18833,    r = 1.1013769574
h = 0.0075,    E = 0.12489,    r = 1.4277737072
h = 0.0050,    E = 0.06146,    r = 1.7488635717
h = 0.0025,    E = 0.02407,    r = 1.3525336227
'''
