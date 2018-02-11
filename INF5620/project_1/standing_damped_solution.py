import numpy as np
from solver import *
from plot import *
from convergence_rate import *

def test_standing_damped(Lx, Ly, T, dt, dx, dy):
    '''
    In this function we will look at a standing damped
    wave and check if the convergence rate approaches 2, 
    and give a error message if it doesn't. 
    '''
    mx = 5                          # Number of nodes in x-dir
    my = 5                          # Number of nodes in y-dir
    kx = mx * np.pi/Lx              # Wave number in x-dir
    ky = my * np.pi/Ly              # Wave number in y-dir
    w = np.sqrt(kx**2 + ky**2)      # k = wc, k = sqrt(kx^2 + ky^2)
    A = 1.0
    B = 1.0
    c = 1.0
    b = 1.0

    u_exact = lambda x, y, t: (A * np.cos(w * t) + B * np.sin(w * t)) * np.exp(-c * t) * np.cos(kx * x) * np.cos(ky * y)
    I       = lambda x, y:    u_exact(x, y, 0)
    V       = lambda x, y:    (w * B - c * A) * np.cos(kx * x) * np.cos(ky * y)
    q       = lambda x, y:    1 + (x - Lx/2.)**4 + (y - Ly/2.)**4
    dqdx    = lambda x:       4 * (x - Lx/2.)**3
    dqdy    = lambda y:       4 * (y - Ly/2.)**3
    f       = lambda x, y, t: (c**2 - w**2 + kx**2 + ky**2 - b*c) * u_exact(x, y, t) - c * (2 + b) * \
                              (-A * np.sin(w * t) + B * np.cos(w * t)) * np.exp(-c * t) * np.cos(kx * x) * np.cos(ky * y) \
                              - dqdx(x) - dqdy(y)

    u, cpu = solver(f, q, I, V, b, Lx, Ly, T, dt, dx, dy,
                    version = 'vector', stability_safety_factor = 1.0)
    
    plot(u, u_exact, 0, 'Standing damped wave', Lx, dy, T)

    print '\n === Standing damped wave ==='
    r = convergence_rate(u_exact, I, V, q, f, b, Lx, Ly, T, dt)
    assert r > -1 and r < 2

'''
Output:
 === Standing damped wave ===
h = 0.5000,    E = 2.44493,
h = 0.2500,    E = 1.75152,    r = 0.4811874999
h = 0.1000,    E = 1.17767,    r = 0.4332061748
h = 0.0750,    E = 0.98987,    r = 0.6038626143
h = 0.0500,    E = 0.88618,    r = 0.2729185937
h = 0.0250,    E = 0.92701,    r = -0.0649910675
h = 0.0100,    E = 1.15850,    r = -0.2432815108
h = 0.0075,    E = 1.22835,    r = -0.2035092713
h = 0.0050,    E = 1.31574,    r = -0.1694966571
h = 0.0025,    E = 1.41679,    r = -0.1067512312
'''
