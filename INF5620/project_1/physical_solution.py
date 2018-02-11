import numpy as np
from solver import *
from plot import *

def physical(B_i, Lx, Ly, T, dt, dx, dy):
    '''
    Simulate wave with different shaped bottoms
    Takes the bottom shape as argument
    B_i element of [gaussian, cosine-hat, box]
    '''
    g       = 9.81              # Acceleration of gravity
    I0      = 1.1
    Ia      = 0.4
    Im      = Lx/4.
    Is      = 0.2
    B0      = 1.0
    Ba      = 1.0
    Bmx     = 0.0
    Bmy     = 0.0
    Bs      = 1.0
    c       = 1.0               # Scaling parameter
    b       = 0.0               # Damping constant
    h       = 2.0               # Maximum depth
    
    I       = lambda x, y:    I0 + Ia * np.exp(-((x-Im)/Is)**2)
    V       = lambda x, y:    0

    if B_i == 'gaussian':
        B   = lambda x, y:    B0 + Ba * np.exp(-((x-Bmx)/Bs)**2-((y-Bmy)/(c * Bs))**2)

    elif B_i == 'cosine-hat':
        B   = lambda x, y:    B0 + Ba * np.cos(np.pi * (x - Bmx)/(2 * Bs)) * np.cos(np.pi * (y - Bmy)/(2 * Bs))

    elif B_i == 'box':
        B   = lambda x, y:    B0 + Ba if y <= (Bmy - Bs) and y >= Bmy + Bs else 0

    else:
        raise NameError("Invalid argument, '%s'. Please use 'gaussian', 'cosine-hat' or 'box'" % B_i)
          
    H       = lambda x, y:    h - B(x, y)
    q       = lambda x, y:    g * H(x, y)
    f       = lambda x, y, t: 0

    for T_plot in [dt, 10, 100]:#, 0.6, 0.9]:
        u, cpu = solver(f, q, I, V, b, Lx, Ly, T_plot, dt, dx, dy,
                        version = 'vector', stability_safety_factor=1.0)
        plot(u = u, ue = 0, j = 0, title = 'Physical wave', Lx = Lx, dy = dy, T = T_plot)
    return None
