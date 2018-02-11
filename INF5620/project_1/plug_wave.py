from solver import *

def test_pulse(Lx, Ly, T, dt, dx, dy):
    '''
    This function sends a pulse in x-direction and tests if 
    the pulse only propagates in x-direction
    '''
    b = 0
    c = 1.0
    j_pulse = 0         # The index in y-dir where we send the pulse
    Nx = int(round(Lx/dx)); Ny = int(round(Ly/dy))
    Y = np.linspace(0, Ly, Ny + 1)

    I       = lambda x, y:    1 if y == Y[j_pulse] and x < Nx/2. + sigma \
                              and x > Nx/2. - sigma else 0
    V       = lambda x, y:    0
    q       = lambda x, y:    c
    f       = lambda x, y, t: 0

    sigma = 2       #Length of pulse (number of cells)
    u, cpu = solver(f, q, I, V, b, Lx, Ly, T, dt, dx, dy,
                    version = 'vector', stability_safety_factor=1.0)

    for i in range(Nx + 1):
        for j in range(Ny + 1):
            msg = 'test_pulse failed'
            assert u[i, j] <= 1, msg
