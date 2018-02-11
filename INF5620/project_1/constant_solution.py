from solver import *

def test_constant(Lx, Ly, T, dt, dx, dy):
    '''
    If we set u_exact to c, u should always be c
    test_constant() is testing for c=1
    '''
    b       = 1.0
    c       = 1.0
    u_exact = lambda x, y, t: c
    I       = lambda x, y:    c
    V       = lambda x, y:    0
    q       = lambda x, y:    1 + (x - Lx/2.)**4 + (y - Ly/2.)**4
    dqdx    = lambda x:       4 * (x - Lx/2.)**3
    dqdy    = lambda y:       4 * (y - Ly/2.)**3
    f       = lambda x, y, t: 0

    u, cpu = solver(f, q, I, V, b, Lx, Ly, T, dt, dx, dy,
                    version = 'vector', stability_safety_factor=1.0)

    for i in xrange(int(round(Lx/dx)) + 1):
        for j in xrange(int(round(Ly/dy)) + 1):
            msg = 'test_constant failed'
            assert u[i, j] == c, msg
