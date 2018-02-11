import numpy as np
import matplotlib.pyplot as plt
import time
import pylab

def solver(f, q, I, V, b, Lx, Ly, T, dt, dx, dy,
           stability_safety_factor=1.0):

    Nt = int(round(T/dt))
    t = np.linspace(0, T, Nt+1)         # Mesh points in time

    Nx = int(round(Lx/dx))
    x = np.linspace(0, Lx, Nx+3)         # Mesh points in x-dir

    Ny = int(round(Ly/dy))
    y = np.linspace(0, Ly, Ny+3)         # Mesh points in y-dir
    '''
    # Check stability
    if max(np.sqrt(q(x,y))*(dt/dx)) > stability_safety_factor:
        print "StabilityWarning: The solution is unstable due to Courant number" 
    '''

    dtx = (dt/dx)**2
    dty = (dt/dy)**2
    dt2 = dt * dt                       # Help variables in the scheme

    u = np.zeros([Nx+3, Ny+3, Nt+1])          # Matrix for storing all u-values

    t0 = time.clock()                   # CPU time measurement

    Ix = range(1, Nx+2)
    Iy = range(1, Ny+2)
    It = range(1, Nt)    

    # --- Initial conditions based on exact solution ---
    for i in xrange(Nx+3):
        for j in xrange(Ny+3):
            u[i][j][0] = I(x[i], y[j])


    # --- Special formula for the case when t=1 ---
    # Calculate u for the midpoints
    for i in Ix:
        for j in Iy:
            u[i, j, 1] = u[i, j, 0] + (1 - 0.5 * b * dt) * dt * V(x[i], y[j]) + \
                         0.25 * (dtx * ((q(x[i], y[j]) + q(x[i+1], y[j])) * (u[i+1, j, 0] - u[i, j, 0]) \
                         - (q(x[i-1], y[j]) + q(x[i], y[j])) * (u[i, j, 0] - u[i-1, j, 0])) \
                         + dty * ((q(x[i], y[j]) + q(x[i], y[j+1])) * (u[i, j+1, 0] - u[i, j, 0]) \
                         - (q(x[i], y[j-1]) + q(x[i],y[j])) * (u[i, j, 0] - u[i, j-1, 0]))) + 0.5 * dt2 * f(x[i], y[j], t[0])

    # Create ghost cells
    for i in Ix:
        u[i][0][1] = u[i][2][1]
        u[i][Nx+2][1] = u[i][Nx][1]
    for j in Iy:
        u[0][j][1] = u[2][j][1]
        u[Ny+2][j][1] = u[Ny][j][1]

    # --- Calculate for t=[2,Nt] ---
    # Inner points
    for n in It:
        for i in Ix:
            for j in Iy:
                u[i, j, n+1] = 2 * u[i, j, n] - u[i, j, n-1] - 0.5 * b * dt * (u[i, j, n] - u[i, j, n-1]) \
                             + 0.5 * (dtx * ((q(x[i], y[j]) + q(x[i+1], y[j])) * (u[i+1, j, n] - u[i, j, n]) \
                             - (q(x[i-1], y[j]) + q(x[i], y[j])) * (u[i, j, n] - u[i-1, j, n])) \
                             + dty * ((q(x[i], y[j]) + q(x[i], y[j+1])) * (u[i, j+1, n] - u[i, j, n]) \
                             - (q(x[i], y[j-1]) + q(x[i],y[j])) * (u[i, j, n] - u[i, j-1, n]))) + dt2 * f(x[i], y[j], t[n])

        # Create ghost cells
        for i in Ix:
            u[i][0][n+1] = u[i][2][n+1]
            u[i][Nx+2][n+1] = u[i][Nx][n+1]
        for j in Iy:
            u[0][j][n+1] = u[2][j][n+1]
            u[Ny+2][j][n+1] = u[Ny][j][n+1]

    cpu_time = t0 - time.clock()
    return u, cpu_time


def physical(B_i):
    '''
    Simulate wave with different shaped bottoms
    Takes the bottom shape as argument
    B_i element of [gaussian, cosine-hat, box]
    '''
    g       = 9.81              # Acceleration of gravity
    I0      = 1.0
    Ia      = 1.0
    Im      = 0.0
    Is      = 1.0
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
        B   = lambda x, y: B0 + Ba if y <= (Bmy - Bs) and y >= Bmy + Bs else 0

    else:
        raise NameError("Invalid argument, '%s'. Please use 'gaussian', 'cosine-hat' or 'box'" % B_i)
     

    H       = lambda x, y:    h - B(x, y)
    q       = lambda x, y:    g * H(x, y)
    f       = lambda x, y, t: 0

    u, cpu = solver(f, q, I, V, b, Lx, Ly, T, dt, dx, dy,
                    stability_safety_factor=1.0)

    for n in np.linspace(0, int(round(T/dt)), 11):
        plot(u = u, ue = 0, j = 6, n = int(n))


def plot(u, ue, j, n):
    v = []
    for i in range(len(u[1][:])-2):
        v.append(u[i+1][j][n])

    x = np.linspace(0, Lx, int(round(Lx/dx)) + 3)
    plt.plot(v)
    if ue != 0:
        plt.plot(ue(x[1:-1],j*dy,n*dt))
    plt.title('T=%.2f' % (n*dt))
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend(['Numerical','Exact'],loc='best')
    plt.show()
    return None

if __name__ == '__main__':
    pass

# --- Run solver ---
Lx      = 1           # Size in x-dir
Ly      = Lx          # Size in y-dir
T       = 1           # Total time
dt      = 0.005       # Time step
dx      = 0.1        # Spatial step x-dir
dy      = dx          # Spatial step y-dir
b       = 1.0         # Damping constant
k       = 1.0         # Wave number

physical('box')
