import numpy as np
import matplotlib.pyplot as plt
import time

def solver(f, q, I, V, b, Lx, Ly, T, dt, dx, dy,
           version = 'vector', stability_safety_factor=1.0):

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

    It = range(1, Nt)

    if version == 'scalar':
        u = np.zeros([Nx+3, Ny+3, Nt+1])          # Matrix for storing all u-values

        Ix = range(1, Nx+2)
        Iy = range(1, Ny+2)

        t0 = time.clock()                   # CPU time measurement

        # --- Initial conditions based on exact solution ---
        for i in xrange(Nx+3):
            for j in xrange(Ny+3):
                u[i][j][0] = I(x[i], y[j])

        # --- Special formula for the case when t=1 ---
        # Calculate u for the midpoints
        for i in Ix:
            for j in Iy:
                u[i, j, 1] = u[i, j, 0] + (1 - 0.5 * b * dt) * dt * V(x[i], y[j]) \
                             + 0.25 * (dtx * ((q(x[i], y[j]) + q(x[i+1], y[j])) \
                             * (u[i+1, j, 0] - u[i, j, 0]) - (q(x[i-1], y[j]) \
                             + q(x[i], y[j])) * (u[i, j, 0] - u[i-1, j, 0])) \
                             + dty * ((q(x[i], y[j]) + q(x[i], y[j+1])) \
                             * (u[i, j+1, 0] - u[i, j, 0]) - (q(x[i], y[j-1]) \
                             + q(x[i],y[j])) * (u[i, j, 0] - u[i, j-1, 0]))) \
                             + 0.5 * dt2 * f(x[i], y[j], t[0])

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
                    u[i, j, n+1] = 2 * u[i, j, n] - u[i, j, n-1] - b * dt \
                                   * (u[i, j, n] - u[i, j, n-1]) + 0.5 * (dtx \
                                   * ((q(x[i], y[j]) + q(x[i+1], y[j])) \
                                   * (u[i+1, j, n] - u[i, j, n]) - (q(x[i-1], y[j]) \
                                   + q(x[i], y[j])) * (u[i, j, n] - u[i-1, j, n])) \
                                   + dty * ((q(x[i], y[j]) + q(x[i], y[j+1])) \
                                   * (u[i, j+1, n] - u[i, j, n]) - (q(x[i], y[j-1]) \
                                   + q(x[i], y[j])) * (u[i, j, n] - u[i, j-1, n]))) \
                                   + dt2 * f(x[i], y[j], t[n])
                    
            # Create ghost cells
            for i in Ix:
                u[i][0][n+1] = u[i][2][n+1]
                u[i][Nx+2][n+1] = u[i][Nx][n+1]
            for j in Iy:
                u[0][j][n+1] = u[2][j][n+1]
                u[Ny+2][j][n+1] = u[Ny][j][n+1]

        v = np.empty((Nx + 3, Ny + 3))
        for i in xrange(len(u)):
            v[i] = u[i][:,0]
        u = v

    elif version == 'vector':
        u1 = np.zeros([Nx + 3, Ny + 3])     # u_{i,j}^{n+1}
        u2 = np.zeros([Nx + 3, Ny + 3])     # u_{i,j}^{n-1}
        u  = np.zeros([Nx + 3, Ny + 3])     # u_{i,j}^{n}

        t0 = time.clock()                   # CPU time measurement

        # --- Initial conditions based on exact solution ---
        for i in xrange(Nx + 3):
            for j in xrange(Ny + 3):
                u[i, j] = I(x[i], y[j])
 
        # --- Special formula for the case when t=1 ---
        # Calculate u for the midpoints
        u1[1:-1, 1:-1] = u[1:-1, 1:-1] + (1 - 0.5 * b * dt) * dt * V(x[1:-1], y[1:-1]) \
                         + 0.25 * (dtx * ((q(x[1:-1], y[1:-1]) + q(x[2:], y[1:-1])) \
                         * (u[2:, 1:-1] - u[1:-1, 1:-1]) - (q(x[:-2], y[1:-1]) \
                         + q(x[1:-1], y[2:])) * (u[1:-1, 2:] - u[1:-1, 1:-1]) \
                         - (q(x[1:-1], y[:-2]) + q(x[1:-1], y[1:-1])) * (u[1:-1, 1:-1] \
                         - u[1:-1, :-2]))) + 0.5 * dt2 * f(x[1:-1], y[1:-1], t[0])
        

        # Create ghost cells
        u1[1:-1, 0] = u1[1:-1, 2]
        u1[1:-1, Nx+2] = u1[1:-1, Nx]
        u1[0, 1:-1] = u1[2, 1:-1]
        u1[Ny+2, 1:-1] = u1[Ny, 1:-1]
        
        u2 = u; u = u1
        
        
        # --- Calculate for t=[2,Nt] ---
        # Inner points
        for n in It:
            u1[1:-1, 1:-1] = 2 * u[1:-1, 1:-1] - u2[1:-1, 1:-1] - b * dt \
                             * (u[1:-1, 1:-1] - u2[1:-1, 1:-1]) + 0.5 * (dtx \
                             * ((q(x[1:-1], y[1:-1]) + q(x[2:], y[1:-1])) \
                             * (u[2:, 1:-1] - u[1:-1, 1:-1]) - (q(x[:-2], y[1:-1]) \
                             + q(x[1:-1], y[1:-1])) * (u[1:-1, 1:-1] - u[:-2, 1:-1])) \
                             + dty * ((q(x[1:-1], y[1:-1]) + q(x[1:-1], y[2:])) \
                             * (u[1:-1, 2:] - u[1:-1, 1:-1]) - (q(x[1:-1], y[:-2]) \
                             + q(x[1:-1], y[1:-1])) * (u[1:-1, 1:-1] - u[1:-1, :-2]))) \
                             + dt2 * f(x[1:-1], y[1:-1], t[n])
            

            # Create ghost cells
            u1[1:-1, 0] = u1[1:-1, 2]
            u1[1:-1, Nx+2] = u1[1:-1, Nx]
            u1[0, 1:-1] = u1[2, 1:-1]
            u1[Ny+2, 1:-1] = u1[Ny, 1:-1]
            
            u2 = u; u = u1; u1 = u2

    else:
        raise NameError("Invalid operation method, please use 'vector' or 'scalar'")
    
    cpu_time = time.clock() - t0
    
    return u[1:-1, 1:-1], cpu_time
