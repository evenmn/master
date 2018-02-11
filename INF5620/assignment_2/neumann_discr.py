import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import time

def solver(f, q, U_0, L, T, dt, dx, exercise,
           stability_safety_factor=1.0):

    Nt = int(round(T/dt))
    t = np.linspace(0, T, Nt+1)         # Mesh points in time

    Nx = int(round(L/dx))
    x = np.linspace(0, L, Nx+1)         # Mesh points in space

    # Check stability
    if max(np.sqrt(q(x))*(dt/dx)) > stability_safety_factor:
        print "StabilityWarning: The solution is unstable due to Courant number" 

    C = (dt/dx)**2; dt2 = dt * dt       # Help variables in the scheme

    u = np.zeros([Nx+1, Nt+1])          # Matrix for storing all u-values

    t0 = time.clock()                   # CPU time measurement

    Ix = range(0, Nx+1)
    It = range(0, Nt+1)

    def u_neumann(i, n, exercise):
        if exercise == "a":
            if n == 1:
                if i == 0:
                    result = u[0, 0]  + C * q(x[0])  * (u[1, 0] - u[0, 0]) \
                             + 0.5 * dt2 * f(x[0],  t[0])
                elif i == Nx:
                    result = u[Nx, 0] + C * q(x[Nx]) * (u[Nx-1, 0] - u[Nx, 0]) \
                             + 0.5 * dt2 * f(x[Nx], t[0])
                else:
                    print "Error: Cannot use Neumann conditions on this index"
            else:
                if i == 0:
                    result = - u[0, n-1] + 2 * u[0, n] + 2 * C * q(x[0]) \
                             * (u[1, n] - u[0, n]) + dt2 * f(x[0], t[n])
                elif i == Nx:
                    result = - u[Nx, n-1] + 2 * u[Nx, n] + 2 * C * q(x[Nx]) \
                             * (u[Nx-1, n] - u[Nx, n]) + dt2 * f(x[Nx], t[n])
                else:
                    print "Error: Cannot use Neumann conditions on this index"

        elif exercise == "b":
            if n == 1:
                if i == 0:
                    result = u[0, 0]  + 0.5 * C * (q(x[1]) - q(x[0])) * (u[1, 0] - u[0, 0]) \
                             + 0.5 * dt2 * f(x[0],  t[0])
                elif i == Nx:
                    result = u[Nx, 0]  + 0.5 * C * (q(x[Nx-1]) - q(x[Nx])) * (u[Nx-1, 0] - u[Nx, 0]) \
                             + 0.5 * dt2 * f(x[Nx],  t[0])
                else:
                    print "Error: Cannot use Neumann conditions on this index"
            else:
                if i == 0:
                    result = - u[0, n-1] + 2 * u[0, n] + C * (q(x[1]) + q(x[0])) \
                             * (u[1, n] - u[0, n]) + dt2 * f(x[0], t[n])
                elif i == Nx:
                    result = - u[Nx, n-1] + 2 * u[Nx, n] + C * (q(x[Nx-1]) + q(x[0])) \
                             * (u[Nx-1, n] - u[Nx, n]) + dt2 * f(x[Nx], t[n])
                else:
                    print "Error: Cannot use Neumann conditions on this index"

        elif exercise == "c":
            if n == 1:
                if i == 0:
                    result = u[0, 0]  + 0.25 * C * (q(x[0]) + q(x[1])) * (u[1, 0] - u[0, 0]) \
                             + 0.5 * dt2 * f(x[0],  t[0])
                elif i == Nx:
                    result = u[Nx, 0]  + 0.25 * C * (q(x[Nx]) + q(x[Nx-1])) * (u[Nx-1, 0] - u[Nx, 0]) \
                             + 0.5 * dt2 * f(x[Nx],  t[0])
                else:
                    print "Error: Cannot use Neumann conditions on this index"
            else:
                if i == 0:
                    result = - u[0, n-1] + 2 * u[0, n] + 0.5 * C * (q(x[1]) + q(x[0])) \
                             * (u[1, n] - u[0, n]) + dt2 * f(x[0], t[n])
                elif i == Nx:
                    result = - u[Nx, n-1] + 2 * u[Nx, n] + 0.5 * C * (q(x[Nx-1]) + q(x[0])) \
                             * (u[Nx-1, n] - u[Nx, n]) + dt2 * f(x[Nx], t[n])
                else:
                    print "Error: Cannot use Neumann conditions on this index"

        elif exercise == "d":
            if n == 1:
                if i == 0:
                    result = u[0, 0]  + 0.25 * C * (q(x[0]) + q(x[1])) * (u[1, 0] - u[0, 0]) \
                             + 0.5 * dt2 * f(x[0],  t[0])
                elif i == Nx:
                    result = u[Nx, 0]  + 0.25 * C * (q(x[Nx]) + q(x[Nx-1])) * (u[Nx-1, 0] - u[Nx, 0]) \
                             + 0.5 * dt2 * f(x[Nx],  t[0])
                else:
                    print "Error: Cannot use Neumann conditions on this index"
            else:
                if i == 0:
                    result = - u[0, n-1] + 2 * u[0, n] + 0.5 * C * (q(x[1]) + q(x[0])) \
                             * (u[1, n] - u[0, n]) + dt2 * f(x[0], t[n])
                elif i == Nx:
                    result = - u[Nx, n-1] + 2 * u[Nx, n] + 0.5 * C * (q(x[Nx-1]) + q(x[0])) \
                             * (u[Nx-1, n] - u[Nx, n]) + dt2 * f(x[Nx], t[n])
                else:
                    print "Error: Cannot use Neumann conditions on this index"

        else:
            print 'Error: No exercise with name "%s"' % exercise
    
        return result

    # --- Initial conditions based on exact solution ---
    for i in Ix:
        u[i, 0] = U_0(x[i])


    # --- Special formula for the case when t=1 ---
    # First calculate the midpoints (all points except for x=0 and x=Nx)
    for i in Ix[1:-1]:
        u[i, 1] = u[i, 0] + 0.25 * C * ((q(x[i]) + q(x[i+1])) * (u[i+1, 0] - u[i, 0]) \
                  - (q(x[i]) + q(x[i-1])) * (u[i, 0] - u[i-1, 0])) + 0.5 * dt2 * f(x[i], t[0])
    # Then calculate for x=0 and x=Nx (Neumann conditions)
    u[0, 1] = u_neumann(0, 1, exercise)
    u[Nx, 1] = u_neumann(Nx, 1, exercise)

    
    # --- Loop over all the other elements ---
    for n in It[1:-1]:
        # Calculate all point in the interval x=[1,Nx-1], t=[2,Nt]
        for i in Ix[1:-1]:
            u[i, n+1] = - u[i, n-1] + 2 * u[i, n] + 0.5 * C * ((q(x[i]) + q(x[i+1])) * (u[i+1, n] - u[i, n]) \
                  - (q(x[i]) + q(x[i-1])) * (u[i, n] - u[i-1, n])) + dt2 * f(x[i], t[n])
        # Then calculate for x=0 and x=Nx, t=[2,Nt] (Neumann conditions)
        u[0, n+1] = u_neumann(0, n, exercise)
        u[Nx, n+1] = u_neumann(Nx, n, exercise)
    
    cpu_time = t0 - time.clock()

    return u, cpu_time

# --- Run solver ---
'''
L, x, t, k, w    = sym.symbols('L x t k w')
u_exact = sym.cos(k * x) * sym.cos(w * t)
q       = 1 + (x - L/2.)**4
dqdx    = sym.diff(q,x)
f       = sym.diff(sym.diff(u_exact, t), t) - sym.diff(q * sym.diff(u_exact, x), x)
'''
L       = 1           # Size
T       = 1           # Total time
dt      = 0.001       # Time step
dx      = 0.01        # Spatial step
k       = np.pi/L     # Wavenumber
w       = 1.0         # Angular frequency

u_exact = lambda x, t: np.cos(k * x) * np.cos(w * t)
U_0     = lambda x: u_exact(x, 0)

# Exercise A
exercise = "a"
q       = lambda x: 1 + (x - L/2.)**4
dqdx    = lambda x: 4 * (x - L/2.)**3
f       = lambda x, t: dqdx(x) * k * np.sin(k * x) * np.cos(w * t) \
                       + (q(x) * k**2 - w**2) * np.cos(k * x) * np.cos(w * t)

u, cpu = solver(f, q, U_0, L, T, dt, dx, exercise,
                stability_safety_factor=1.0)
'''
# Exercise B
exercise = "b"
q       = lambda x: 1 + np.cos(k * x)
dqdx    = lambda x: - k * np.sin(k * x)
f       = lambda x, t: dqdx(x) * k * np.sin(k * x) * np.cos(w * t) \
                       + (q(x) * k**2 - w**2) * np.cos(k * x) * np.cos(w * t)

u, cpu = solver(f, q, U_0, L, T, dt, dx, exercise,
                stability_safety_factor=1.0)

# Exercise C
exercise = "c"

u, cpu = solver(f, q, U_0, L, T, dt, dx, exercise,
                stability_safety_factor=1.0)

# Exercise D
exercise = "d"

u, cpu = solver(f, q, U_0, L, T, dt, dx, exercise,
                stability_safety_factor=1.0)
'''

# --- Measurements ---
T_measure = 0.9
x = np.linspace(0, L, int(round(L/dx)) + 1)
'''
for dt_ in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
    u, cpu = solver(f, q, U_0, L, T, dt_, dx, exercise,
                    stability_safety_factor=1.0)
    error = abs(u[:, T_measure/dt_] - u_exact(x, T_measure))
    print dt_, '  ', sum(error), sum(error)/len(error)
'''

SZ = {'size':'16'}
plt.plot(x, u[:, T_measure/dt])
plt.plot(x, u_exact(x, T_measure))
plt.title('Method A, T=0.9',**SZ)
plt.xlabel('x-values, $x$',**SZ)
plt.ylabel('u-values, $u(x)$',**SZ)
plt.grid()
plt.legend(["Numerical", "Exact"],loc="best")
plt.show()
