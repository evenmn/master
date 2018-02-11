import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

# --- Use sympy to do the symbolic calculations ---
V, t, I, w, b, dt = sym.symbols('V t I w b dt')  # global symbols
f = None  # global variable for the source term in the ODE

def ode_source_term(u):
    # Return the terms in the ODE, here u'' + w**2*u.
    return sym.diff(u(t), t, t) + w**2*u(t)

def residual_discrete_eq(u):
    # Return the residual of the discrete eq. with u inserted.
    R = ode_source_term(u) - (DtDt(u, dt) + w**2*u(t))
    return sym.simplify(R)

def residual_discrete_eq_step1(u):
    # Return the residual of the discrete eq. at the first step with u inserted.
    f0 = (w**2)*I # f = w^2*u(t), u(1) = I
    u1 = I + 0.5*(dt**2)*(f0 - (w**2)*I) + V*dt
    R = u(dt) - u1
    return sym.simplify(R)

def DtDt(u, dt):
    # Return 2nd-order finite difference for u_tt.
    u_tt = (u(t + dt) - 2*u(t) + u(t - dt))/(dt**2)
    return u_tt

def main(u):
    """
    Given some chosen solution u (as a function of t, implemented
    as a Python function), use the method of manufactured solutions
    to compute the source term f, and check if u also solves
    the discrete equations.
    """
    print '\n === Testing exact solution: u = Vt + I ==='
    print 'Initial conditions u(0)=%s, Dt(u(0))=%s:' % (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0))

    # Method of manufactured solution requires fitting f
    global f  # source term in the ODE
    f = sym.simplify(ode_source_term(u))

    # Residual in discrete equations (should be 0)
    print 'residual step1:', residual_discrete_eq_step1(u)
    print 'residual:', residual_discrete_eq(u)

def linear():
    main(lambda t: V*t + I)

if __name__ == '__main__':
    linear()

# --- Quadratic function ---
def residual_discrete_eq_step1_quadratic(u):
    # Return the residual of the discrete eq. at the first step with u inserted.
    f0 = (w**2)*I + 2*b  # f = 2*b + w**2*u(t)
    u1 = I + 0.5*(dt**2)*(f0 - (w**2)*I) + V*dt
    R = u(dt) - u1
    return sym.simplify(R)

def main_quadratic(u):
    # Method of manufactured solution requires fitting f
    # Residual in discrete equations (should be 0)
    print '\n === Testing exact solution: u = bt^2 + Vt + I ==='
    print 'residual step1:', residual_discrete_eq_step1_quadratic(u)
    print 'residual:', residual_discrete_eq(u)

def quadratic():
    main_quadratic(lambda t: b*(t**2) + V*t + I)

if __name__ == '__main__':
    quadratic()

# --- Computing the numerical solution ---
def solver(I, V, b, w, dt, T, f):
    """
    Solve u(t)'' + w^2u(t) = f(t) for t in (0,T],
    Initial: u(0) = I and u'(0) = V,
    by a central finite difference method with time step dt.
    """  
    N = int(round(T/dt))
    t = np.zeros(N + 2)
    u = np.zeros(N + 2)

    u[0] = I
    u[1] = u[0] + dt * V + (dt**2/2) * (f(t[0]) - (w**2) * u[0])
    t[1] = dt
 
    for n in range(N):
        t[n+2] = (n + 2) * dt
        u[n+2] = 2 * u[n+1] - u[n] + (dt**2) * (f(t[n+1]) - (w**2) * u[n+1])
        print 't = %.2f,   u = %.3f' % (t[n], u[n])      
    return u

# --- Calling function ---
I = 1.0
V = 1.0
b = 1.0
w = 1.0
dt = 0.01
T = 2.0
f_linear = lambda t: (V*t + I)*(w**2)
f_quad =  lambda t: (b*(t**2) + V*t + I)*(w**2) + 2*b

# --- Printing results ---
print '\n === Computing the numerical solution ==='
print 'Linear: u = ct + d'
print '------------------'
Linear = solver(I, V, b, w, dt, T, f_linear)

print '\n Quadric: u = bt^2 + ct + d'
print '---------------------------'
Quadratic = solver(I, V, b, w, dt, T, f_quad)

# --- Plotting results ---
label_size = {'size':'16'}
plt.plot(np.linspace(0, int(round(T/dt)) * dt, int(round(T/dt)) + 2), Linear)
plt.plot(np.linspace(0, int(round(T/dt)) * dt, int(round(T/dt)) + 2), Quadratic)
plt.legend(['Linear', 'Quadratic'])
plt.xlabel('Time, t', **label_size)
plt.ylabel('u(t)', **label_size)
plt.grid()
plt.show()

# --- Nose test: solver vs. sympy ---
def test_solver():
    tol = 1e-1
    N = int(round(T/dt))
    t = np.linspace(0, N * dt, N + 2)
    expected = b * (t**2) + V * t + I
    computed = Quadratic
    
    for i in range(N):
        success = abs(expected[i] - computed[i]) < tol
        msg = 'Numerical solution of u is incorrect'
    assert success, msg
test_solver()
