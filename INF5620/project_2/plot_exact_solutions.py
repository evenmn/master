import matplotlib.pyplot as plt
import numpy as np

def u_constant(x, t):
    return 1.0

def u_linear(x, t):
    return np.exp(-np.pi**2*t)*np.cos(np.pi*x)

def u_non_linear(x, t):
    return t*x**2*(0.5 - x/3.)
    
def plot_exact(f, Nx, T):
    x = np.linspace(0, 1, Nx + 1)
    t = T                           # Plot the final function
    plt.plot(x, f(x, t))
    plt.xlabel('x')
    plt.ylabel('$u_e(x, t=T)$')
    plt.grid()
    plt.show()
