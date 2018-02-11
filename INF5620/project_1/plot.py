import matplotlib.pyplot as plt
from numpy import linspace

def plot(u, ue, j, title, Lx, dy, T):
    x = linspace(0, Lx, len(u))
    plt.plot(x, u[:,j])
    if ue != 0:
        plt.plot(x, ue(x, j*dy, T))
    plt.title('%s,  T=%.2f' % (title, T))
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend(['Numerical','Exact'],loc='best')
    plt.grid()
    plt.show()
