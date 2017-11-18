from sympy import *
import numpy as np
import matplotlib.pyplot as plt


g_list = np.linspace(-1, 1, 100)
eigenvalues = []

for g in g_list:
    M = np.matrix([[-2 - g, -0.5 * g, -0.5 * g, -0.5 * g, -0.5 * g, 0], \
                   [-0.5 * g, -4 - g, -0.5 * g, -0.5 * g, 0, -0.5 * g], \
                   [-0.5 * g, -0.5 * g, -6 - g, 0, -0.5 * g, -0.5 * g], \
                   [-0.5 * g, -0.5 * g, 0, -6 - g, -0.5 * g, -0.5 * g], \
                   [-0.5 * g, 0, -0.5 * g, -0.5 * g, -8 - g, -0.5 * g], \
                   [0, -0.5 * g, -0.5 * g, -0.5 * g, -0.5 * g, -10 - g]])
    eigenvalues.append((np.linalg.eigh(M)[0]))

for k in range(M.shape[0]):
    new_list = []
    for i in range(len(eigenvalues)):
        new_list.append(eigenvalues[i][k])
    plt.plot(g_list, new_list, label='Eigenvalue {}'.format(k))
plt.legend(loc='best')
plt.show()
