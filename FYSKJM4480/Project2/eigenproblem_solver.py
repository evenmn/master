import numpy as np
import matplotlib.pyplot as plt

def M_FCI(g):
    M = np.matrix([[2 - g, -0.5 * g, -0.5 * g, -0.5 * g, -0.5 * g, 0], \
                   [-0.5 * g, 4 - g, -0.5 * g, -0.5 * g, 0, -0.5 * g], \
                   [-0.5 * g, -0.5 * g, 6 - g, 0, -0.5 * g, -0.5 * g], \
                   [-0.5 * g, -0.5 * g, 0, 6 - g, -0.5 * g, -0.5 * g], \
                   [-0.5 * g, 0, -0.5 * g, -0.5 * g, 8 - g, -0.5 * g], \
                   [0, -0.5 * g, -0.5 * g, -0.5 * g, -0.5 * g, 10 - g]])
    return M

def M_CID(g):
    M = np.matrix([[2 - g, -0.5 * g, -0.5 * g, -0.5 * g, -0.5 * g], \
                   [-0.5 * g, 4 - g, -0.5 * g, -0.5 * g, 0], \
                   [-0.5 * g, -0.5 * g, 6 - g, 0, -0.5 * g], \
                   [-0.5 * g, -0.5 * g, 0, 6 - g, -0.5 * g], \
                   [-0.5 * g, 0, -0.5 * g, -0.5 * g, 8 - g]])
    return M

def eigenvalue_solver(M, g_list):
    eigenvalues = []                    # Going to be a nested list
    for g in g_list:       
        eigenvalues.append((np.linalg.eigh(M(g))[0]))
    
    eigenvalues_sorted = []
    for k in range(M(g).shape[0]):         # M.shape[0] is length(col1(M_FCI))
        new_list = []
        for i in range(len(eigenvalues)):
            new_list.append(eigenvalues[i][k])
        eigenvalues_sorted.append(new_list)
    return eigenvalues_sorted
          

if __name__ == '__main__':
    pass

g_list = np.linspace(-1, 1, 100)        # 100 g values

# -- FCI energy --
eigenvalues = eigenvalue_solver(M_FCI, g_list)
for i in range(len(eigenvalues)):
    plt.plot(g_list, eigenvalues[i])
plt.title('FCI energy')
plt.xlabel('g')
plt.ylabel('Energy')
plt.grid()
plt.show()

# -- CID energy --
eigenvalues = eigenvalue_solver(M_CID, g_list)
for i in range(len(eigenvalues)):
    plt.plot(g_list, eigenvalues[i])
plt.title('CID energy')
plt.xlabel('g')
plt.ylabel('Energy')
plt.grid()
plt.show()
