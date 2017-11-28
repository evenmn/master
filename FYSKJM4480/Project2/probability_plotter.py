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

def probability_finder(M, g_list):
    prob = []
    for g in g_list:
        eigenvalues, eigenvectors = np.linalg.eigh(M(g))
        prob.append(np.sum(np.square(eigenvectors[:1,:1])))
    return prob

if __name__ == '__main__':
    pass
    
g_list = np.linspace(-1, 1, 100)        # 100 g values
     
# -- FCI probability --
probability = probability_finder(M_FCI, g_list)
plt.plot(g_list, probability)
plt.title('FCI probability of finding system in unperturbed groundstate')
plt.xlabel('g')
plt.ylabel('Probability')
plt.grid()
plt.savefig('probability_FCI.png')
plt.show()

# -- CID probability --
probability = probability_finder(M_CID, g_list)
plt.plot(g_list, probability)
plt.title('CID probability of finding system in unperturbed groundstate')
plt.xlabel('g')
plt.ylabel('Probability')
plt.grid()
plt.savefig('probability_CID.png')
plt.show()
