import numpy as np
import matplotlib.pyplot as plt


g_list = np.linspace(-1, 1, 100)        # 100 g values

# --- Full Configuration-Interaction (FCI) ---
eigenvalues_FCI = []                    # Going to be a nested list (matrix)
                                        # with eigenvalues for each g

for g in g_list:
    M_FCI = np.matrix([[2 - g, -0.5 * g, -0.5 * g, -0.5 * g, -0.5 * g, 0], \
                       [-0.5 * g, 4 - g, -0.5 * g, -0.5 * g, 0, -0.5 * g], \
                       [-0.5 * g, -0.5 * g, 6 - g, 0, -0.5 * g, -0.5 * g], \
                       [-0.5 * g, -0.5 * g, 0, 6 - g, -0.5 * g, -0.5 * g], \
                       [-0.5 * g, 0, -0.5 * g, -0.5 * g, 8 - g, -0.5 * g], \
                       [0, -0.5 * g, -0.5 * g, -0.5 * g, -0.5 * g, 10 - g]])
                       
    eigenvalues_FCI.append((np.linalg.eigh(M_FCI)[0]))

for k in reversed(range(M_FCI.shape[0])):         # M_FCI.shape[0] is length(col1(M_FCI))
    new_list_FCI = []
    for i in range(len(eigenvalues_FCI)):
        new_list_FCI.append(eigenvalues_FCI[i][k])
    if k == 0:
        lw = 3
    else:
        lw = 1
    plt.plot(g_list, new_list_FCI, label='k={}'.format(k), linewidth=lw)
plt.legend(loc='best')
plt.title('Eigenvalues')
plt.xlabel('g')
plt.ylabel('Energy')
plt.savefig('eigenvalues_FCI.png')
plt.show()

# --- Configuration-Interaction Doubles (CID) ---
eigenvalues_CID = []                        # Going to be a nested list (matrix)
                                        # with eigenvalues for each g

for g in g_list:
    M_CID = np.matrix([[2 - g, -0.5 * g, -0.5 * g, -0.5 * g, -0.5 * g], \
                       [-0.5 * g, 4 - g, -0.5 * g, -0.5 * g, 0], \
                       [-0.5 * g, -0.5 * g, 6 - g, 0, -0.5 * g], \
                       [-0.5 * g, -0.5 * g, 0, 6 - g, -0.5 * g], \
                       [-0.5 * g, 0, -0.5 * g, -0.5 * g, 8 - g]])
                       
    eigenvalues_CID.append((np.linalg.eigh(M_CID)[0]))

for k in reversed(range(M_CID.shape[0])):         # M_CID.shape[0] is length(col1(M_CID))
    new_list_CID = []
    for i in range(len(eigenvalues_CID)):
        new_list_CID.append(eigenvalues_CID[i][k])
    plt.plot(g_list, new_list_CID, label='k={}'.format(k))
plt.legend(loc='best')
plt.title('Eigenvalues')
plt.xlabel('g')
plt.ylabel('Energy')
plt.savefig('eigenvalues_CID.png')
plt.show()
