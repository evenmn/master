import numpy as np
import matplotlib.pyplot as plt


g_list = np.linspace(-1, 1, 100)        # 100 g values

# --- Full Configuration-Interaction (FCI) ---
eigenvalues_FCI = []                    # Going to be a nested list (matrix)
                                        # with eigenvalues for each g

for g in g_list:
    M_FCI = np.matrix([[-2 - g, -0.5 * g, -0.5 * g, -0.5 * g, -0.5 * g, 0], \
                       [-0.5 * g, -4 - g, -0.5 * g, -0.5 * g, 0, -0.5 * g], \
                       [-0.5 * g, -0.5 * g, -6 - g, 0, -0.5 * g, -0.5 * g], \
                       [-0.5 * g, -0.5 * g, 0, -6 - g, -0.5 * g, -0.5 * g], \
                       [-0.5 * g, 0, -0.5 * g, -0.5 * g, -8 - g, -0.5 * g], \
                       [0, -0.5 * g, -0.5 * g, -0.5 * g, -0.5 * g, -10 - g]])
                       
    eigenvalues_FCI.append((np.linalg.eigh(M_FCI)[0]))

for k in range(M_FCI.shape[0]):         # M_FCI.shape[0] is length(col1(M_FCI))
    new_list_FCI = []
    for i in range(len(eigenvalues_FCI)):
        new_list_FCI.append(eigenvalues_FCI[i][k])
    plt.plot(g_list, new_list_FCI, label='Eigenvalue {}'.format(k))
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
    M_CID = np.matrix([[-2-g, 0, 0], [0, -4-g, -0.5*g], [0, -0.5*g, -10-g]])
                       
    eigenvalues_CID.append((np.linalg.eigh(M_CID)[0]))

for k in range(M_CID.shape[0]):         # M_CID.shape[0] is length(col1(M_CID))
    new_list_CID = []
    for i in range(len(eigenvalues_CID)):
        new_list_CID.append(eigenvalues_CID[i][k])
    plt.plot(g_list, new_list_CID, label='Eigenvalue {}'.format(k))
plt.legend(loc='best')
plt.title('Eigenvalues')
plt.xlabel('g')
plt.ylabel('Energy')
plt.savefig('eigenvalues_CID.png')
plt.show()

# --- Comparison of ground-state energy ---

plt.plot(g_list, new_list_FCI, label='FCI GS')
plt.plot(g_list, new_list_CID, label='CID GS')
plt.title('FCI and CID ground-state comparison')
plt.xlabel('g')
plt.ylabel('Energy')
plt.savefig('groundstate_comparison.png')
plt.legend(loc='best')
plt.show()
