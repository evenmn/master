from CCD_energy import ccd_energy
from eigenproblem_solver import M_FCI, M_CID, eigenvalue_solver
from RSPT_energy import rspt3_energy
import matplotlib.pyplot as plt
from numpy import linspace

g_list = linspace(-1, 1, 100)

# --- Comparison of FCI and CID ground-state energy ---
plt.plot(g_list, eigenvalue_solver(M_FCI, g_list)[0], label='FCI GS')
plt.plot(g_list, eigenvalue_solver(M_CID, g_list)[0], label='CID GS')
plt.title('FCI and CID ground-state comparison')
plt.xlabel('g')
plt.ylabel('Energy')
plt.savefig('groundstate_comparison.png')
plt.legend(loc='best')
plt.show()

# --- Comparation of FCI, RSPT3 and CCD groundstate energy ---
plt.plot(g_list, eigenvalue_solver(M_FCI, g_list)[0], label='FCI')
plt.plot(g_list, rspt3_energy(g_list), label='RSPT3')
plt.plot(g_list, ccd_energy(g_list, 100), label='CCD')
plt.title('FCI, RSPT3 and CCD ground-state comparison')
plt.xlabel('g')
plt.ylabel('Energy')
plt.legend(loc='best')
plt.grid()
plt.savefig('total_comparison.png')
plt.show()
