import matplotlib.pyplot as plt
from numpy import linspace

def rspt3_energy(g):
    return 2 - g**1 - (25/48)*g**2 - (231/1152)*g**3

if __name__ == '__main':
    pass

g_list = linspace(-1, 1, 100)

plt.plot(g_list, rspt3_energy(g_list))
plt.title('RSPT3 groundstate energy')
plt.xlabel('g')
plt.ylabel('Energy')
plt.grid()
plt.savefig('RSPT_energy.png')
plt.show()
