import numpy as np
import matplotlib.pyplot as plt

def ccd_energy(g_list, N, sigma=0.0):
    def e(p, xi=1):
        return xi*(p-1)
       
    def t_mat(g, N, sigma):
        t = np.zeros([N, 2, 2])
        t[0] = 0                    # Initial guess

        for n in range(N-1):      
            for i in range(2):
                for a in range(2):
                    t[n+1] = (2*(e(i+1) -0.5*g - e(a+3)) \
                             + sigma)**(-1)*(sigma*t[n][i][a] \
                             - 0.5*g*(1+t[n][i][0] + t[n][i][1] \
                             + t[n][0][a] + t[n][1][a]+t[n][0][0] \
                             * t[n][1][1]+t[n][0][1]*t[n][1][0] \
                             - t[n][i][a]*(t[n][0][0] + t[n][0][1] \
                             + t[n][1][0] + t[n][1][1])))
        return t[N-1]

    energies = []
    for g in g_list:
        last_term = -0.5*g*np.sum(t_mat(g, N, sigma))
        energies.append(2*e(1) + 2*e(2) - g + last_term)
    return energies

if __name__ == '__main':
    pass

g_list = np.linspace(-1, 1, 100)
energy = ccd_energy(g_list=g_list, N=100, sigma=0.2)

plt.plot(g_list, energy)
plt.title('CCD energy')
plt.xlabel('g')
plt.ylabel('Energy')
plt.grid()
plt.savefig('CCD_energy.png')
plt.show()
