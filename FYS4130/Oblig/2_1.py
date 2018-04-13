import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

mu = 1.0
muBH = 0.7

def N(kT, sign):
    return (mu + sign * muBH)**(3/2)*(1 + (np.pi**2/8)*(kT/(mu + sign * muBH))**2)

T = np.linspace(0, 4, 100)

sns.set()
label_size = {'size':'14'}
plt.plot(T, N(T, +1) + N(T, -1), linewidth=2.5, label=r"$\langle N\rangle$")
plt.plot(T, N(T, -1), label=r"$\langle N_-\rangle$")
plt.plot(T, N(T, +1), label=r"$\langle N_+\rangle$")
plt.legend(loc='best', prop=label_size)
plt.xlabel("Temperature, $T$", **label_size)
plt.ylabel("Average number of particles", **label_size)
plt.savefig("2_1.png")
plt.show()
