import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def L_avg(T, k=1, epsilon=1):
    e_plus = np.exp(epsilon/(k*T))
    e_minus = np.exp(-epsilon/(k*T))
    
    return e_minus*e_minus*((e_plus + 2*e_minus)/(e_plus + e_minus))
    
T = np.linspace(0, 100, 10000)

sns.set()
label_size={'size':'14'}
plt.plot(T, L_avg(T))
plt.xlabel("Temperature, $T\cdot(k_B/\epsilon)$", **label_size)
plt.ylabel(r"$\langle L\rangle$", **label_size)
plt.savefig("1_4.png")
plt.show()
